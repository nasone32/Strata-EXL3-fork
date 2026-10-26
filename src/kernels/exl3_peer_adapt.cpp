#include "strata/kernels/exl3_peer_adapt.hpp"
#include "strata/kernels/exl3_experts.hpp"
#include "strata/artifact/exl3_model.hpp"
#include "strata/core/peer_adapt_plan.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace strata::kernels {
struct Exl3PeerAdapt::Impl {
    Exl3Peer& peer;
    Exl3ExpertStream& primary;
    strata::Exl3Model model;
    int layers, experts;
    std::vector<uint8_t>& owned;
    Options options;
    std::vector<uint8_t> incoming;
    std::vector<float> usage;
    std::vector<int32_t> primary_res, peer_res;
    std::vector<Exl3Peer::Replacement> batch, landed;
    int64_t windows = 0, rounds = 0, submitted = 0, published = 0, bytes = 0, incompatible = 0, disjoint_checks = 0;
    double submit_ms = 0, wait_ms = 0;
    Impl(Exl3Peer& p, Exl3ExpertStream& s, const std::string& path, int l, int e,
          std::vector<uint8_t>& own, Options opt)
        : peer(p), primary(s), model(path), layers(l), experts(e), owned(own), options(opt),
          incoming((size_t) l * e, 0), usage((size_t) l * e, 0) {}

    bool poll(bool wait, std::string& err) {
        if (!peer.valid()) {
            std::fill(owned.begin(), owned.end(), 0);
            std::fill(incoming.begin(), incoming.end(), 0);
            primary.set_pending_elsewhere(nullptr);
            return true; // the callback's full-primary failure fallback remains available
        }
        const auto start = std::chrono::steady_clock::now();
        if (!peer.apply_refill(wait, landed, err)) return false;
        if (wait) wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        for (const auto& r : landed) {
            const size_t i = (size_t) r.in.layer * experts + r.in.expert;
            if (primary.holds(r.in.layer, r.in.expert)) {
                err = "exl3 peer adapt: an incoming peer expert was admitted to the primary"; return false;
            }
            owned[i] = 1; incoming[i] = 0;
            ++published;
        }
        return true;
    }
};

Exl3PeerAdapt::Exl3PeerAdapt(Exl3Peer& peer, Exl3ExpertStream& primary, const std::string& model,
                             int layers, int experts, std::vector<uint8_t>& owned, Options options) {
    if (layers <= 0 || experts <= 0 || owned.size() != (size_t) layers * experts ||
        options.every <= 0 || options.max_swaps <= 0 || !std::isfinite(options.decay) ||
        options.decay < 0 || options.decay > 1 || !std::isfinite(options.min_gain) || options.min_gain < 0)
        throw std::invalid_argument("exl3 peer adapt: invalid dimensions/options");
    auto m = std::make_unique<Impl>(peer, primary, model, layers, experts, owned, options);
    m->batch.reserve(options.max_swaps); m->landed.reserve(options.max_swaps);
    std::string err;
    if (!peer.enable_adaptation(options.max_swaps, options.byte_cap, err)) throw std::runtime_error(err);
    primary.set_pending_elsewhere(m->incoming.data());
    impl_ = std::move(m);
    primary.set_route_observer([](void* p, int l, const int32_t* ids, int n) {
        ((Exl3PeerAdapt*) p)->observe(l, ids, n);
    }, this);
    std::fprintf(stderr, "strata EXL3 peer adapt: every=%d windows swaps=%d decay=%g margin=%g cap=%.2f MiB "
                 "(0=swap limit only); resident and in-flight ownership disjoint\n",
                 options.every, options.max_swaps, options.decay, options.min_gain,
                 (double) options.byte_cap / (1<<20));
}

Exl3PeerAdapt::~Exl3PeerAdapt() {
    if (!impl_) return;
    std::string err;
    if (!drain(err)) std::fprintf(stderr, "strata EXL3 peer adapt drain: %s\n", err.c_str());
    auto& m = *impl_;
    m.primary.set_route_observer(nullptr, nullptr);
    m.primary.set_pending_elsewhere(nullptr);
    std::fprintf(stderr, "strata EXL3 peer adapt summary: %lld windows %lld rounds %lld submitted %lld landed, "
                 "%.2f MiB uploaded, %.2f ms selection/pack/enqueue, %.2f ms blocking drain, "
                 "%.2f ms GPU refill span, %lld incompatible slot candidates, %lld disjoint snapshots checked\n",
                 (long long) m.windows, (long long) m.rounds,
                 (long long) m.submitted, (long long) m.published, (double) m.bytes/(1<<20),
                 m.submit_ms, m.wait_ms, m.peer.refill_ms(), (long long) m.incompatible, (long long) m.disjoint_checks);
}

void Exl3PeerAdapt::observe(int layer, const int32_t* ids, int routes) {
    auto& m = *impl_;
    if (layer < 0 || layer >= m.layers) return;
    for (int i = 0; i < routes; ++i)
        if (ids[i] >= 0 && ids[i] < m.experts) m.usage[(size_t) layer * m.experts + ids[i]] += 1.f;
}

bool Exl3PeerAdapt::before_window(std::string& err) {
    auto& m = *impl_;
    if (!m.poll(false, err)) return false;
    const bool due = m.windows > 0 && m.windows % m.options.every == 0;
    ++m.windows;
    if (!due || !m.peer.valid()) return true;
    ++m.rounds;
    const auto start = std::chrono::steady_clock::now();
    if (!m.peer.refilling()) {
        m.primary.residency(m.layers, m.experts, m.primary_res);
        m.peer.residency(m.layers, m.experts, m.peer_res);
        for (size_t i = 0; i < m.primary_res.size(); ++i)
            if (m.primary_res[i] >= 0 && (m.peer_res[i] >= 0 || m.incoming[i])) {
                err = "exl3 peer adapt: primary and peer ownership overlap"; return false;
            }
        ++m.disjoint_checks;
        const auto plan = strata::core::peer_adapt_plan(m.usage.data(), m.primary_res.data(),
            m.peer_res.data(), m.layers, m.experts, m.options.min_gain);
        m.batch.clear();
        size_t bytes = 0;
        for (const auto& s : plan) {
            if ((int) m.batch.size() >= m.options.max_swaps) break;
            Exl3Peer::Replacement r; r.out = s.out; r.in.layer = s.layer; r.in.expert = s.in;
            const std::string b = "model.language_model.layers." + std::to_string(s.layer) +
                                  ".mlp.experts." + std::to_string(s.in) + ".";
            auto load = [&](const char* name) {
                const auto l = m.model.linear(b + name);
                return Exl3Mat{l.trellis, l.suh, l.svh, l.ki, l.nj, l.bits, l.cb};
            };
            r.in.g = load("gate_proj"); r.in.u = load("up_proj"); r.in.d = load("down_proj");
            const size_t cost = m.peer.replacement_bytes(r);
            if (!cost) { ++m.incompatible; continue; }
            if (m.options.byte_cap && cost > m.options.byte_cap - bytes) continue;
            bytes += cost; m.batch.push_back(r);
        }
        if (!m.batch.empty()) {
            if (!m.peer.replace(m.batch, err)) return false;
            for (const auto& r : m.batch) {
                m.owned[(size_t) r.in.layer * m.experts + r.out] = 0;
                m.incoming[(size_t) r.in.layer * m.experts + r.in.expert] = 1;
            }
            m.submitted += m.batch.size(); m.bytes += bytes;
        }
    }
    for (float& u : m.usage) u *= m.options.decay;
    m.submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
    return true;
}

bool Exl3PeerAdapt::drain(std::string& err) { return impl_->poll(true, err); }
int64_t Exl3PeerAdapt::swaps() const { return impl_->published; }
int64_t Exl3PeerAdapt::upload_bytes() const { return impl_->bytes; }
} // namespace strata::kernels
