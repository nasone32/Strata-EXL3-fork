// src/mimo/mimo_mtp_parity.cpp - GPU MiMo v2 MTP drafter (strata::mimo::MimoMtp) vs the CPU
// reference, notes/MIMO-V2-PORT.md §1 "MTP" / §2 "MTP drafter".
//
// Synthetic F16 weights, no model, at a scaled-down geometry the kernels accept: n_embd 512, 3 heads,
// n_q 8 / n_kv 8 (GQA group 1 of the compiled {1,2,4,8,16}), head_dim_qk 192 / head_dim_v 128,
// n_rot 64, window 128, inter 1024, theta 1e4, eps 1e-6, capacity 2048.  The CPU reference below is
// deliberately independent of the module: fp64 arithmetic end to end, written from the notes (the
// eh input order [enorm(embed); hnorm(hidden)], the exact norm sites, the window INCLUDING the query,
// the RAW sink, silu, F32 residual accumulation) while mirroring every F16 rounding point of the GPU
// path (projection inputs and attention outputs are F16 images; residuals and states are F32), so the
// printed errors measure the GPU's fp32 arithmetic alone.
//
// Cases: a 6-round chain from a 60-cell base (the accept pattern 3,3,2,0,1,3 crosses the 128-key
// window edge mid-chain and includes a full rejection), a 200-cell base with TWO full rejections from
// the same anchor (speculative rows rolled back and overwritten), a 1596-cell base (positions >
// 1500, catch_up chunked past max_rows, the window long past), heads with and without a sink (head
// 0's sinks partly dominate the softmax max, head 1's are mild, head 2 has none), and the K/V-only
// catch_up path vs explicit full forwards (identical drafts - a committed cell's head state is never
// read by the chain).
//
// TOLERANCES.  The gate is finite outputs, norm_rel < 2e-2, max_abs < 0.15 and every draft pick
// equal to the reference's fp64 argmax (a near-tie within the fp32 sums counts).  max_rel is PRINTED
// but not gated: with random weights the head states are deep cancellation chains (residual terms
// of a few against channels of a tenth), and one F16-boundary flip between the GPU's fp32
// arithmetic and the oracle's fp64 - a value that rounds one ulp differently inside any of the ten
// F16 images the pipeline takes - amplifies through attention and the residual sums to 10-30% of an
// ISOLATED channel while the row as a whole stays at 1e-3..1e-2 (the module's own two arithmetic
// paths disagree by the same magnitudes on the same channels: one cell's channel measured 0.219 by
// the 1-row gemv path against 0.182 by the multi-row GEMM path, oracle 0.225).  Composition errors
// (a swapped norm weight, the hidden half first, the anchor state where a cell's own belongs) move
// whole rows and sit at 10-100x the gate.  The window edge, the RAW sink and the rope are pinned to
// 2e-3 by mimo_attn_parity AT THE KERNEL; this test pins the drafter's composition on top of them.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "strata/artifact/mimo_pack.hpp"
#include "strata/core/mimo_geometry.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/mimo/mtp.hpp"

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

// --------------------------------------------------------------------------- geometry

struct Geom {
    int n = 512, nq = 8, nkv = 8, hdq = 192, hdv = 128, nrot = 64, window = 128, inter = 1024;
    int vocab = 291, heads = 3;
    int64_t capacity = 2048, max_rows = 512;
    double theta = 1e4, eps = 1e-6;
    int qk_dim() const { return nq * hdq; }
    int kv_dim() const { return nkv * hdq; }
    int v_dim() const { return nq * hdv; }
};

strata::core::MimoGeometry full_geometry(const Geom& q) {
    strata::core::MimoGeometry g;
    g.n_embd = q.n;
    g.n_layers = 4;
    g.n_vocab = q.vocab;
    g.rms_eps = q.eps;
    g.n_q_head = q.nq;
    g.head_dim_qk = q.hdq;
    g.head_dim_v = q.hdv;
    g.n_rot = q.nrot;
    g.n_head_kv_full = q.nkv;
    g.n_head_kv_swa = q.nkv;
    g.rope_theta_full = 1e7;
    g.rope_theta_swa = q.theta;
    g.sliding_window = q.window;
    g.dense_intermediate = q.inter;
    g.n_mtp_layers = q.heads;
    g.layer_kind.assign((size_t) g.n_layers, 1);
    g.moe_layer.assign((size_t) g.n_layers, 1);
    return g;
}

// --------------------------------------------------------------------------- fp64 reference

struct HostHead {
    std::vector<uint16_t> eh, q, k, v, o, gate, up, down;   // F16 [out][in], the pack layout
    std::vector<float> hnorm, enorm, na, nm, nf;            // F32
    std::vector<float> sink;                                // F32 [nq] when has_sink
    bool has_sink = false;
    std::vector<uint16_t> kc, vc;   // the head's KV: [capacity][nkv][hdq] / [capacity][nkv][hdv] F16
};

struct RefModel {
    Geom q;
    std::vector<uint16_t> embed, lm_head;   // F16 [vocab][n]
    std::vector<HostHead> heads;

    static double f64_of(uint16_t h) { return (double) f32_from_f16(h); }
    static uint16_t f16_of(double x) { return f16_from_f32((float) x); }
    static void rms64(double* v, const float* w, int64_t n, double eps) {
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) s += v[i] * v[i];
        const double inv = 1.0 / std::sqrt(s / (double) n + eps);
        for (int64_t i = 0; i < n; ++i) v[i] = v[i] * w[i] * inv;
    }
    static std::vector<double> gemv64(const std::vector<uint16_t>& w, const uint16_t* x, int n_out, int n_in) {
        std::vector<double> y((size_t) n_out, 0.0);
        for (int o = 0; o < n_out; ++o) {
            double acc = 0.0;
            const uint16_t* row = w.data() + (size_t) o * n_in;
            for (int i = 0; i < n_in; ++i) acc += f64_of(row[i]) * f64_of(x[i]);
            y[(size_t) o] = acc;
        }
        return y;
    }
    /// NEOX partial rope over a [heads][dim] F16 row set at one absolute position (fp64 angle+trig).
    static void rope_row(uint16_t* row, int64_t pos, int heads, int dim, int n_rot, double theta) {
        const int half = n_rot / 2;
        for (int h = 0; h < heads; ++h) {
            uint16_t* r = row + (size_t) h * dim;
            for (int i = 0; i < half; ++i) {
                const double inv = std::pow(theta, -2.0 * i / n_rot);
                const double ang = std::fmod((double) pos * inv, 2.0 * 3.14159265358979323846);
                const double c = std::cos(ang), s = std::sin(ang);
                const double a = f64_of(r[i]), b = f64_of(r[i + half]);
                r[i] = f16_from_f32((float) (a * c - b * s));
                r[i + half] = f16_from_f32((float) (a * s + b * c));
            }
        }
    }

    /// The pre-attention part of one cell: the eh input [enorm(embed(tok)); hnorm(hidden)]
    /// (embedding FIRST), eh_proj, norm_attn, the q/k/v projections as F16 images, rope on q and k,
    /// the k/v rows appended at the absolute position (v trimmed to its first hdv lanes per head).
    /// x_out (the eh_proj output) and q16_out (nq*hdq, roped) are written when non-null.
    void pre_row_x(HostHead& H, int32_t tok, const float* hidden, int64_t p, uint16_t* q16_out, double* x_out) {
        const int n = q.n;
        std::vector<double> cat((size_t) 2 * n), h((size_t) n);
        for (int i = 0; i < n; ++i) {
            cat[(size_t) i] = f64_of(embed[(size_t) tok * n + i]);
            cat[(size_t) n + i] = (double) hidden[i];
        }
        rms64(cat.data(), H.enorm.data(), n, q.eps);
        rms64(cat.data() + n, H.hnorm.data(), n, q.eps);
        std::vector<uint16_t> cat16((size_t) 2 * n);
        for (int i = 0; i < 2 * n; ++i) cat16[(size_t) i] = f16_of(cat[(size_t) i]);
        auto x = gemv64(H.eh, cat16.data(), n, 2 * n);
        if (x_out) std::copy(x.begin(), x.end(), x_out);
        for (int i = 0; i < n; ++i) h[(size_t) i] = x[(size_t) i];
        rms64(h.data(), H.na.data(), n, q.eps);
        std::vector<uint16_t> h16((size_t) n);
        for (int i = 0; i < n; ++i) h16[(size_t) i] = f16_of(h[(size_t) i]);
        if (q16_out) {
            auto qd = gemv64(H.q, h16.data(), q.qk_dim(), n);
            for (int i = 0; i < q.qk_dim(); ++i) q16_out[i] = f16_of(qd[(size_t) i]);
            rope_row(q16_out, p, q.nq, q.hdq, q.nrot, q.theta);
        }
        std::vector<uint16_t> kk((size_t) q.kv_dim()), vv((size_t) q.kv_dim());
        auto kd = gemv64(H.k, h16.data(), q.kv_dim(), n);
        auto vd = gemv64(H.v, h16.data(), q.kv_dim(), n);
        for (int i = 0; i < q.kv_dim(); ++i) {
            kk[(size_t) i] = f16_of(kd[(size_t) i]);
            vv[(size_t) i] = f16_of(vd[(size_t) i]);
        }
        rope_row(kk.data(), p, q.nkv, q.hdq, q.nrot, q.theta);
        std::memcpy(&H.kc[(size_t) p * q.kv_dim()], kk.data(), (size_t) q.kv_dim() * 2);
        for (int kvh = 0; kvh < q.nkv; ++kvh)
            for (int d = 0; d < q.hdv; ++d)
                H.vc[((size_t) p * q.nkv + kvh) * q.hdv + d] = vv[(size_t) (kvh * q.hdq + d)];
    }

    /// The rest of one cell given its x (eh_proj output, fp64) and roped q16: attention over the
    /// head's cache (window INCLUDING the query - every key <= p is real - sink RAW in the
    /// denominator), o_proj + F32 residual, the dense SwiGLU MLP + residual, final_layernorm.
    void cell_rest(HostHead& H, const double* x_in, const uint16_t* q16, int64_t p, double* state) {
        const int n = q.n, group = q.nq / q.nkv;
        std::vector<double> x(x_in, x_in + n);
        std::vector<uint16_t> attn16((size_t) q.v_dim());
        const int lo = (int) std::max<int64_t>(0, p - (q.window - 1));
        for (int h = 0; h < q.nq; ++h) {
            const int kvh = h / group;
            const uint16_t* qrow = q16 + (size_t) h * q.hdq;
            double m = -std::numeric_limits<double>::infinity();
            std::vector<double> s((size_t) (p - lo + 1));
            for (int64_t j = lo; j <= p; ++j) {
                const uint16_t* krow = &H.kc[(size_t) j * q.kv_dim() + (size_t) kvh * q.hdq];
                double dot = 0.0;
                for (int d = 0; d < q.hdq; ++d) dot += f64_of(qrow[d]) * f64_of(krow[d]);
                dot *= 1.0 / std::sqrt((double) q.hdq);
                s[(size_t) (j - lo)] = dot;
                m = std::max(m, dot);
            }
            const double sink = H.has_sink ? (double) H.sink[(size_t) h] : -std::numeric_limits<double>::infinity();
            m = std::max(m, sink);
            double den = H.has_sink ? std::exp(sink - m) : 0.0;
            std::vector<double> num((size_t) q.hdv, 0.0);
            for (int64_t j = lo; j <= p; ++j) {
                const double w = std::exp(s[(size_t) (j - lo)] - m);
                den += w;
                const uint16_t* vrow = &H.vc[((size_t) j * q.nkv + kvh) * q.hdv];
                for (int d = 0; d < q.hdv; ++d) num[(size_t) d] += w * f64_of(vrow[d]);
            }
            for (int d = 0; d < q.hdv; ++d) attn16[(size_t) h * q.hdv + d] = f16_of(num[(size_t) d] / den);
        }
        auto o = gemv64(H.o, attn16.data(), n, q.v_dim());
        for (int i = 0; i < n; ++i) x[(size_t) i] += o[(size_t) i];
        std::vector<double> h2(x);
        rms64(h2.data(), H.nm.data(), n, q.eps);
        std::vector<uint16_t> h216((size_t) n);
        for (int i = 0; i < n; ++i) h216[(size_t) i] = f16_of(h2[(size_t) i]);
        auto g = gemv64(H.gate, h216.data(), q.inter, n);
        auto u = gemv64(H.up, h216.data(), q.inter, n);
        std::vector<uint16_t> act16((size_t) q.inter);
        for (int i = 0; i < q.inter; ++i) {
            const double a = g[(size_t) i];
            double sv = a / (1.0 + std::exp(-a)) * u[(size_t) i];
            if (sv > 65504.0) sv = 65504.0;
            else if (sv < -65504.0) sv = -65504.0;
            act16[(size_t) i] = f16_of(sv);
        }
        auto dn = gemv64(H.down, act16.data(), n, q.inter);
        for (int i = 0; i < n; ++i) x[(size_t) i] += dn[(size_t) i];
        rms64(x.data(), H.nf.data(), n, q.eps);
        std::copy(x.begin(), x.end(), state);
    }

    /// One chunk of `rows` cells at [pos0, pos0+rows) for one head: phase 1 appends k/v (and stashes
    /// each row's x and roped q when states are wanted), phase 2 runs the rest.  Rows are
    /// independent inside a phase, so both parallelise; row r's attention reads only keys <= pos0+r,
    /// which phase 1 appended for the whole chunk first - the prefill kernel's causality.
    void chunk(int head, const int32_t* tokens, const float* hidden, int rows, int64_t pos0, bool want_states,
               double* states) {
        HostHead& H = heads[(size_t) head];
        std::vector<uint16_t> q_all((size_t) rows * q.qk_dim(), 0);
        std::vector<double> x_all((size_t) rows * q.n, 0.0);
        uint16_t* q_ptr = want_states ? q_all.data() : nullptr;
        double* x_ptr = want_states ? x_all.data() : nullptr;
        par_for(rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; ++r)
                pre_row_x(H, tokens[r], hidden + (size_t) r * q.n, pos0 + r,
                          q_ptr ? q_ptr + (size_t) r * q.qk_dim() : nullptr,
                          x_ptr ? x_ptr + (size_t) r * q.n : nullptr);
        });
        if (!want_states) return;
        par_for(rows, [&](int r0, int r1) {
            std::vector<double> st((size_t) q.n);
            for (int r = r0; r < r1; ++r) {
                cell_rest(H, &x_all[(size_t) r * q.n], &q_all[(size_t) r * q.qk_dim()], pos0 + r, st.data());
                std::copy(st.begin(), st.end(), states + (size_t) r * q.n);
            }
        });
    }

    /// The draft chain (the module's protocol): step k runs heads k..2 on the previous round token
    /// (step 0: the anchor) at P + k - head k fully, the deeper heads K/V-only; every hidden input is
    /// the anchor's state.  Returns the three picks, their producing states and their fp64 logits.
    void draft(int32_t anchor_tok, const float* anchor_hidden, int64_t P, int32_t out_ids[3],
               std::vector<double> producing_states[3], std::vector<double> logits_out[3]) {
        int32_t tok = anchor_tok;
        for (int k = 0; k < q.heads; ++k) {
            for (int h = k; h < q.heads; ++h)
                chunk(h, &tok, anchor_hidden, 1, P + k, /*want_states =*/h == k,
                      h == k ? producing_states[k].data() : nullptr);
            std::vector<uint16_t> st16((size_t) q.n);
            for (int i = 0; i < q.n; ++i) st16[(size_t) i] = f16_of(producing_states[k][(size_t) i]);
            auto lg = gemv64(lm_head, st16.data(), q.vocab, q.n);
            int pick = 0;
            for (int v = 1; v < q.vocab; ++v)
                if (lg[(size_t) v] > lg[(size_t) pick]) pick = v;
            out_ids[k] = pick;
            logits_out[k] = std::move(lg);
            tok = pick;
        }
    }

    template <typename F>
    static void par_for(int rows, F&& fn) {
        if (rows <= 0) return;
        unsigned hw = std::thread::hardware_concurrency();
        const int nt = std::max(1, std::min<int>(rows, hw ? (int) hw : 8));
        if (nt == 1) {
            fn(0, rows);
            return;
        }
        std::vector<std::thread> ts;
        std::atomic<int> next{0};
        const int grain = std::max(1, rows / (nt * 4));
        for (int t = 0; t < nt; ++t)
            ts.emplace_back([&] {
                for (;;) {
                    const int r0 = next.fetch_add(grain);
                    if (r0 >= rows) break;
                    fn(r0, std::min(rows, r0 + grain));
                }
            });
        for (auto& t : ts) t.join();
    }
};

// --------------------------------------------------------------------------- comparison

struct Compare {
    double max_abs = 0.0, max_rel = 0.0, norm_rel = 0.0;
    size_t bad = 0, n = 0;
    size_t worst = 0;   // index (within the last add's buffer) of the max relative deviation
    float worst_gpu = 0.0f;
    double worst_ref = 0.0;
    void add(const float* gpu, const double* ref, size_t count) {
        double num = 0.0, den = 0.0, ref_max = 0.0;
        for (size_t i = 0; i < count; ++i) ref_max = std::max(ref_max, std::fabs(ref[i]));
        const double floor_den = 0.05 * ref_max;
        for (size_t i = 0; i < count; ++i) {
            const double g = gpu[i], b = ref[i];
            ++n;
            if (!std::isfinite(g)) { ++bad; continue; }
            const double d = std::fabs(g - b);
            num += d * d;
            den += b * b;
            max_abs = std::max(max_abs, d);
            const double rel = d / std::max(std::fabs(b), floor_den);
            if (rel > max_rel) {
                max_rel = rel;
                worst = i;
                worst_gpu = gpu[i];
                worst_ref = ref[i];
            }
        }
        norm_rel = std::max(norm_rel, std::sqrt(num) / (std::sqrt(den) + 1e-30));
    }
    bool ok() const { return bad == 0 && norm_rel < 2e-2 && max_abs < 0.15; }
};

// --------------------------------------------------------------------------- the harness

struct Harness {
    Geom q;
    RefModel ref;
    strata::MimoPack pack;
    strata::mimo::MimoMtp mtp;
    std::vector<void*> dev;
    std::mt19937 rng;

    void* upload(const void* host, size_t bytes) {
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::printf("upload: %s\n", cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
        dev.push_back(d);
        return d;
    }

    std::vector<uint16_t> rand_f16_mat(int64_t out, int64_t in) {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<uint16_t> m((size_t) out * in);
        for (auto& x : m) x = f16_from_f32(u(rng));
        return m;
    }
    std::vector<float> rand_norm(int64_t n) {
        std::uniform_real_distribution<float> u(0.5f, 1.5f);
        std::vector<float> v((size_t) n);
        for (auto& x : v) x = u(rng);
        return v;
    }

    void build(uint32_t seed) {
        rng.seed(seed);
        ref.q = q;
        ref.embed = rand_f16_mat(q.vocab, q.n);
        ref.lm_head = rand_f16_mat(q.vocab, q.n);
        ref.heads.resize((size_t) q.heads);
        pack.mtp.resize((size_t) q.heads);
        for (int h = 0; h < q.heads; ++h) {
            HostHead& H = ref.heads[(size_t) h];
            H.eh = rand_f16_mat(q.n, 2 * q.n);
            H.q = rand_f16_mat(q.qk_dim(), q.n);
            H.k = rand_f16_mat(q.kv_dim(), q.n);
            H.v = rand_f16_mat(q.kv_dim(), q.n);
            H.o = rand_f16_mat(q.n, q.v_dim());
            H.gate = rand_f16_mat(q.inter, q.n);
            H.up = rand_f16_mat(q.inter, q.n);
            H.down = rand_f16_mat(q.n, q.inter);
            H.hnorm = rand_norm(q.n);
            H.enorm = rand_norm(q.n);
            H.na = rand_norm(q.n);
            H.nm = rand_norm(q.n);
            H.nf = rand_norm(q.n);
            // head 0: sinks that partly dominate the softmax max; head 1: mild sinks; head 2: none
            H.has_sink = h < 2;
            if (H.has_sink) {
                std::uniform_real_distribution<float> u(-1.0f, 1.0f);
                H.sink.resize((size_t) q.nq);
                for (int i = 0; i < q.nq; ++i)
                    H.sink[(size_t) i] = (h == 0 && i % 4 == 0) ? 4.0f + 2.0f * u(rng) : u(rng);
            }
            H.kc.assign((size_t) q.capacity * q.kv_dim(), 0);
            H.vc.assign((size_t) q.capacity * q.nkv * q.hdv, 0);

            auto& P = pack.mtp[(size_t) h];
            P.eh = upload(H.eh.data(), H.eh.size() * 2);
            P.q = upload(H.q.data(), H.q.size() * 2);
            P.k = upload(H.k.data(), H.k.size() * 2);
            P.v = upload(H.v.data(), H.v.size() * 2);
            P.o = upload(H.o.data(), H.o.size() * 2);
            P.mlp_gate = upload(H.gate.data(), H.gate.size() * 2);
            P.mlp_up = upload(H.up.data(), H.up.size() * 2);
            P.mlp_down = upload(H.down.data(), H.down.size() * 2);
            P.hnorm = upload(H.hnorm.data(), H.hnorm.size() * 4);
            P.enorm = upload(H.enorm.data(), H.enorm.size() * 4);
            P.norm_attn = upload(H.na.data(), H.na.size() * 4);
            P.norm_mlp = upload(H.nm.data(), H.nm.size() * 4);
            P.norm_final = upload(H.nf.data(), H.nf.size() * 4);
            P.sink = H.has_sink ? upload(H.sink.data(), H.sink.size() * 4) : nullptr;
        }
        pack.embed = upload(ref.embed.data(), ref.embed.size() * 2);
        pack.lm_head = upload(ref.lm_head.data(), ref.lm_head.size() * 2);

        std::string err;
        if (!mtp.init(full_geometry(q), &pack, /*device =*/0, q.capacity, (int) q.max_rows, nullptr, err)) {
            std::printf("mimo mtp init: %s\n", err.c_str());
            std::exit(1);
        }
    }

    ~Harness() {
        for (void* p : dev) (void) cudaFree(p);
    }
};

#define CHK(e, what)                                                                              \
    do {                                                                                          \
        const cudaError_t _e = (e);                                                               \
        if (_e != cudaSuccess) {                                                                  \
            std::printf("CUDA %s: %s\n", what, cudaGetErrorString(_e));                           \
            return 1;                                                                             \
        }                                                                                         \
    } while (0)

/// One draft round's comparison: the three picks (an argmax whose fp64 margin is below the GPU's
/// fp32 rounding counts as a near-tie, not a failure) and the producing heads' states.
int compare_round(Harness& H, const char* name, int round, const int32_t* gpu_ids, const int32_t* ref_ids,
                  const std::vector<double> ref_logits[3], const std::vector<double> producing[3], Compare& c,
                  int& ties, int& pick_fails) {
    std::vector<float> st((size_t) H.q.n);
    for (int k = 0; k < 3; ++k) {
        if (gpu_ids[k] != ref_ids[k]) {
            const double top = ref_logits[k][(size_t) ref_ids[k]];
            const double got = ref_logits[k][(size_t) gpu_ids[k]];
            if (got < top - 1e-3 * (std::fabs(top) + 1.0)) {
                std::printf("%s r%d: draft %d: gpu pick %d vs ref %d (logits %.4f vs %.4f)  (FAIL)\n", name, round,
                            k, gpu_ids[k], ref_ids[k], got, top);
                ++pick_fails;
            } else {
                ++ties;
            }
        }
        CHK(cudaMemcpy(st.data(), H.mtp.head_state(k), H.q.n * 4, cudaMemcpyDeviceToHost), "state download");
        c.add(st.data(), producing[k].data(), (size_t) H.q.n);
    }
    return 0;
}

/// Builds `base` committed cells and runs the accept pattern: per round draft -> compare, rewind,
/// catch_up the accepted cells (K/V-only, the committed path), next anchor.  `via_catch_up` builds
/// the base with catch_up() (no base-state compare) instead of explicit full head_forward() calls
/// (which compare every row's final-norm state).
int run_chain(const char* name, int base, const std::vector<int>& accepts, uint32_t seed, bool via_catch_up) {
    Harness H;
    H.build(seed);
    const int total = base + 4 * (int) accepts.size() + 8;
    std::uniform_int_distribution<int> tok_u(0, H.q.vocab - 1);
    std::uniform_real_distribution<float> hid_u(-1.0f, 1.0f);
    std::vector<int32_t> tokens((size_t) total);
    std::vector<float> hidden((size_t) total * H.q.n);
    for (auto& t : tokens) t = tok_u(H.rng);
    for (auto& x : hidden) x = hid_u(H.rng);

    int32_t* d_tok = (int32_t*) H.upload(tokens.data(), tokens.size() * 4);
    float* d_hid = (float*) H.upload(hidden.data(), hidden.size() * 4);

    Compare c, c_rounds;
    int pick_fails = 0, ties = 0, fails = 0;

    // the base context, cells [0, base)
    if (via_catch_up) {
        std::string err;
        if (!H.mtp.catch_up(d_tok, d_hid, base, 0, err)) {
            std::printf("%s: base catch_up: %s\n", name, err.c_str());
            return 1;
        }
        for (int h = 0; h < H.q.heads; ++h) H.ref.chunk(h, tokens.data(), hidden.data(), base, 0, false, nullptr);
    } else {
        std::vector<double> states((size_t) H.q.max_rows * H.q.n);
        std::vector<float> gpu((size_t) H.q.max_rows * H.q.n);
        // STRATA_MTP_PARITY_DECODE_BASE=1: the base rows one call each (the decode path: gemv
        // projections + mimo_attn_decode), isolating the prefill path's GEMM/attention
        const bool one_by_one = std::getenv("STRATA_MTP_PARITY_DECODE_BASE") != nullptr;
        for (int h = 0; h < H.q.heads; ++h)
            for (int64_t off = 0; off < base; off += (one_by_one ? 1 : H.q.max_rows)) {
                const int rows = (int) std::min<int64_t>(one_by_one ? 1 : H.q.max_rows, base - off);
                std::string err;
                if (!H.mtp.head_forward(h, d_tok + off, d_hid + (size_t) off * H.q.n, rows, off, false, err)) {
                    std::printf("%s: base forward: %s\n", name, err.c_str());
                    return 1;
                }
                H.ref.chunk(h, tokens.data() + off, hidden.data() + (size_t) off * H.q.n, rows, off, true,
                            states.data());
                CHK(cudaMemcpy(gpu.data(), H.mtp.head_state(h), (size_t) rows * H.q.n * 4, cudaMemcpyDeviceToHost),
                    "base state download");
                {
                    Compare cb;
                    cb.add(gpu.data(), states.data(), (size_t) rows * H.q.n);
                    if (std::getenv("STRATA_MTP_PARITY_VERBOSE") && cb.max_rel > 0.05)
                        std::printf("  base h=%d pos=%lld row=%zu ch=%zu: gpu=%.5f ref=%.5f (max_rel=%.3e)\n", h,
                                    (long long) off, cb.worst / H.q.n, cb.worst % H.q.n, cb.worst_gpu, cb.worst_ref,
                                    cb.max_rel);
                    c.add(gpu.data(), states.data(), (size_t) rows * H.q.n);
                }
            }
        H.mtp.rewind(base);   // head_forward never touches the cursor; the base cells are now committed
    }

    // the draft / verify rounds
    int64_t P = base - 1;
    for (size_t i = 0; i < accepts.size(); ++i) {
        const int a = accepts[i];
        int32_t ids[3] = {};
        std::string err;
        if (!H.mtp.draft(d_tok + P, d_hid + (size_t) P * H.q.n, P, ids, err)) {
            std::printf("%s r%zu: draft: %s\n", name, i, err.c_str());
            return 1;
        }
        int32_t ref_ids[3] = {};
        std::vector<double> producing[3], logits[3];
        for (int k = 0; k < 3; ++k) producing[k].resize((size_t) H.q.n);
        H.ref.draft(tokens[(size_t) P], hidden.data() + (size_t) P * H.q.n, P, ref_ids, producing, logits);
        {
            Compare cr;
            compare_round(H, name, (int) i, ids, ref_ids, logits, producing, cr, ties, pick_fails);
            if (std::getenv("STRATA_MTP_PARITY_VERBOSE"))
                std::printf("  r%zu a=%d P=%lld: max_rel=%.3e norm=%.3e worst ch=%zu gpu=%.5f ref=%.5f; picks "
                            "%d/%d/%d vs %d/%d/%d\n",
                            i, a, (long long) P, cr.max_rel, cr.norm_rel, cr.worst, cr.worst_gpu, cr.worst_ref,
                            ids[0], ids[1], ids[2], ref_ids[0], ref_ids[1], ref_ids[2]);
            c_rounds.n += cr.n;
            c_rounds.max_rel = std::max(c_rounds.max_rel, cr.max_rel);
            c_rounds.norm_rel = std::max(c_rounds.norm_rel, cr.norm_rel);
            c_rounds.max_abs = std::max(c_rounds.max_abs, cr.max_abs);
            c_rounds.bad += cr.bad;
        }

        // the verify outcome: commit the accepted drafts PLUS the bonus token (the next anchor's
        // cell), each with the target state of the window row AFTER it - the "hidden = target states
        // shifted by one" of the notes.  n = a + 1 cells at pos0 = P + 1, so committed() lands at
        // P + a + 2 and the next anchor P + a + 1 is committed() - 1 again.  The rewind is where the
        // engine drops aborted rounds (a request switch rewinds to its own cursor); in the normal
        // flow the cursor never left P + 1.
        H.mtp.rewind(P + 1);
        if (!H.mtp.catch_up(d_tok + P + 1, d_hid + (size_t) (P + 1) * H.q.n, a + 1, P + 1, err)) {
            std::printf("%s r%zu: catch_up: %s\n", name, i, err.c_str());
            return 1;
        }
        for (int h = 0; h < H.q.heads; ++h)
            H.ref.chunk(h, tokens.data() + P + 1, hidden.data() + (size_t) (P + 1) * H.q.n, a + 1, P + 1, false,
                        nullptr);
        P += a + 1;
    }
    fails = pick_fails + (c.ok() ? 0 : 1) + (c_rounds.ok() ? 0 : 1);

    const int picks = (int) accepts.size() * 3;
    const bool ok = fails == 0;
    std::printf("%-30s base n=%zu max_rel=%.3e norm=%.3e | rounds n=%zu max_rel=%.3e norm=%.3e max_abs=%.3e "
                "bad=%zu; picks %d/%d (near-ties %d)%s\n",
                name, c.n, c.max_rel, c.norm_rel, c_rounds.n, c_rounds.max_rel, c_rounds.norm_rel, c_rounds.max_abs,
                c_rounds.bad, picks - pick_fails, picks, ties, ok ? "" : "  (FAIL)");
    return ok ? 0 : 1;
}

/// The K/V-only catch_up path against explicit full forwards: the same base built both ways must
/// draft identically (a committed cell's state is never read, so the two caches agree bit for bit).
int run_kvonly_equiv(uint32_t seed) {
    const char* name = "catch_up kv-only vs full";
    const int base = 130;
    Harness H;
    H.build(seed);
    std::uniform_int_distribution<int> tok_u(0, H.q.vocab - 1);
    std::uniform_real_distribution<float> hid_u(-1.0f, 1.0f);
    std::vector<int32_t> tokens((size_t) base);
    std::vector<float> hidden((size_t) base * H.q.n);
    for (auto& t : tokens) t = tok_u(H.rng);
    for (auto& x : hidden) x = hid_u(H.rng);
    int32_t* d_tok = (int32_t*) H.upload(tokens.data(), tokens.size() * 4);
    float* d_hid = (float*) H.upload(hidden.data(), hidden.size() * 4);
    std::string err;

    for (int h = 0; h < H.q.heads; ++h) H.ref.chunk(h, tokens.data(), hidden.data(), base, 0, false, nullptr);
    int32_t ref_ids[3] = {};
    std::vector<double> producing[3], logits[3];
    for (int k = 0; k < 3; ++k) producing[k].resize((size_t) H.q.n);
    H.ref.draft(tokens[(size_t) base - 1], hidden.data() + (size_t) (base - 1) * H.q.n, base - 1, ref_ids, producing,
                logits);

    // pass 1: the production path (K/V-only catch_up)
    if (!H.mtp.catch_up(d_tok, d_hid, base, 0, err)) {
        std::printf("%s: catch_up: %s\n", name, err.c_str());
        return 1;
    }
    int32_t ids1[3] = {};
    if (!H.mtp.draft(d_tok + base - 1, d_hid + (size_t) (base - 1) * H.q.n, base - 1, ids1, err)) {
        std::printf("%s: draft 1: %s\n", name, err.c_str());
        return 1;
    }

    // pass 2: rewind to 0, rebuild the same cells with FULL forwards, draft again
    H.mtp.rewind(0);
    for (int h = 0; h < H.q.heads; ++h)
        for (int64_t off = 0; off < base; off += H.q.max_rows) {
            const int rows = (int) std::min<int64_t>(H.q.max_rows, base - off);
            if (!H.mtp.head_forward(h, d_tok + off, d_hid + (size_t) off * H.q.n, rows, off, false, err)) {
                std::printf("%s: full forward: %s\n", name, err.c_str());
                return 1;
            }
        }
    H.mtp.rewind(base);
    int32_t ids2[3] = {};
    if (!H.mtp.draft(d_tok + base - 1, d_hid + (size_t) (base - 1) * H.q.n, base - 1, ids2, err)) {
        std::printf("%s: draft 2: %s\n", name, err.c_str());
        return 1;
    }

    const bool same = ids1[0] == ids2[0] && ids1[1] == ids2[1] && ids1[2] == ids2[2];
    const bool vs_ref = ids1[0] == ref_ids[0] && ids1[1] == ref_ids[1] && ids1[2] == ref_ids[2];
    const bool ok = same && vs_ref;
    std::printf("%-30s drafts %d/%d/%d, full %d/%d/%d, ref %d/%d/%d%s\n", name, ids1[0], ids1[1], ids1[2], ids2[0],
                ids2[1], ids2[2], ref_ids[0], ref_ids[1], ref_ids[2], ok ? "" : "  (FAIL)");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA/HIP device");
        return 77;
    }

    int fails = 0;
    // the 6-round chain: the window edge (the chain's positions cross 128 mid-rounds), a full
    // rejection (accept 0), heads with dominating / mild / no sinks
    fails += run_chain("chain 6 rounds base60", 60, {3, 3, 2, 0, 1, 3}, 101, false);
    // two full rejections from the SAME anchor: speculative rows rolled back, then overwritten
    fails += run_chain("chain 2 rollbacks base200", 200, {0, 0, 3, 1}, 202, false);
    // positions > 1500, catch_up chunked past max_rows (1596 = 3x512 + 60), the window long past
    fails += run_chain("chain pos>1500 base1596", 1596, {3}, 303, true);
    // the K/V-only committed path vs explicit full forwards
    fails += run_kvonly_equiv(404);

    std::printf("%s\n", fails ? "FAIL" : "OK");
    return fails ? 1 : 0;
}
