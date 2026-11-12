#pragma once

#include "strata/kernels/exl3_peer.hpp"
#include <memory>

namespace strata {
class Exl3ExpertBacking;   // artifact/exl3_expert_backing.hpp - optional packed-expert source
class Exl3ResidentBacking;
}

namespace strata::kernels {
class Exl3ExpertStream;

// Conversation-following peer policy, called only between serial verifier windows.
class Exl3PeerAdapt {
public:
    struct Options {
        int every = 4;
        int max_swaps = 96;
        float decay = .7f;
        float min_gain = 1.5f;
        size_t byte_cap = 0; // zero: swap cap only (GGUF policy)
        std::string expert_prefix = "model.language_model.layers.";
        bool external_observer = false; // caller counts all routes, including peer hits
        /// Optional packed-expert source (trailing, default none): the adapter's model serves
        /// incoming refill planes from the SAME backing the engine opened (one host copy), and
        /// the loaded descriptors carry its layout tag - which the peer's refill format guard
        /// then matches against the tier's own mode.  Every existing caller (Qwen: no backing)
        /// is unchanged.
        std::shared_ptr<const Exl3ExpertBacking> expert_backing;
        std::shared_ptr<Exl3ResidentBacking> resident_exchange;
    };
    Exl3PeerAdapt(Exl3Peer& peer, Exl3ExpertStream& primary, const std::string& model,
                  int layers, int experts, std::vector<uint8_t>& owned, Options options);
    ~Exl3PeerAdapt();
    void observe(int layer, const int32_t* ids, int routes, float weight = 1.f);
    bool before_window(std::string& err);
    bool drain(std::string& err);
    /// Experiment support (opt-in, 2026-11-01 freeze experiment): land every in-flight
    /// refill (same blocking drain as drain()), re-check the two-card disjoint invariant,
    /// then stop planning and submitting forever. Poll and observe continue — no further
    /// PLACEMENT changes (the usage accounting observe() feeds keeps updating; nothing
    /// reads it while frozen). The placement is frozen exactly as drain() left it.
    bool freeze(std::string& err);
    bool frozen() const;
    struct Counters {
        int64_t windows = 0, rounds = 0, submitted = 0, published = 0, bytes = 0;
        double submit_ms = 0, wait_ms = 0;
    };
    Counters counters() const;
    int64_t swaps() const;
    int64_t upload_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::kernels
