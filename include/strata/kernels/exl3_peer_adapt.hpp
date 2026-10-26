#pragma once

#include "strata/kernels/exl3_peer.hpp"
#include <memory>

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
    };
    Exl3PeerAdapt(Exl3Peer& peer, Exl3ExpertStream& primary, const std::string& model,
                  int layers, int experts, std::vector<uint8_t>& owned, Options options);
    ~Exl3PeerAdapt();
    void observe(int layer, const int32_t* ids, int routes);
    bool before_window(std::string& err);
    bool drain(std::string& err);
    int64_t swaps() const;
    int64_t upload_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::kernels
