#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>

namespace strata::core {
// Bounded, aging frequency history. Count once per routed row; age once per service
// window, not per expert. Prefill callers average counts over their token rows so
// a large chunk does not dominate hundreds of subsequent decode tokens.
// No device storage and no allocation in the hot path.
struct ExpertFrequency {
    std::array<float, 512> count{};
    uint32_t windows = 0;
    void observe(const int32_t* ids, int n, uint32_t every, float decay, float weight = 1.f) {
        if (!every || !(decay >= 0 && decay <= 1)) throw std::invalid_argument("invalid frequency aging");
        if (++windows % every == 0) for (auto& c : count) c *= decay;
        for (int i = 0; i < n; ++i) if (ids[i] >= 0 && ids[i] < (int) count.size()) count[ids[i]] += weight;
    }
};
} // namespace strata::core
