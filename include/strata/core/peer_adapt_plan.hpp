#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

struct PeerAdaptSwap { float gain; int32_t layer, in, out; };

// The GGUF PeerExperts::adapt selection rule: per-layer hottest non-residents
// absent from BOTH cards against coldest peer residents, then global gain order.
// Defaults deliberately match src/core/peer_experts.cpp (minimum 2, margin 1.5).
// Keep the established GGUF implementation untouched; EXL3 supplies its own snapshots.
inline std::vector<PeerAdaptSwap> peer_adapt_plan(const float* usage, const int32_t* primary,
                                                 const int32_t* peer, int layers, int experts,
                                                 float margin = 1.5f) {
    std::vector<PeerAdaptSwap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int l = 0; l < layers; ++l) {
        cand.clear(); vict.clear();
        for (int e = 0; e < experts; ++e) {
            const int64_t i = (int64_t) l * experts + e;
            if (peer[i] >= 0) vict.emplace_back(usage[i], e);
            else if (primary[i] < 0 && usage[i] >= 2.0f) cand.emplace_back(usage[i], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + margin) break;
            swaps.push_back({cand[i].first - vict[i].first, l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](auto& a, auto& b) { return a.gain > b.gain; });
    return swaps;
}
} // namespace strata::core
