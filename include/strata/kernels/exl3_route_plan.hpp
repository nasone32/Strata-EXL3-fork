#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::kernels {

// Stable counting sort of token-major routes into expert-major work. Load each group's expert once,
// then compute all its token rows. destinations preserves the original [token][top-k] output layout.
// Reuse this object between windows/chunks; vector capacities are retained.
struct Exl3RoutePlan {
    std::vector<int> experts;
    std::vector<int> offsets;
    std::vector<int> destinations;
    int tokens = 0;
    int top_k = 0;

    void build(const int32_t* ids, int n_tokens, int k, int n_experts) {
        experts.clear(); offsets.clear(); destinations.clear();
        tokens = top_k = 0;
        if (n_tokens < 0 || k <= 0 || n_experts <= 0 || n_experts > 65536 ||
            n_tokens > std::numeric_limits<int>::max() / k || (n_tokens > 0 && ids == nullptr))
            throw std::invalid_argument("exl3: invalid route-plan dimensions");
        const int routes = n_tokens * k;
        counts_.assign((size_t) n_experts, 0);
        for (int r = 0; r < routes; ++r) {
            if (ids[r] < 0 || ids[r] >= n_experts)
                throw std::invalid_argument("exl3: routed expert index out of range");
            ++counts_[(size_t) ids[r]];
        }
        cursors_.resize((size_t) n_experts);
        int base = 0;
        for (int e = 0; e < n_experts; ++e) {
            cursors_[(size_t) e] = base;
            if (counts_[(size_t) e] != 0) {
                experts.push_back(e);
                offsets.push_back(base);
                base += counts_[(size_t) e];
            }
        }
        offsets.push_back(base);
        destinations.resize((size_t) routes);
        for (int r = 0; r < routes; ++r)
            destinations[(size_t) cursors_[(size_t) ids[r]]++] = r;
        tokens = n_tokens;
        top_k = k;
    }

    int token_at(size_t grouped_row) const {
        if (top_k == 0 || grouped_row >= destinations.size())
            throw std::out_of_range("exl3: grouped route row out of range");
        return destinations[grouped_row] / top_k;
    }

    // Consume resident groups before admitting misses. Otherwise an ascending full-prompt scan
    // can evict every resident expert before reaching it, even with a half-model cache.
    void prioritize_resident(const std::vector<uint8_t>& resident) {
        if (resident.size() != experts.size())
            throw std::invalid_argument("exl3: invalid resident-group mask");
        next_experts_.clear(); next_offsets_.clear(); next_destinations_.clear();
        for (int pass = 1; pass >= 0; --pass) {
            for (size_t g = 0; g < experts.size(); ++g) {
                if ((resident[g] != 0) != (pass != 0)) continue;
                next_experts_.push_back(experts[g]);
                next_offsets_.push_back((int) next_destinations_.size());
                next_destinations_.insert(next_destinations_.end(), destinations.begin() + offsets[g],
                                          destinations.begin() + offsets[g + 1]);
            }
        }
        next_offsets_.push_back((int) next_destinations_.size());
        experts.swap(next_experts_); offsets.swap(next_offsets_); destinations.swap(next_destinations_);
    }

    // Remove whole groups from the plan (drop[g] != 0 drops group g), compacting offsets and
    // destinations in place; the surviving groups and their routes keep their relative order.
    // The CPU-tier pre-pass uses this: refused groups leave the GPU plan entirely, so every
    // downstream arm (waves, overlap split, batching, tiles) sees only GPU work.
    void drop_groups(const std::vector<uint8_t>& drop) {
        if (drop.size() != experts.size())
            throw std::invalid_argument("exl3: invalid drop-group mask");
        next_experts_.clear(); next_offsets_.clear(); next_destinations_.clear();
        for (size_t g = 0; g < experts.size(); ++g) {
            if (drop[g] != 0) continue;
            next_experts_.push_back(experts[g]);
            next_offsets_.push_back((int) next_destinations_.size());
            next_destinations_.insert(next_destinations_.end(), destinations.begin() + offsets[g],
                                      destinations.begin() + offsets[g + 1]);
        }
        next_offsets_.push_back((int) next_destinations_.size());
        experts.swap(next_experts_); offsets.swap(next_offsets_); destinations.swap(next_destinations_);
    }

private:
    std::vector<int> counts_, cursors_;
    std::vector<int> next_experts_, next_offsets_, next_destinations_;
};

} // namespace strata::kernels
