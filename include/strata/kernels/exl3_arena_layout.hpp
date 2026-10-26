#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::kernels {

// Rank-major backing: [slot rank][owned layers][gate/up/down planes]. All plane starts
// and rank boundaries are 256-byte aligned. No allocation or residency policy lives here.
struct Exl3ArenaLayout {
    using Planes = std::array<size_t, 9>;
    std::vector<Planes> offsets;
    size_t rank_bytes = 0;
    size_t bytes = 0;

    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::overflow_error("exl3: arena size overflow");
        return a + b;
    }
    static size_t align(size_t n) { return add(n, 255) & ~size_t(255); }

    Exl3ArenaLayout(const std::vector<Planes>& layers, int slots) {
        if (layers.empty() || layers.size() > 48 || slots < 0 || slots > 512)
            throw std::invalid_argument("exl3: invalid arena geometry");
        offsets.resize(layers.size());
        for (size_t l = 0; l < layers.size(); ++l) {
            for (size_t p = 0; p < 9; ++p) {
                if (layers[l][p] == 0) throw std::invalid_argument("exl3: empty arena plane");
                rank_bytes = align(rank_bytes);
                offsets[l][p] = rank_bytes;
                rank_bytes = add(rank_bytes, layers[l][p]);
            }
        }
        rank_bytes = align(rank_bytes);
        if (slots && rank_bytes > std::numeric_limits<size_t>::max() / (size_t) slots)
            throw std::overflow_error("exl3: arena capacity overflow");
        bytes = rank_bytes * (size_t) slots;
    }
};

} // namespace strata::kernels
