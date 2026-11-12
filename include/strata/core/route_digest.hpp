// include/strata/core/route_digest.hpp - routing-window diagnostics for
// STRATA_MIMO_ROUTE_DIGEST (CPU-only helper, no GPU dependency).
//
// One routing window's diagnostics: an ORDER-SENSITIVE FNV-1a over EVERY routed id
// (all `count` entries - invalid ids included, so the hash changes on any value or
// order change), plus the EXACT count of distinct valid expert ids over the whole
// window (a 512-slot bitset, no sampling, no truncation: a window of 512 tokens x
// top-8 = 4096 entries is counted in full; duplicated ids must not mask later ones).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::core {

struct RouteDigest {
    uint64_t fnv;     // FNV-1a over all entries in order (includes invalid ids)
    int distinct;     // distinct ids in [0, id_limit) across ALL entries
};

/// `id_limit` bounds the counted range and is capped at the helper's 512-bit set;
/// ids outside [0, id_limit) are hashed but not counted (visible in `fnv`).
inline RouteDigest route_digest(const int32_t* ids, size_t count, int id_limit = 512)
{
    if (id_limit < 0) id_limit = 0;
    if (id_limit > 512) id_limit = 512;
    uint64_t seen[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // 512 one-bit slots
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < count; ++i) {
        const int32_t e = ids[i];
        h = (h ^ (uint64_t) (uint32_t) e) * 1099511628211ull;
        if (e >= 0 && e < id_limit)
            seen[(unsigned) e >> 6] |= 1ull << ((unsigned) e & 63);
    }
    int distinct = 0;
    for (uint64_t w : seen)
        while (w != 0) { distinct += (int) (w & 1u); w >>= 1; }
    return {h, distinct};
}

}  // namespace strata::core
