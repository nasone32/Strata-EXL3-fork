// src/core/route_digest_test.cpp - the routing-window digest helper's exactness contract
// (CPU-only, no GPU): the distinct count must cover ALL entries of a >256-entry window
// (the regression this guards: a first-256-entries capture returned 1 for a 256-expert
// window), the 512-id range must be counted in full, invalid ids must be hashed but not
// counted, and the hash must be order-sensitive and deterministic.
#include "strata/core/route_digest.hpp"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using strata::core::route_digest;

    {   // determinism + empty window
        const int32_t none[] = {0};
        CHECK(route_digest(none, 0).fnv == 1469598103934665603ull);
        CHECK(route_digest(none, 0).distinct == 0);
        const int32_t x[] = {1, 2, 3};
        CHECK(route_digest(x, 3).fnv == route_digest(x, 3).fnv);
    }
    {   // THE REGRESSION CASE: 4096 entries (512 tokens x top-8); the FIRST 256 entries
        // are all expert 7, every id 0..255 appears later.  distinct must be 256 over the
        // whole window - a first-256-entries capture would answer 1.
        std::vector<int32_t> ids(4096);
        for (int i = 0; i < 256; ++i) ids[(size_t) i] = 7;
        for (size_t i = 256; i < ids.size(); ++i) ids[i] = (int32_t) ((i - 256) % 256);
        const auto d = route_digest(ids.data(), ids.size());
        CHECK(d.distinct == 256);
        // and the same window with the tail REMOVED is 1 distinct expert (7 only)
        const auto head = route_digest(ids.data(), 256);
        CHECK(head.distinct == 1);
        // tail-only: ids 0..255 once each (256 entries, no duplicates masking)
        std::vector<int32_t> tail(ids.begin() + 256, ids.end());
        CHECK(route_digest(tail.data(), 256).distinct == 256);
        // the full window's hash differs from the head's (all entries hashed)
        CHECK(d.fnv != head.fnv);
    }
    {   // the full 512-id range, duplicates interleaved: distinct == 512
        std::vector<int32_t> ids;
        for (int round = 0; round < 8; ++round)              // 8 passes x 512 ids
            for (int e = 0; e < 512; ++e) ids.push_back(e);  // 4096 entries
        CHECK(route_digest(ids.data(), ids.size(), 512).distinct == 512);
        // id_limit default 512 already covers it; a smaller limit counts only the range
        CHECK(route_digest(ids.data(), ids.size(), 256).distinct == 256);
        // ids at/above the limit are hashed but not counted
        const int32_t wide[] = {5, 600, -3, 511, 512, 5};
        const auto w = route_digest(wide, 6, 512);
        CHECK(w.distinct == 2);                              // {5, 511}
        const int32_t reordered[] = {511, 5, 600, -3, 512, 5};
        CHECK(route_digest(reordered, 6, 512).fnv != w.fnv); // order-sensitive
        CHECK(route_digest(reordered, 6, 512).distinct == w.distinct);
        // an out-of-range id changing value still changes the hash (it is hashed)
        const int32_t wide2[] = {5, 601, -3, 511, 512, 5};
        CHECK(route_digest(wide2, 6, 512).fnv != w.fnv);
    }

    if (fails == 0) std::printf("route_digest_test PASS\n");
    else std::printf("route_digest_test FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
