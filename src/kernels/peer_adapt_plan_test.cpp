#include "strata/core/peer_adapt_plan.hpp"
#include <cstdio>

int main() {
    // Two layers: exclude a very hot primary resident, ignore below-two candidates,
    // require the inclusive 1.5 boundary, pair within layers and rank globally.
    const float usage[] = {0, 10, 100, 1.99f, 9, 2,  .5f, 4, 8, 0, 2, 0};
    const int32_t primary[] = {-1,-1,0,-1,-1,-1, -1,-1,-1,-1,-1,-1};
    const int32_t peer[] = {0,1,-1,-1,-1,-1, 2,-1,-1,3,-1,-1};
    const auto p = strata::core::peer_adapt_plan(usage, primary, peer, 2, 6);
    bool ok = p.size() == 3 && p[0].layer == 0 && p[0].in == 4 && p[0].out == 0 && p[0].gain == 9 &&
              p[1].layer == 1 && p[1].in == 2 && p[1].out == 3 && p[1].gain == 8 &&
              p[2].layer == 1 && p[2].in == 1 && p[2].out == 0 && p[2].gain == 3.5f;
    const float edge[] = {.5f, 2};
    const int32_t pr[] = {-1,-1}, pe[] = {0,-1};
    ok = ok && strata::core::peer_adapt_plan(edge, pr, pe, 1, 2).size() == 1 &&
         strata::core::peer_adapt_plan(edge, pr, pe, 1, 2, 1.51f).empty();
    std::printf("GGUF peer policy: disjoint candidates, per-layer victims, gain ranking and thresholds %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
