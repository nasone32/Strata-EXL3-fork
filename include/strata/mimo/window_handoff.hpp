#pragma once

namespace strata::mimo {
// Host-only eligibility: the mapped frame and the peer's route plans hold 64 routes.
constexpr bool window_handoff_eligible(int tokens, int k, int width, int chunk, bool all_rows) {
    return all_rows && tokens >= 2 && tokens <= 8 && k > 0 && k <= 64 / tokens &&
           width > 0 && chunk >= tokens;
}
} // namespace strata::mimo
