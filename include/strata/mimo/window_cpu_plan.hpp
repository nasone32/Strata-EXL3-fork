// include/strata/mimo/window_cpu_plan.hpp - the verify-window CPU overflow PLANNER (pure host).
//
// Decides WHICH whole expert groups of one mapped verify window (T = 2..8 rows x k routes,
// T*k <= 64, the frame's route cap in include/strata/mimo/window_handoff.hpp) the CPU tier
// computes while the primary GPU serves the rest.  Pure fixed-capacity host code: no engine
// includes, no allocation, no exceptions, no clocks - callable on the handoff worker thread
// between layers.  The result is a plain value the engine keeps immutable for the CPU
// background job's lifetime and drops into run_multi_complement's hard whole-group skip mask.
//
// POLICY (the user-approved bounds; see notes/MIMO-WINDOW-CPU-PLANNER-2026-11-06.md):
//   - CPU candidates are NON-RESIDENT PRIMARY MISSES only: a group whose expert is already
//     held by the primary GPU cache is a GPU hit (never stolen), and a group that belongs to
//     the peer side in ANY state - published residency (peer->has) or an ownership-table
//     entry, including PENDING INCOMING during a refill - is never taken, so the CPU never
//     computes inside a partial-ownership window (the disjoint-residency bound).
//   - Selection is WHOLE GROUPS, deterministically greedy: eligible miss groups in ascending
//     row count (cheapest CPU work first), ties by first appearance in route order; take at
//     most `max_groups` (initial default 2; 3/4 are user-experiment values via env).  Groups
//     with 4-row reuse are NOT excluded - the joint 384-cell matrix measured them net wins
//     despite slower packed kernels (they avoid whole H2D uploads); they simply sort later.
//   - `min_groups` (default 1) is the engagement floor: one cold group already beats one
//     expert's H2D upload wait.  Fewer eligible groups than the floor means no CPU work.
//   - The cap counts UNIQUE GROUPS, not rows - unlike T1's STRATA_EXL3_CPU_MAX_PER_CALL=2,
//     which counts ROUTES and would split T4 groups (run_multi_complement rejects a mask
//     whose group is half-set; the planner's mask is atomic by construction).
//   - T = 1 is REFUSED here on purpose: single-token decode keeps its existing legacy
//     classify/CPU_MIN_MISSES path (serve_decode_routes); the two policies stay disjoint.
//
// The caller supplies residency through two callbacks, each invoked AT MOST ONCE per unique
// valid expert id in the window (deduplicated here): the engine answers primary_hold with
// experts->holds(layer, e) and peer_owned with (peer && peer->valid() && peer->has(layer, e))
// || experts' ownership-exclusion tables (static owned + pending incoming).  Malformed route
// ids (outside [0, n_expert)) are never queried and never masked - the existing GPU path
// owns their error reporting.  No residency or admission state is mutated by planning.
#pragma once

#include <cstdint>

namespace strata::mimo {

/// The mapped window frame's hard capacities: T <= 8, k <= 8, T*k <= 64 routes (and the
/// user-experiment group caps 3/4 keep well inside 8).
inline constexpr int kWindowCpuMaxRoutes = 64;
inline constexpr int kWindowCpuMaxGroups = 8;

/// Residency oracle: layer context is the caller's (`user`); `expert` is always a valid id
/// for THIS layer.  A null callback means "false everywhere" (tests, or a no-peer machine).
struct WindowCpuResidency {
    void* user = nullptr;
    /// True when the primary expert stream currently holds the expert (a GPU hit to keep).
    bool (*primary_hold)(void* user, int32_t expert) = nullptr;
    /// True when the expert belongs to the peer side in ANY state: published residency
    /// (peer->has) or excluded elsewhere (the ownership tables, pending incoming included).
    bool (*peer_owned)(void* user, int32_t expert) = nullptr;
};

/// The plan: plain immutable data, valid until the caller discards it.  `row_*` lists are
/// the CPU job's dense output rows: row i is written to cpu_out + i * n_embd, owes global
/// route row_route[i] of token row_token[i], and belongs to selected group row_group[i].
struct WindowCpuPlan {
    bool ok = false;               // false: bounds violated; EVERY field is zeroed, use nothing
    int groups = 0;                // selected whole groups, 0..max_groups (<= kWindowCpuMaxGroups)
    int rows = 0;                  // total routes owed by the CPU (sum of the groups' rows)
    int64_t bytes = 0;             // groups * bytes_per_group metadata (0 when unused);
                                   // negative or overflowing inputs are refused, see below
    int32_t group_id[kWindowCpuMaxGroups] = {};    // selected experts, selection order
    int group_rows[kWindowCpuMaxGroups] = {};      // routes per selected group, 1..T*k: reuse
                                                   // is <= T only when a token's k slots hold
                                                   // DISTINCT ids (the router's normal output);
                                                   // duplicated ids within one token's k are
                                                   // legal input the existing GPU path computes,
                                                   // so one group may gather all T*k routes.
                                                   // The CPU tier serves any row count - run_rows
                                                   // chunks a group at its internal row width.
    int group_first[kWindowCpuMaxGroups] = {};     // each group's first route index (ordering)
    int row_route[kWindowCpuMaxRoutes] = {};       // ascending global route index t*k+j
    int row_token[kWindowCpuMaxRoutes] = {};       // token index t = row_route / k
    int row_group[kWindowCpuMaxRoutes] = {};       // index into group_id[row_group[i]]
    uint8_t cpu_skip[kWindowCpuMaxRoutes] = {};    // T*k whole-group mask, original route order
};

/// Plan one window.  Never throws; a bounds violation returns ok = false with all fields
/// zeroed (the caller skips the CPU tier entirely - every row stays on the ordinary GPU
/// path, which is always correct).  A default-constructed `WindowCpuResidency` (no
/// callbacks) treats every valid id as a cold miss - the pure-test default.  `bytes_per_group`
/// is uniform-per-layer metadata echoed into `bytes` for logging/bounds; it does not
/// influence selection (no unmeasured cost model - the greedy row-count order IS the
/// policy).  A NEGATIVE `bytes_per_group` is a caller bug and is refused up front with the
/// other bounds; a positive value whose product with the SELECTED group count would
/// overflow int64_t (checked as bytes_per_group <= INT64_MAX / nsel, so INT64_MAX stays
/// valid for exactly one selected group, and any nonnegative value for zero groups) is
/// likewise refused as an all-zero plan - the multiply itself never overflows.
inline WindowCpuPlan plan_window_cpu(const int32_t* routes, int tokens, int k, int n_expert,
                                     const WindowCpuResidency& residency = WindowCpuResidency{},
                                     int max_groups = 2, int min_groups = 1,
                                     int64_t bytes_per_group = 0) {
    WindowCpuPlan p;   // zero-initialized aggregate; ok stays false until the end
    const int64_t route_count = static_cast<int64_t>(tokens) * k;
    if (routes == nullptr || tokens < 2 || tokens > 8 || k <= 0 ||
        route_count <= 0 || route_count > kWindowCpuMaxRoutes || n_expert <= 0 ||
        max_groups < 0 || max_groups > kWindowCpuMaxGroups || bytes_per_group < 0)
        return p;
    if (max_groups > 0 && (min_groups < 1 || min_groups > max_groups))
        return p;
    const int count = static_cast<int>(route_count);

    // Unique-group table in first-appearance order: O(R^2) at R <= 64, no allocation.
    int32_t gid[kWindowCpuMaxRoutes] = {};
    int grows[kWindowCpuMaxRoutes] = {};
    int gfirst[kWindowCpuMaxRoutes] = {};
    bool route_ok[kWindowCpuMaxRoutes] = {};
    int ng = 0;
    for (int r = 0; r < count; ++r) {
        const int32_t e = routes[r];
        if (e < 0 || e >= n_expert) continue;   // malformed id: never queried, never masked
        route_ok[r] = true;
        int g = -1;
        for (int i = 0; i < ng; ++i)
            if (gid[i] == e) { g = i; break; }
        if (g < 0) { gid[ng] = e; grows[ng] = 0; gfirst[ng] = r; g = ng++; }
        ++grows[g];
    }

    // Residency: exactly one query pair per unique expert.  Eligible = a primary miss that
    // no peer-side state claims.
    bool eligible[kWindowCpuMaxRoutes] = {};
    for (int g = 0; g < ng; ++g) {
        const bool hold = residency.primary_hold != nullptr &&
                          residency.primary_hold(residency.user, gid[g]);
        const bool owned = residency.peer_owned != nullptr &&
                           residency.peer_owned(residency.user, gid[g]);
        eligible[g] = !hold && !owned;
    }

    // Deterministic greedy: ascending row count, ties by first appearance (a total order -
    // first-route indices are unique).  Insertion sort over at most 64 candidates.
    int cand[kWindowCpuMaxRoutes] = {};
    int nc = 0;
    for (int g = 0; g < ng; ++g)
        if (eligible[g]) {
            int i = nc++;
            while (i > 0 && (grows[cand[i - 1]] > grows[g] ||
                             (grows[cand[i - 1]] == grows[g] && gfirst[cand[i - 1]] > gfirst[g]))) {
                cand[i] = cand[i - 1];
                --i;
            }
            cand[i] = g;
        }
    int nsel = 0;
    int sel[kWindowCpuMaxGroups] = {};
    if (nc >= min_groups)   // engagement floor (default 1: one cold group beats one upload)
        for (; nsel < max_groups && nsel < nc; ++nsel) sel[nsel] = cand[nsel];

    // The bytes product must not overflow signed int64 (UB).  Checked AFTER nsel is known
    // and BEFORE any field is set: an overflowing product returns an all-zero plan.  nsel
    // == 0 skips the check (0 * anything == 0); nsel == 1 accepts bytes_per_group up to
    // INT64_MAX itself.
    if (nsel > 0 && bytes_per_group > INT64_MAX / nsel) return WindowCpuPlan{};

    // Emit: groups in selection order, rows in ascending global route order (the dense CPU
    // output row IS the list index), the skip mask atomic over every selected occurrence.
    p.ok = true;
    p.groups = nsel;
    p.bytes = static_cast<int64_t>(nsel) * bytes_per_group;   // guarded above
    bool chosen[kWindowCpuMaxRoutes] = {};   // parallel to the unique-group table
    for (int i = 0; i < nsel; ++i) {
        chosen[sel[i]] = true;
        p.group_id[i] = gid[sel[i]];
        p.group_rows[i] = grows[sel[i]];
        p.group_first[i] = gfirst[sel[i]];
    }
    int row = 0;
    for (int r = 0; r < count; ++r) {
        if (!route_ok[r]) continue;
        int g = -1, i = -1;
        for (int q = 0; q < ng; ++q)
            if (gid[q] == routes[r]) { g = q; break; }
        for (int q = 0; q < nsel; ++q)
            if (sel[q] == g) { i = q; break; }
        if (g < 0 || !chosen[g]) continue;
        p.cpu_skip[r] = 1;
        p.row_route[row] = r;
        p.row_token[row] = r / k;
        p.row_group[row] = i;
        ++row;
    }
    p.rows = row;
    return p;
}

}  // namespace strata::mimo
