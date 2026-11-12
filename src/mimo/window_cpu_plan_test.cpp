// src/mimo/window_cpu_plan_test.cpp - CPU unit test for the verify-window CPU overflow
// planner (include/strata/mimo/window_cpu_plan.hpp).  New file; not in any CMake target on
// purpose (no project reconfigure while GPU builds run); compiled standalone:
//   g++ -O2 -std=c++17 -Iinclude -o /tmp/opencode/window_cpu_plan_test src/mimo/window_cpu_plan_test.cpp
//
// What it falsifies, and what it deliberately does NOT:
//
//  1. Bounds/malformed inputs: T=1 (the legacy T1 policy's exclusive domain), T>8, k<=0,
//     T*k>64 (the mapped frame's route cap), null routes, n_expert<=0, group caps outside
//     0..8, min_groups outside 1..max_groups - each returns ok=false with EVERY field zero,
//     so a caller that mis-reads it still skips the CPU tier rather than masking garbage.
//  2. Whole-group atomicity and coverage: for every scenario, an invariant checker re-derives
//     the mask from the returned groups (a route is masked IFF its expert is a selected
//     group), counts rows, walks row_route/row_token/row_group, and proves no group is ever
//     split - the exact property run_multi_complement's skip validation would throw on.
//  3. Exclusion rules: primary-resident groups (GPU hits) and ANY peer-side claim (published,
//     pending-incoming via the ownership tables) are never selected, even when cheapest.
//  4. The group cap counts UNIQUE GROUPS (0/1/2), not rows; duplicates across all T rows form
//     ONE group; selection order is ascending row count with first-appearance tie-breaks.
//  5. The residency callbacks fire EXACTLY once per unique VALID expert and never for a
//     malformed id.
//
// It does NOT test the engine wiring (serve_window_routes, run_rows_bg overlap, the CPU
// output overlay) - that is the integration proposal's own test list, and it does not
// measure anything (the joint 384-cell matrix in notes/results/2026-11-06-packed-cpu-partition
// is the measurement; this file checks policy, not speed).
#include "strata/mimo/window_cpu_plan.hpp"

#include <cstdio>
#include <cstring>

using namespace strata::mimo;

namespace {

int fails = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("window_cpu_plan_test: FAIL: %s\n", what);
        ++fails;
    }
}

// The test-side residency oracle: caller-controlled sets + per-unique query counters.
struct Oracle {
    bool hold[4096] = {};
    bool owned[4096] = {};
    int hold_queries[4096] = {};
    int owned_queries[4096] = {};
    static bool q_hold(void* user, int32_t e) {
        auto* o = static_cast<Oracle*>(user);
        ++o->hold_queries[e];
        return o->hold[e];
    }
    static bool q_owned(void* user, int32_t e) {
        auto* o = static_cast<Oracle*>(user);
        ++o->owned_queries[e];
        return o->owned[e];
    }
    WindowCpuResidency hooks() { return {this, &Oracle::q_hold, &Oracle::q_owned}; }
};

/// The universal invariant checker: coverage, atomicity, cross-field agreement.  Runs on
/// EVERY plan any test obtains (also the !ok ones, which must be all-zero).
void verify_invariants(const WindowCpuPlan& p, const int32_t* routes, int tokens, int k,
                       int n_expert, const char* tag) {
    const int count = tokens * k;
    char what[256];

    // Groups: distinct ids, plausible rows/first, sums agree.
    int sum_rows = 0;
    for (int i = 0; i < p.groups; ++i) {
        std::snprintf(what, sizeof what, "%s: group %d id in range", tag, i);
        check(p.group_id[i] >= 0 && p.group_id[i] < n_expert, what);
        for (int j = 0; j < i; ++j) {
            std::snprintf(what, sizeof what, "%s: group ids distinct (%d repeats %d)", tag, i, j);
            check(p.group_id[i] != p.group_id[j], what);
        }
        std::snprintf(what, sizeof what, "%s: group %d rows in 1..routes", tag, i);
        check(p.group_rows[i] >= 1 && p.group_rows[i] <= count, what);   // duplicated ids
        // within one token's k are legal, so a group may exceed T - see the dedicated test.
        std::snprintf(what, sizeof what, "%s: group %d first in range", tag, i);
        check(p.group_first[i] >= 0 && p.group_first[i] < count, what);
        std::snprintf(what, sizeof what, "%s: group %d first points at its own id", tag, i);
        check(routes[p.group_first[i]] == p.group_id[i], what);
        sum_rows += p.group_rows[i];
    }
    std::snprintf(what, sizeof what, "%s: rows == sum of group rows", tag);
    check(p.rows == sum_rows, what);

    // Mask: a route is masked IFF its (valid) expert is a selected group.  Whole groups only.
    int masked = 0;
    for (int r = 0; r < count; ++r) {
        bool in_group = false;
        if (routes[r] >= 0 && routes[r] < n_expert)
            for (int i = 0; i < p.groups; ++i)
                in_group |= p.group_id[i] == routes[r];
        std::snprintf(what, sizeof what, "%s: route %d mask matches selected groups", tag, r);
        check(p.cpu_skip[r] == (in_group ? 1 : 0), what);
        masked += p.cpu_skip[r] != 0;
    }
    std::snprintf(what, sizeof what, "%s: masked count == rows", tag);
    check(masked == p.rows, what);

    // Row list: ascending, exactly the masked routes, token math, group back-reference.
    int prev = -1;
    for (int i = 0; i < p.rows; ++i) {
        std::snprintf(what, sizeof what, "%s: row %d route ascending", tag, i);
        check(p.row_route[i] > prev, what);
        prev = p.row_route[i];
        std::snprintf(what, sizeof what, "%s: row %d route is masked", tag, i);
        check(p.cpu_skip[p.row_route[i]] != 0, what);
        std::snprintf(what, sizeof what, "%s: row %d token math", tag, i);
        check(p.row_token[i] == p.row_route[i] / k, what);
        std::snprintf(what, sizeof what, "%s: row %d group backref", tag, i);
        check(p.row_group[i] >= 0 && p.row_group[i] < p.groups, what);
        check(routes[p.row_route[i]] == p.group_id[p.row_group[i]], what);
    }
    for (int i = p.rows; i < kWindowCpuMaxRoutes; ++i) {
        std::snprintf(what, sizeof what, "%s: row tail %d zeroed", tag, i);
        check(p.row_route[i] == 0 && p.row_token[i] == 0 && p.row_group[i] == 0, what);
    }
    for (int r = count; r < kWindowCpuMaxRoutes; ++r) {   // mask tail beyond this window
        std::snprintf(what, sizeof what, "%s: mask tail %d zeroed", tag, r);
        check(p.cpu_skip[r] == 0, what);
    }
}

/// A rejected plan must be ALL zero (not just ok=false): a caller that ignores `ok` and
/// reads the arrays still sees an empty selection, never a stale or garbage mask.
/// (Bounds rejection happens before any residency query, so no oracle is needed here.)
void expect_rejected(const int32_t* routes, int tokens, int k, int n_expert,
                     int max_groups, int min_groups, const char* tag) {
    const WindowCpuPlan p = plan_window_cpu(routes, tokens, k, n_expert, {}, max_groups, min_groups);
    const WindowCpuPlan zero{};
    char what[256];
    std::snprintf(what, sizeof what, "%s: rejected (ok=false)", tag);
    check(!p.ok, what);
    std::snprintf(what, sizeof what, "%s: rejected plan is bit-identical to a fresh one", tag);
    check(std::memcmp(&p, &zero, sizeof(WindowCpuPlan)) == 0, what);
}

}  // namespace

int main() {
    // ---- 1. malformed bounds: every one must return an all-zero, !ok plan -------------
    {
        const int32_t r[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        expect_rejected(r, 1, 8, 64, 2, 1, "T1 refused (legacy T1 policy owns it)");
        expect_rejected(r, 0, 8, 64, 2, 1, "T0");
        expect_rejected(r, 9, 7, 64, 2, 1, "T9");
        expect_rejected(r, 2, 0, 64, 2, 1, "k0");
        expect_rejected(r, 2, -1, 64, 2, 1, "k<0");
        expect_rejected(r, 2, 33, 64, 2, 1, "T2 k33 = 66 routes > 64");
        expect_rejected(r, 8, 9, 64, 2, 1, "T8 k9 = 72 routes > 64");
        expect_rejected(nullptr, 4, 8, 64, 2, 1, "null routes");
        expect_rejected(r, 4, 8, 0, 2, 1, "n_expert 0");
        expect_rejected(r, 4, 8, 64, -1, 1, "max_groups -1");
        expect_rejected(r, 4, 8, 64, 9, 1, "max_groups 9");
        expect_rejected(r, 4, 8, 64, 2, 0, "min_groups 0");
        expect_rejected(r, 4, 8, 64, 2, 3, "min_groups 3 > max 2");
        check(kWindowCpuMaxRoutes == 64 && kWindowCpuMaxGroups == 8, "frame capacities");
    }

    // ---- 2. duplicate experts across ALL 4 rows: one whole group, never split ----------
    {
        Oracle o;   // 64 experts, nothing held/owned: every valid id is a cold miss
        const int32_t routes[8] = {7, 5, 7, 9, 7, 5, 7, 5};   // T4 k2: e7 x4, e5 x3, e9 x1
        const WindowCpuPlan p = plan_window_cpu(routes, 4, 2, 64, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 2 && p.rows == 4, "T4 duplicates: two cheapest groups, 4 rows");
        check(p.group_id[0] == 9 && p.group_rows[0] == 1, "1-row group first (ascending rows)");
        check(p.group_id[1] == 5 && p.group_rows[1] == 3, "then the 3-row group");
        for (int i = 0; i < p.groups; ++i) {
            int seen = 0;
            for (int r = 0; r < 8; ++r) seen += routes[r] == p.group_id[i];
            check(seen == p.group_rows[i], "group_rows counts every duplicate");
        }
        verify_invariants(p, routes, 4, 2, 64, "T4 duplicates");
        // The exactly-once-per-unique contract (7, 5, 9 queried once each):
        check(o.hold_queries[7] == 1 && o.hold_queries[5] == 1 && o.hold_queries[9] == 1,
              "one holds query per unique expert");
        check(o.owned_queries[7] + o.owned_queries[5] + o.owned_queries[9] == 3,
              "one owned query per unique expert");
    }

    // ---- 3. mixed peer masks: any peer-side claim excludes the group outright ---------
    {
        Oracle o;
        o.owned[2] = true;    // published peer residency
        o.owned[8] = true;    // pending incoming (ownership table) - same callback, same rule
        o.hold[6] = true;     // primary-resident: a GPU hit, not the CPU's to take
        const int32_t routes[8] = {2, 3, 6, 8, 2, 3, 6, 8};   // T4 k2, each expert twice
        const WindowCpuPlan p = plan_window_cpu(routes, 4, 2, 64, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 1 && p.group_id[0] == 3, "only the unclaimed miss survives");
        verify_invariants(p, routes, 4, 2, 64, "mixed peer masks");
        // Now make e3 ALSO peer-owned: nothing left, but ok stays true (a legal empty plan).
        o.owned[3] = true;
        const WindowCpuPlan q = plan_window_cpu(routes, 4, 2, 64, o.hooks(), 2, 1, 0);
        check(q.ok && q.groups == 0 && q.rows == 0, "all claimed: empty plan, not an error");
        verify_invariants(q, routes, 4, 2, 64, "all claimed");
    }

    // ---- 4. cheapest group still excluded: a 1-row peer group loses to a 2-row miss ---
    {
        Oracle o;
        o.owned[1] = true;
        const int32_t routes[4] = {1, 1, 4, 4};   // T2 k2
        const WindowCpuPlan p = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 1 && p.group_id[0] == 4, "exclusion beats row-count order");
        verify_invariants(p, routes, 2, 2, 64, "exclusion beats order");
    }

    // ---- 5. group cap 0/1/2 counts UNIQUE GROUPS, not rows ----------------------------
    {
        Oracle o;
        const int32_t routes[6] = {10, 11, 12, 10, 11, 12};   // T3 k2: three 2-row groups
        const WindowCpuPlan p0 = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 0, 1, 0);
        check(p0.ok && p0.groups == 0 && p0.rows == 0, "cap 0: CPU tier off for this window");
        verify_invariants(p0, routes, 3, 2, 64, "cap 0");
        const WindowCpuPlan p1 = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 1, 1, 0);
        check(p1.ok && p1.groups == 1 && p1.rows == 2 && p1.group_id[0] == 10,
              "cap 1: first-appearance wins the 2-row tie");
        verify_invariants(p1, routes, 3, 2, 64, "cap 1");
        const WindowCpuPlan p2 = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 2, 1, 0);
        check(p2.ok && p2.groups == 2 && p2.rows == 4 && p2.group_id[1] == 11, "cap 2");
        verify_invariants(p2, routes, 3, 2, 64, "cap 2");
        // min_groups: one eligible below the floor selects NOTHING (engagement floor).
        o.hold[11] = true; o.hold[12] = true;
        const WindowCpuPlan m2 = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 2, 2, 0);
        check(m2.ok && m2.groups == 0, "min_groups 2 with one eligible: no CPU work");
        const WindowCpuPlan m1 = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 2, 1, 0);
        check(m1.ok && m1.groups == 1 && m1.group_id[0] == 10,
              "min_groups 1 (default): one cold group still engages");
        verify_invariants(m2, routes, 3, 2, 64, "min 2");
        verify_invariants(m1, routes, 3, 2, 64, "min 1");
    }

    // ---- 6. route cap boundary: the full 8x8 frame plans, 2 groups out of 64 ----------
    {
        Oracle o;
        int32_t routes[64];
        for (int r = 0; r < 64; ++r) routes[r] = r;   // 64 distinct cold misses
        const WindowCpuPlan p = plan_window_cpu(routes, 8, 8, 256, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 2 && p.rows == 2, "full frame: cap 2 of 64 one-row groups");
        check(p.group_id[0] == 0 && p.group_id[1] == 1, "all-rows tie: first-appearance order");
        verify_invariants(p, routes, 8, 8, 256, "full frame");
    }

    // ---- 7. invalid ids: skipped, unqueried, unmasked; the rest still plans ------------
    {
        Oracle o;
        const int32_t routes[6] = {-1, 12, 64, 12, -1, 3};   // T3 k2, n_expert = 64: -1 and 64 bad
        const WindowCpuPlan p = plan_window_cpu(routes, 3, 2, 64, o.hooks(), 2, 1, 0);
        check(p.ok, "invalid ids do not reject the window");
        check(p.cpu_skip[0] == 0 && p.cpu_skip[2] == 0 && p.cpu_skip[4] == 0,
              "malformed routes never masked");
        verify_invariants(p, routes, 3, 2, 64, "invalid ids");
        int total = 0;
        for (int e = 0; e < 64; ++e) total += o.hold_queries[e];
        check(total == 2 && o.hold_queries[12] == 1 && o.hold_queries[3] == 1,
              "only the two unique VALID experts queried");
    }

    // ---- 8. bytes metadata, null-oracle default, determinism --------------------------
    {
        const int32_t routes[4] = {20, 21, 20, 21};
        const WindowCpuPlan p = plan_window_cpu(routes, 2, 2, 64, {}, 2, 1, 6291456);
        check(p.ok && p.groups == 2 && p.bytes == 2 * 6291456,
              "bytes = groups * bytes_per_group (uniform-rate metadata)");
        verify_invariants(p, routes, 2, 2, 64, "bytes/null residency");
        const WindowCpuPlan again = plan_window_cpu(routes, 2, 2, 64, {}, 2, 1, 6291456);
        check(std::memcmp(&p, &again, sizeof(WindowCpuPlan)) == 0,
              "deterministic: identical inputs, byte-identical plan");
        // No residency oracle at all: every valid route is a cold miss (2 groups taken).
        check(p.groups == 2 && p.rows == 4, "null oracle treats all valid ids as misses");
    }

    // ---- 9. resident-only and empty-eligible shapes ------------------------------------
    {
        Oracle o;
        for (int e = 0; e < 64; ++e) o.hold[e] = true;
        const int32_t routes[4] = {1, 2, 3, 4};
        const WindowCpuPlan p = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 0, "all resident: empty plan (GPU hits are never stolen)");
        verify_invariants(p, routes, 2, 2, 64, "all resident");
    }

    // ---- 10. bytes-product edges: negative refused early, overflow refused all-zero ---
    // The product nsel * bytes_per_group must never overflow signed int64 (UB).  INT64_MAX
    // with exactly ONE selected group is valid; with TWO it overflows and the WHOLE plan is
    // rejected all-zero; zero selected groups accepts any nonnegative value; a negative
    // bytes parameter is a caller bug refused up front - with a positive cap, with cap 0,
    // and even when nothing would have been selected anyway.
    {
        Oracle o;                                   // e30/e31 cold: two 2-row groups
        Oracle full;                                // everything resident: nothing eligible
        for (int e = 0; e < 64; ++e) full.hold[e] = true;
        const WindowCpuPlan zero{};                 // the all-zero reference
        const int32_t routes[4] = {30, 31, 30, 31};

        const WindowCpuPlan one = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 1, 1, INT64_MAX);
        check(one.ok && one.groups == 1 && one.bytes == INT64_MAX,
              "INT64_MAX bytes, 1 selected group: valid (INT64_MAX / 1 accepts it)");
        verify_invariants(one, routes, 2, 2, 64, "bytes 1-group");

        const WindowCpuPlan two = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 2, 1, INT64_MAX);
        check(!two.ok && std::memcmp(&two, &zero, sizeof(WindowCpuPlan)) == 0,
              "INT64_MAX bytes, 2 selected groups: overflow -> all-zero rejection");

        const WindowCpuPlan none = plan_window_cpu(routes, 2, 2, 64, full.hooks(), 2, 1, INT64_MAX);
        check(none.ok && none.groups == 0 && none.bytes == 0,
              "0 selected groups + huge nonnegative bytes: valid empty plan");
        verify_invariants(none, routes, 2, 2, 64, "bytes 0-group");

        const WindowCpuPlan neg = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 2, 1, -1);
        check(!neg.ok && std::memcmp(&neg, &zero, sizeof(WindowCpuPlan)) == 0,
              "negative bytes, positive cap: all-zero rejection");
        const WindowCpuPlan neg0 = plan_window_cpu(routes, 2, 2, 64, o.hooks(), 0, 1, -1);
        check(!neg0.ok && std::memcmp(&neg0, &zero, sizeof(WindowCpuPlan)) == 0,
              "negative bytes, cap 0: all-zero rejection");
        const WindowCpuPlan negs = plan_window_cpu(routes, 2, 2, 64, full.hooks(), 2, 1, -8);
        check(!negs.ok && std::memcmp(&negs, &zero, sizeof(WindowCpuPlan)) == 0,
              "negative bytes, nothing eligible: still refused (the check is early)");
    }

    // ---- 11. duplicated ids WITHIN one token's k slots: legal input, grouped, not split -
    // Routers normally emit distinct ids per token, but the existing GPU path computes
    // duplicated routes, so the planner must group them - never reject, never silently
    // drop.  A group may then exceed T rows (the "reuse <= T" claim only holds for
    // distinct ids); run_rows serves any row count, chunking a group at its own width.
    {
        Oracle o;
        o.hold[9] = true;   // keep the 1-row group a GPU hit: the duplicate-heavy ones win
        // T2 k4: token 0 routes e5 three times, token 1 twice more -> e5 x4, e7 x3, e9 x1.
        const int32_t routes[8] = {5, 5, 7, 5, 5, 7, 7, 9};
        const WindowCpuPlan p = plan_window_cpu(routes, 2, 4, 64, o.hooks(), 2, 1, 0);
        check(p.ok && p.groups == 2 && p.rows == 7, "duplicate-token ids accepted, grouped whole");
        check(p.group_id[0] == 7 && p.group_rows[0] == 3, "e7: 3 rows > T=2 (cross-token reuse)");
        check(p.group_id[1] == 5 && p.group_rows[1] == 4,
              "e5: 4 rows > T=2 (duplicates within token 0's k slots)");
        verify_invariants(p, routes, 2, 4, 64, "duplicate-token ids");

        // The extreme: every route of the full 8x8 frame names ONE expert - a single whole
        // group of 64 rows, mask 64/64, dense row list of 64.
        int32_t all5[64];
        for (int r = 0; r < 64; ++r) all5[r] = 5;
        const WindowCpuPlan q = plan_window_cpu(all5, 8, 8, 64, o.hooks(), 2, 1, 0);
        check(q.ok && q.groups == 1 && q.rows == 64 && q.group_rows[0] == 64,
              "one expert across all 64 routes: one group, rows = 64");
        verify_invariants(q, all5, 8, 8, 64, "all-same expert");
    }

    if (fails == 0) std::printf("window_cpu_plan_test: ALL OK\n");
    else std::printf("window_cpu_plan_test: %d FAILURE(S)\n", fails);
    return fails == 0 ? 0 : 1;
}
