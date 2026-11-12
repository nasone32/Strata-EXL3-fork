#include "strata/mimo/window_handoff.hpp"
#include <cassert>
#include <initializer_list>

int main() {
    using strata::mimo::window_handoff_eligible;
    for (int t : {0, 1, 2, 8, 9})
        for (int k : {0, 8, 64})
            assert(window_handoff_eligible(t, k, 4096, 8, true) ==
                   ((t == 2 || t == 8) && k == 8));
    assert(!window_handoff_eligible(2, 8, 0, 8, true));
    assert(!window_handoff_eligible(2, 8, -1, 8, true));
    assert(!window_handoff_eligible(8, 8, 4096, 7, true));
    assert(!window_handoff_eligible(2, 8, 4096, 8, false));
    assert(window_handoff_eligible(2, 32, 1, 2, true));
    assert(!window_handoff_eligible(2, 33, 1, 2, true));
}
