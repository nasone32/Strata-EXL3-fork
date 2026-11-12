// src/mimo/mimo_main.cpp - strata-mimo: the MiMo v2 engine path on its own (the same entry point `strata`
// switches to for a MiMo EXL3 dir).
#include "strata/mimo/engine.hpp"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // the interesting output precedes crashes
    return strata::mimo::run(argc, argv);
}
