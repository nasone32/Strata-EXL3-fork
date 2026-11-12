// include/strata/mimo/engine.hpp - the Xiaomi MiMo v2 (EXL3, config model_type mimo_v2) engine path as a
// library: `strata` hands a MiMo EXL3 dir to it at the top of main() (the family switch, before any Exl3Pack
// is built), and strata-mimo is a thin wrapper around the same entry point - the shape of strata/qwen35/engine.hpp.
// See src/mimo/mimo_engine.cu and notes/MIMO-V2-PORT.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::mimo {
// Synthetic parity seam: teacher-forced verify windows, returning every row's logits.
int run_verify_logits(const char* exl3_dir, const std::vector<int>& widths,
                      std::vector<float>& logits, std::string& error,
                      bool calibrate_t1 = false, bool* small_m_enabled = nullptr);
int run_greedy_dflash(const char* exl3_dir, const char* draft_dir, int proposals, const int32_t* prompt,
                      int64_t n_prompt, int n_gen, int64_t ctx, int chunk,
                      std::vector<int32_t>& out, std::string& error);

/// The engine's command line (strata's own flag names where they mean the same thing: --exl3 / --model,
/// --max-context, --prefill / --chunk, --serve, --mtp N, --pool-workers, --tokens, --max-new,
/// --no-token-graph; the family's own: --expert-vram, --expert-ram-cache).  Returns the process exit code.
int run(int argc, char** argv);

/// The engine core on a token-id prompt, past the argument parsing: config + pack + the full greedy forward
/// (the CLI above and the parity test src/mimo/mimo_engine_parity.cpp drive the same code).  `out` receives
/// the generated ids, starting with the argmax after the prompt.  Returns 0, non-zero with `err` set.
int run_greedy(const char* exl3_dir, const int32_t* prompt, int64_t n_prompt, int n_gen, int64_t ctx, int chunk,
               std::vector<int32_t>& out, std::string& err);

}  // namespace strata::mimo
