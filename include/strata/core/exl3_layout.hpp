// Physical compressed EXL3 trellis layout; this is not a quantization setting.
#pragma once

#include <cstdint>

namespace strata {

enum class Exl3TrellisLayout : uint32_t {
    Native = 0,
    // Tile (kt, nt) is stored at ((nt / 8) * ki + kt) * 8 + nt % 8.
    // Words within each tile and the suh/svh scales are unchanged.
    Band8 = 1,
};

inline constexpr uint32_t kExl3Band8FormatVersion = 1;

}  // namespace strata
