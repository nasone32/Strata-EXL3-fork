// include/strata/mimo/peer_reserve.hpp - STRATA_MIMO_PEER_RESERVE_MIB parsing, shared by
// the engine (src/mimo/mimo_engine.cu setup_peer) and mimo_config_test.  The value is the
// peer tier's free-VRAM floor in MiB (free space left on the card; what anything else
// already allocated - a desktop, another process - is already absent from "free").
#pragma once

#include <cerrno>
#include <cstdlib>
#include <string>

namespace strata::mimo {

/// Parses one full, nonempty base-10 integer in [1024, 16384] (MiB).  Rejects empty
/// input, trailing garbage (strtol alone would read "1024garbage" as 1024), overflow
/// (errno ERANGE) and out-of-range values; returns false with `err` set.
inline bool parse_peer_reserve_mib(const char* v, long long& out, std::string& err) {
    if (v == nullptr || *v == '\0') {
        err = "STRATA_MIMO_PEER_RESERVE_MIB is empty";
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long long r = std::strtoll(v, &end, 10);
    if (errno == ERANGE || end == v || *end != '\0') {
        err = "STRATA_MIMO_PEER_RESERVE_MIB must be one plain integer (MiB)";
        return false;
    }
    if (r < 1024 || r > 16384) {
        err = "STRATA_MIMO_PEER_RESERVE_MIB must be 1024..16384 MiB";
        return false;
    }
    out = r;
    return true;
}

}  // namespace strata::mimo
