// Optional decode-only ROCTX ranges for profiling. No ROCm dependency in normal
// builds: Linux profilers preload the ROCTX library to resolve these weak APIs.
// STRATA_TRACE_DECODE_REQUESTS=2,4 selects the warm requests of a two-prompt run.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__) && defined(__GNUC__)
extern "C" {
uint64_t roctxRangeStartA(const char*) __attribute__((weak));
void roctxRangeStop(uint64_t) __attribute__((weak));
int roctxProfilerResume(uint64_t) __attribute__((weak));
int roctxProfilerPause(uint64_t) __attribute__((weak));
}
#endif

namespace strata::platform {
class DecodeTrace {
    uint64_t range_ = 0;
    bool active_ = false;
public:
    DecodeTrace() {
#if defined(__linux__) && defined(__GNUC__)
        const char* selection = std::getenv("STRATA_TRACE_DECODE_REQUESTS");
        if (!selection || !*selection) return;
        static unsigned request = 0;
        const unsigned current = ++request;
        bool selected = std::strcmp(selection, "all") == 0;
        for (const char* p = selection; !selected && *p;) {
            char* end = nullptr;
            const unsigned long n = std::strtoul(p, &end, 10);
            if (end == p) break;
            selected = n == current;
            if (*end != ',') break;
            p = end + 1;
        }
        if (!selected) return;
        if (!roctxRangeStartA || !roctxRangeStop || !roctxProfilerResume || !roctxProfilerPause) {
            std::fprintf(stderr, "strata trace: preload librocprofiler-sdk-roctx for decode tracing\n");
            return;
        }
        char name[64];
        std::snprintf(name, sizeof name, "strata.decode.request%u", current);
        roctxProfilerResume(0);
        range_ = roctxRangeStartA(name);
        active_ = true;
#endif
    }
    DecodeTrace(const DecodeTrace&) = delete;
    DecodeTrace& operator=(const DecodeTrace&) = delete;
    void stop() {
#if defined(__linux__) && defined(__GNUC__)
        if (active_) {
            roctxRangeStop(range_);
            roctxProfilerPause(0);
            active_ = false;
        }
#endif
    }
    ~DecodeTrace() { stop(); }
};
}  // namespace strata::platform
