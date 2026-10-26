#include "strata/kernels/cpu/f16_convert.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    // Scramble the complete fp16 space so exceptional and ordinary lanes mix.
    std::vector<uint16_t> src(65540);
    std::vector<float> dst(src.size() + 2, 123.0f);
    for (size_t i = 0; i < src.size(); ++i) src[i] = (uint16_t) (i * 40503u);
    for (size_t offset = 0; offset < 4; ++offset) {
        for (size_t tail = 0; tail < 4; ++tail) {
            const size_t n = 65536 - tail;
            std::fill(dst.begin(), dst.end(), 123.0f);
            strata::kernels::cpu::f16_to_f32_exact(src.data() + offset, dst.data() + 1, n);
            if (dst.front() != 123.0f || dst[n + 1] != 123.0f) return 1;
            for (size_t i = 0; i < n; ++i) {
                const float ref = strata::kernels::f32_from_f16(src[offset + i]);
                if (std::memcmp(&ref, &dst[i + 1], sizeof(float)) != 0) {
                    std::printf("FAIL fp16=%04x offset=%zu tail=%zu\n", src[offset + i], offset, tail);
                    return 1;
                }
            }
        }
    }
    std::puts("f16_convert_test PASS: all 65536 patterns, unaligned inputs/outputs and tails, exact scalar bits");
    return 0;
}
