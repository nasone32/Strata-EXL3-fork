#pragma once

#include "strata/artifact/exl3_model.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>

namespace strata {

struct Exl3WarmRange {
    std::string path;
    size_t offset = 0, bytes = 0;
};

// Startup only. Resolve through locate(), so packed experts warm the packed file,
// never the native shadow. The caller excludes peer residents AFTER placement;
// primary residents remain included because main-cache evictions can reuse them.
inline std::vector<Exl3WarmRange> exl3_host_warmup_plan(
        const Exl3Model& model, const std::vector<std::string>& prefixes) {
    std::vector<Exl3WarmRange> ranges;
    ranges.reserve(prefixes.size() * 3);
    for (const auto& base : prefixes) {
        for (const char* plane : {".trellis", ".suh", ".svh"}) {
            Exl3WarmRange r;
            if (!model.locate(base + plane, r.path, r.offset, r.bytes) || !r.bytes ||
                    r.offset > std::numeric_limits<size_t>::max() - r.bytes)
                throw std::runtime_error("expert host warmup: missing/invalid range " + base + plane);
            ranges.push_back(std::move(r));
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return a.path < b.path || (a.path == b.path && a.offset < b.offset);
    });
    size_t count = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto& r = ranges[i];
        if (count && ranges[count - 1].path == r.path &&
                r.offset <= ranges[count - 1].offset + ranges[count - 1].bytes) {
            auto& prev = ranges[count - 1];
            prev.bytes = std::max(prev.offset + prev.bytes, r.offset + r.bytes) - prev.offset;
        } else {
            if (count != i) ranges[count] = r;
            ++count;
        }
    }
    ranges.resize(count);
    return ranges;
}

// One bounded buffer; no pinning, second expert cache, cache eviction/advice, or
// changes to expert residency/frequency. Checked reads make a short warmup fatal.
inline uint64_t exl3_host_warmup_read(const std::vector<Exl3WarmRange>& ranges) {
    std::vector<char> buffer(4 << 20);
    std::ifstream file;
    std::string path;
    uint64_t bytes = 0;
    for (const auto& r : ranges) {
        if (r.path != path) {
            file.close();
            file.clear();
            file.open(r.path, std::ios::binary);
            if (!file) throw std::runtime_error("expert host warmup: cannot open " + r.path);
            path = r.path;
        }
        if (r.offset > (size_t) std::numeric_limits<std::streamoff>::max())
            throw std::runtime_error("expert host warmup: offset overflow");
        file.seekg((std::streamoff) r.offset);
        size_t left = r.bytes;
        while (left) {
            const size_t n = std::min(left, buffer.size());
            if (!file.read(buffer.data(), (std::streamsize) n))
                throw std::runtime_error("expert host warmup: short/error read " + path);
            left -= n;
            bytes += n;
        }
    }
    return bytes;
}

} // namespace strata
