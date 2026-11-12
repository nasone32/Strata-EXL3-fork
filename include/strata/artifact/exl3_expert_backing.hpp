// Optional routed-expert backing shared by CPU and GPU model accessors.
// Native/dense accessors need no concrete packed-store dependency.
#pragma once

#include <cstddef>
#include <string>

#include "strata/core/exl3_layout.hpp"

namespace strata {

struct Exl3Linear;

class Exl3ExpertBacking {
public:
    virtual ~Exl3ExpertBacking() = default;
    virtual Exl3TrellisLayout layout() const noexcept = 0;

    // Returned pointers belong to this backing and stay stable for its lifetime.
    // False means the name is outside the backing (e.g. a dense projection).
    virtual bool linear(const std::string& base, Exl3Linear& out) const = 0;

    // Must describe the SAME physical bytes as linear(), including all three planes.
    // Expert staging may use pread instead of touching the mapped pointer.
    virtual bool locate(const std::string& tensor, std::string& path,
                         size_t& offset, size_t& bytes) const = 0;

    // A managed RAM source must NOT fall through to a file when locate returns
    // false. Default methods preserve every existing file-backed caller.
    virtual bool memory_only() const noexcept { return false; }
    virtual bool covers_tensor(const std::string&) const { return false; }

    // Direct-DMA eligibility (R1 prototype, opt-in STRATA_EXL3_RESIDENT_DIRECT_DMA, default
    // off): may an ASYNC device copy read [ptr, ptr+bytes) straight out of host memory this
    // backing owns - TRUE only when the WHOLE range lies inside ONE committed span the
    // backing's registrar declared GPU-registered page-locked RAM and the backing promises
    // the contents immutable for as long as the answer stays true (its own state transitions
    // turn it false again).  This is the backing answering for bytes it owns - never a
    // pointer-shape/pinnability guess by the caller - so the default (false) keeps every
    // existing backing, every native mmap and every file source on the established
    // gather/staging path.  Foreign pointers are expected and answered false: the
    // implementation must compare/subtract INTEGERS (uintptr_t), never the pointers
    // themselves.  Const and side-effect free: callers may ask per staged expert.
    virtual bool direct_dma_pinned(const void* ptr, size_t bytes) const noexcept {
        (void) ptr; (void) bytes; return false;
    }
};

}  // namespace strata
