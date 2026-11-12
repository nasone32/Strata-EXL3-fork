#pragma once
#include "strata/artifact/exl3_model.hpp"
#include <array>
#include <functional>
#include <vector>

namespace strata {

// Managed resident backing, independent of the physical EXL3 format. Like the
// GGUF arena, main-GPU experts retain a RAM copy. Optional peer-complement mode
// returns evictions from VRAM and releases incoming experts only after publish.
// One fixed virtual address per expert: CPU descriptors NEVER need rebinding.
// GPU-only addresses are PROT_NONE: they are metadata-only until copy-back, so
// an ownership bug faults instead of silently decoding anonymous zero pages.
// Mutations are ONLY between drained verifier windows, never from service jobs.

// Runtime in-memory repack: while loading the arena, convert the routed experts'
// NATIVE trellis planes into the canonical Band8 tile order straight into the
// final registered/pinned RAM - the same permutation the offline band-8 store
// proves (exl3_band8_permute.hpp), with NO disk cache, descriptor or payload
// file anywhere. The arena ends up byte-identical to one loaded from a packed
// store, so the CPU tier and every GPU fill read one canonical copy. Scales are
// copied verbatim. Source must be native-layout shards (an already-packed Band8
// source backing is refused - it is canonical as-is); layout() reports the FINAL
// Band8 tag from create() on; between create() and activate() linear()/locate()
// THROW rather than hand out native bytes under a Band8 mode probe; activation
// supports the FULL arena only (a peer-complement set would need on-demand loads
// of unconverted bytes, which this first version refuses explicitly).
enum class Exl3ResidentRepack : uint8_t { None = 0, ToBand8 = 1 };

class Exl3ResidentBacking : public Exl3ExpertBacking {
public:
    struct Group { int layer, expert; std::array<std::string, 3> projections; };
    struct Options {
        uint64_t available_bytes = 0, headroom_bytes = 4ull << 30;
        // Production requires every committed range pinned/locked. Tests may
        // supply a bounded fake registrar; no GPU dependency in this artifact.
        std::function<bool(void*, size_t, std::string&)> lock;
        std::function<void(void*, size_t)> unlock;
        // The registrar's OWN declaration that its lock callback GPU-REGISTERED every
        // committed range (cudaHostRegister), making the arena readable by an async device
        // copy.  A TRUSTED-CALLER assertion, not something the artifact can verify: whoever
        // supplies the lock/unlock pair is asserting what it really did - only the
        // production CUDA registrar (exl3_resident_host_options) sets it.  A test MAY set
        // it beside a fake lock to exercise the wiring; that does not make the bytes pinned,
        // and such an arena must never be DMA'd in a real run.  Default false keeps
        // direct_dma_pinned() off for every other registrar (R1 prototype:
        // STRATA_EXL3_RESIDENT_DIRECT_DMA).
        bool gpu_registered = false;
        bool direct_io = true;
        std::function<void(size_t, size_t)> progress;
    };
    static std::shared_ptr<Exl3ResidentBacking> create(
        const std::string& model_dir, const std::vector<Group>& groups,
        std::shared_ptr<const Exl3ExpertBacking> source = {},
        Exl3ResidentRepack repack = Exl3ResidentRepack::None);
    ~Exl3ResidentBacking() override;
    Exl3TrellisLayout layout() const noexcept override;
    bool linear(const std::string&, Exl3Linear&) const override;
    bool locate(const std::string&, std::string&, size_t&, size_t&) const override;
    bool memory_only() const noexcept override;
    bool covers_tensor(const std::string&) const override;
    bool direct_dma_pinned(const void* ptr, size_t bytes) const noexcept override;

    // Startup only, before CPU registration/READY. File-backed source: call after the
    // initial GPU placements are drained (the source mapping is retired inside). With
    // Exl3ResidentRepack::ToBand8 the arena IS the only packed copy: activate BEFORE any
    // consumer or GPU fill exists (the supported lifecycles do exactly that), and
    // `gpu_only` must be empty - the full arena only this first version.
    void activate(const std::vector<std::pair<int, int>>& gpu_only, Options options);
    bool active() const noexcept;
    bool resident(int layer, int expert) const;
    bool has_gpu_only() const;
    uint64_t required_bytes(const std::vector<std::pair<int, int>>& gpu_only) const;
    // Return target is writable/pinned but unpublished until finish_return.
    // Failure keeps the GPU owner; cancel_return releases an unfinished target.
    bool prepare_return(int layer, int expert);
    void finish_return(int layer, int expert);
    void cancel_return(int layer, int expert);
    void release_to_gpu(int layer, int expert);
    struct Stats { uint64_t virtual_bytes, resident_bytes, locked_bytes, file_reads,
                            file_bytes, returned, released; };
    Stats stats() const;
private:
    Exl3ResidentBacking();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata
