// include/strata/artifact/exl3_band8_store.hpp - persistent shared CPU/GPU band-8 expert cache.
//
// A Band8 store packs the trellis planes of routed experts (notes/EXL3-BAND8-FEASIBILITY-2026-11-06.md)
// into ONE mmap-able payload file inside an EXPLICIT per-fingerprint cache directory (typically on
// a different drive than the model): a pure tile permutation, byte-for-byte size-preserving, with
// suh/svh scales copied verbatim. CPU decode reads the mapped pointers directly; GPU staging can
// pread the SAME file ranges through locate() - no native copy exists.
//
// The concrete store is deliberately OUTSIDE exl3_model.hpp: Exl3Model only sees the abstract
// Exl3ExpertBacking, so native/CPU-only builds keep compiling and linking without this TU. The
// MiMo library links strata_exl3_band8_store; nothing else has to.
//
// Memory contract: a store holds exactly ONE immutable mmap - the payload. The manifest is parsed
// at open and dropped. All lookups are const and thread-safe. The CALLER-owned store must outlive
// any CPU job or GPU staging stream that consumed linear()/locate() results.
//
// Concurrency contract: cache_dir/<fingerprint-hex16>/ is an immutable GENERATION. A different
// model/prefix-list builds a different directory and never replaces another generation's bytes,
// so a reader that opens while another build is publishing sees either the complete old
// generation or the complete new one - never a mixed pair. Old generations are never auto-deleted
// (they are not this cache's to remove); locate()/pread paths of live owners stay valid.
//
// Scope: POSIX mmap (like safetensors.hpp); the Windows branch is a clean compile-time stub that
// throws at open, because no Windows machine was available to test against.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "strata/artifact/exl3_expert_backing.hpp"
#include "strata/core/exl3_layout.hpp"

namespace strata {

struct Exl3Band8BuildOptions {
    /// Hard per-projection cap on a NATIVE trellis plane held in RAM while packing. The packed
    /// twin buffer is the same size, so peak working set is about 2x this plus io_block_bytes.
    /// MiMo's largest plane is 2.5 MiB (K2.5, 128x256 tiles); 32 MiB leaves headroom for Qwen.
    size_t max_projection_bytes = 32u << 20;
    /// pread/write chunk for the streaming copy (bounded peak RSS on 80 GiB builds).
    size_t io_block_bytes = 4u << 20;
    /// stderr progress line after every N projection entries. MiMo integration packs
    /// 47 MoE layers x 256 experts x 3 projections = 36,096 entries (108,288 plane records,
    /// ~79.5 GiB); the default 256 keeps that to ~141 lines, and callers override it anyway.
    size_t progress_every_triples = 256;
    /// Record a rolling checksum of the payload while building (cheap, streaming). Only stores
    /// that were built WITH it can be byte-verified on open; without it, open validates
    /// bounds/metadata/fingerprint only (documented detection boundary, see verify_payload).
    bool record_payload_checksum = false;
    /// Stream the whole payload on open and refuse on a checksum mismatch. Requires a cache
    /// built with record_payload_checksum; otherwise opening throws instead of silently skipping.
    bool verify_payload = false;
    /// Read-only open: REFUSE instead of building when the generation is absent, invalid,
    /// fingerprint-mismatched or incomplete - and never create the cache/generation
    /// directories, clear partials or take the build lock. For callers that must not write as
    /// a side effect (benchmarks, inspection): a missing ~80 GiB generation then fails loudly
    /// instead of becoming an implicit pack job. Default true keeps every existing caller.
    bool allow_build = true;
    /// Test seam for the publish window: invoked after the payload file has been renamed into
    /// its generation but BEFORE the manifest (the completeness marker) is written. Tests use it
    /// to prove a concurrent reader of ANOTHER generation is unaffected and that a source change
    /// here aborts the publish. Production leaves it unset.
    std::function<void()> after_payload_publish_for_tests;
    /// Test seam for the low-disk preflight: when set, this free-byte count replaces statvfs so
    /// the "clean own partial + report needed bytes" failure can be exercised on a full disk.
    /// Production leaves it nullopt.
    std::optional<uint64_t> reported_free_bytes_for_tests;
};

/// Persistent band-8 packed-expert cache. Immutable after build_or_open returns.
class Exl3Band8Store : public Exl3ExpertBacking {
public:
    /// Build the cache generation if absent/stale/corrupt, or open the existing one when its
    /// fingerprint, format version and requested-prefix coverage all match. NEVER falls back to
    /// native experts: a requested prefix that cannot be packed or is missing from the cache
    /// throws.
    ///
    /// `expert_prefixes` is caller-generated and explicit, ONE ENTRY PER LINEAR PROJECTION
    /// (e.g. "model.layers.0.mlp.experts.7.gate_proj"); each entry contributes three PLANE
    /// records (.trellis/.suh/.svh). MiMo integration: 36,096 entries -> 108,288 plane records.
    /// The list ORDER is part of the layout and of the fingerprint. Each entry must have
    /// .trellis/.suh/.svh in the source index; duplicates are refused. Source validation: mul1
    /// codebook, H128-aligned dims, and only the four feasibility-verified rates (K=2, 2.5, 3, 4)
    /// per projection.
    ///
    /// Cache layout: cache_dir/<16-hex fingerprint>/{payload.bin, manifest.bin, build.lock}.
    /// The fingerprint (schema+format version, canonical model dir, index.json bytes+size+mtime,
    /// every referenced shard's canonical path/size/mtime, exact prefix list order) names an
    /// immutable generation. It is a fast, ordered change detector - NOT a cryptographic
    /// identity: an adversarial payload edit that preserves every hashed datum can still alias.
    /// Changed sources are normally detected and rebuilt into a NEW generation; the old
    /// generation's files are left in place for live readers.
    static std::shared_ptr<const Exl3Band8Store> build_or_open(
        const std::string& model_dir,
        const std::string& cache_dir,
        const std::vector<std::string>& expert_prefixes,
        const Exl3Band8BuildOptions& opts = {});

    ~Exl3Band8Store() override;
    Exl3Band8Store(const Exl3Band8Store&) = delete;
    Exl3Band8Store& operator=(const Exl3Band8Store&) = delete;

    Exl3TrellisLayout layout() const noexcept override { return Exl3TrellisLayout::Band8; }

    /// Packed lookup by projection base name; out.trellis/suh/svh point into the payload mmap
    /// (stable for this store's lifetime), out.layout == Band8, out.cb == 2 (mul1-only store).
    /// Returns false for any name the store does not cover (dense projections, unpacked experts).
    bool linear(const std::string& base, Exl3Linear& out) const override;

    /// File span of the SAME physical bytes linear() hands out, for pread-based GPU staging:
    /// `path` is the cache payload file; offset/bytes are validated record bounds.
    /// Covers `<base>.trellis`, `<base>.suh`, `<base>.svh` of packed experts; false otherwise.
    bool locate(const std::string& tensor, std::string& path,
                size_t& offset, size_t& bytes) const override;

    size_t record_count() const;         // PLANE records (.trellis/.suh/.svh): 3 per prefix entry
    uint64_t fingerprint() const;        // generation fingerprint the store was validated against
    /// Payload file path; parent_path() is the immutable generation directory
    /// cache_dir/<fingerprint-hex16>/. Derive paths from this, not from a hardcoded root.
    const std::string& payload_path() const;

private:
    Exl3Band8Store() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
