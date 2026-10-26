# EXL3 on HIP (gfx1100): build, baseline and A/B ledger

Machine: Ryzen 9 7950X, 93 GB RAM, RX 7900 XTX (GPU0, PCIe x16, headless) + RX 7900 XTX (GPU1,
chipset PCIe x4, drives a display), system ROCm 7.14 (`/opt/rocm/core-7.14`, clang 23, hipBLASLt
1.4.1 unless said). Model: `Qwen3.8-Flash-Next-exl3-4bpw` (4.05 bpw, EXL3 safetensors, K=4 experts /
K=6 dense, `ng6` PLE; 9 shards, 64.5 GB, on a 2 TB NVMe at ~94% full, hash-verified at download),
tokenizer pack `Strata-data/packs/exl3-4bpw/tokenizer`, PLE through the same directory's
`ngram_embedding.safetensors` (39 GB, **merged single-trellis layout**, see the loader note below).
MTP drafter: `Strata-data/mtp/rt` (packed Qwen3.8 drafter).

Base: Ryutaku `be2ad283` (branch `exl3-hip-rdna3`). Two commits on top of it fix the HIP build and
the merged PLE layout (`de2e62dc`, `6ba9e40e`). Every row below is the build of this branch at the
listed commit. `warm_session.py` runs two GENs in one engine session: turn 0 fills the expert cache
(page cache warm from earlier runs), turn 1 reuses the prompt KV and measures warm decode; the DONE
line is `generated read prefill_ms decode_ms ...`.

## Build (M0)

```sh
cmake -S . -B build-hip-exl3 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DSTRATA_GGML_DIR=<a local llama.cpp checkout>   # pinned 3cf0325, offline
cmake --build build-hip-exl3 -j 6
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/lib \
HIP_VISIBLE_DEVICES=0,1 ctest --test-dir build-hip-exl3 -R 'exl3|ple_exl3'
```

`-DCMAKE_PREFIX_PATH=/opt/rocm` is needed on this split-layout ROCm (the plain docs command fails to
find `hipConfig.cmake`). First build failure: `src/kernels/exl3_moe_window_bench.cpp` uses
`__half_as_ushort` in host code, which ROCm's gcc host fp16 header lacks — fixed in the
`hip_compat/cuda_fp16.h` shim (commit `de2e62dc`); no other source change was needed for the build.

**Reproducible test invocation - both environment settings in the block above are REQUIRED on this
machine; a plain `ctest` fails for environment reasons, not kernel ones:**

1. `LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/lib` — the
   split-layout ROCm 7.14 is not on the default linker path, so without it the test executables die
   at load time (`libhipblas.so.3` / hipBLASLt not found).
2. `HIP_VISIBLE_DEVICES=0,1` — indices 0 and 1 are the two gfx1100 XTX cards; device 2 is the
   gfx1036 iGPU.  `exl3_moe_multi_parity` deliberately iterates EVERY visible device and ABORTS on
   one it cannot run: the installed rocBLAS carries no gfx1036 TensileLibrary kernels and this is a
   gfx1100-only build.  Aborting on unsupported hardware rather than silently skipping it is the
   intended behavior; restricting visibility to the supported cards is an explicit operator
   choice, documented here instead of hidden inside the test.

Independently confirmed on the final tree (restricted rerun, 24/24 PASS in 45.45 s) with the
equivalent top-level library path:

```sh
HIP_VISIBLE_DEVICES=0,1 \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
ctest --test-dir build-hip-exl3 --output-on-failure -R 'exl3|ple_exl3'
```

Tests (ROCm 7.14, both XTX visible): **exl3/ple_exl3 24/24 pass** (`exl3_route_plan_test`, loader
validation, parity K=1..8 CPU, `exl3_gpu_parity`, `exl3_gemv_parity`, `exl3_moe_multi_parity`
26/26 on both cards, `ple_exl3_parity`). Whole suite: 127 pass, 2 fail (`expert_parity`,
`pool_test`: need a `pack/full/experts.bin` fixture), 5 skip (fixtures / hipBLASLt-table tests).
`exl3_moe_multi_parity` iterates *all* devices and aborts on this machine's third GPU (gfx1036
iGPU, no rocBLAS kernels) — run it with `HIP_VISIBLE_DEVICES=0,1`.

## Functional smoke (M0, GPU0 only)

- CLI `--tokens` (capital-of-France prompt), `--expert-cache 0`: answers
  `<think>\n\n</think>\n\nThe capital of France is **Paris**.<|im_end|>` — 3.02 tok/s decode, 1.82
  tok/s prefill (every expert streamed).
- Same prompt, `--expert-cache 128`: identical token stream, 16.23 tok/s decode, 12.10 tok/s prefill.
- `--serve` + verifier windows (bench.py, 512-token needle prompt, needle found):
  `--expert-cache auto --prefill 8192 --max-context 65536 --kv int8 --spec 4 --ple-io mmap
  --vram-reserve-mib 2048` → 94 slots/layer (10.6 GiB), first (cold) prompt 18.4 tok/s,
  decode 13.8 tok/s.
- `--serve --layer-split 24 --split-device 1` (explicit boundary, both XTX): needle found,
  260 slots/layer over both cards. Functional; slower on a cold prompt (below).
- The 4 bpw checkpoint's `ngram_embedding.safetensors` stores **one** `...ngram_embedding.trellis`
  `[320001536, 61]` instead of 128 `shard_N.trellis` tensors; commit `6ba9e40e` accepts it
  (one-tensor case of the same contiguity check). Without it the engine refuses to start.
- Native GGUF regression reference (not an EXL3 A/B): `ud-q4_k_xl` pack serve smoke on this build,
  512-token needle prompt passed, 198.6 tok/s cold prefill, 31.5 tok/s decode. A CLI
  `--tokens` GGUF run with `--spec` degenerates to an immediate `<|im_end|>` **identically on the
  stock 0.1.38 engine**, so it is an invocation artifact of that path, not a regression here.

## Baseline numbers (M1)

All `--serve` through `tools/exl3/bench.py` / `tools/exl3/warm_session.py`, greedy (`temperature=0
seed=7319`), same 512-token needle prompt, KV int8, PLE mmap, GPU0 only unless said. PCIe probe:
28.6 GB/s host→device (GPU0 x16). "Warm" = second GEN in the same session.

| configuration | slots/layer | PP 512 cold (tok/s) | TG warm (tok/s) | notes |
|---|---|---|---|---|
| ctx 65536, prefill 8192 | 94 | 51.4 | 18.3 | first serve config |
| ctx 8192, prefill 2048 | 121 | 51.1 (9991 ms) | 30.9 / 32.1 / 32.2 → **32.1** | 3 sessions |
| same + `--mtp rt --spec 4 --spec-min-p 0.5` | 121 | 50.6 (10239 ms) | 33.6 / 33.2 / 33.7 → **33.6** | 27/27 drafts accepted (needle prompt is predictable) |
| same + `--expert-ram-cache 40960` | 121 (+361 RAM) | 32.5 (15761 ms, the RAM tier fills) | 30.8 | no warm gain; cold PP 1.6x slower |
| layer split 24, both GPUs, ctx 8192 | 260 (both cards) | 25.0 (20539 ms) | 32.3 | 1 session; cold PP much slower |

Longer prompt (fresh engine each run, page cache warm): 8192-token prompt on the ctx-65536 config:
**240.9** and **314.7 tok/s** (34.0 s, 26.0 s); decode after it 12.7 tok/s (94-slot cache, miss-heavy
after a cold prompt). Decode after the 512-token prompt on the same config: 18.2-18.3 tok/s.

Resident-FFN microbench (idle GPU, no uploads): `exl3_moe_window_bench` = 0.820 ms/layer-window at
40 routes, 1.417 ms at 80 routes (the 1.219 ms at 40 routes measured while the other card was busy
is 1.5x that — keep the GPU idle for kernel A/B).

### Where the time goes (STRATA_EXL3_TIME=1)

Warm decode, ctx 8192 config, 6-token verifier window, 48 layers summed: **pool ≈ 95-108 ms,
upload 46-54 ms (280-310 MiB), FFN 47 ms, cache hit 1618/1737 (93%)**. The window wall (≈145 ms)
matches pool+FFN, so warm TG is ~2/3 host-side miss servicing (page-cache `pread` + H2D + sync
waits) and ~1/3 expert FFN kernels. Even at 93% hit, ~120 misses × 2.49 MiB = ~300 MiB per window
must move per 6 tokens.

Cold prefill (512-token chunk, first chunk of the prompt): pool 8972 ms (upload 4714 ms, 36.6 GiB),
FFN 3510 ms — compulsory misses dominate (a 512-token prompt routes through ~15.5k of the 24.6k
expert slots), and the FFN reconstruct+hipBLAS group GEMM is the other ~35% of the chunk.

Bottleneck ranking that follows: (1) expert miss servicing dominates both cold PP and even warm TG;
(2) the routed FFN kernels are ~1/3 of warm TG and ~35% of PP chunks; (3) everything else (dense
Q8, attention, hyper-connections) is behind these two on this workload. The RAM tier at 40 GiB does
not help warm TG here (misses already hit the page cache) and hurts cold PP by filling 40 GiB
while serving; a smaller tier or a freeze policy might behave differently (M4 material, unproven).

## Kernel A/B ledger (M2+)

| # | date | candidate commit | A (baseline) | B | metric / regime | result |
|---|---|---|---|---|---|---|
| 1 | 2026-10-16 | `6ba9e40e` | — | — | `exl3_moe_window_bench` 40/80 routes, idle GPU | A-reference: 0.820 / 1.417 ms/layer-window |
| 2 | 2026-10-16 | see below | `exl3_moe_group_gemm` (reconstruct→FP16 + hipBLAS) | `exl3_moe_group_wmma` (trellis decoded into `v_wmma_f32_16x16x16_f16` fragments; `STRATA_EXL3_WMMA=1`) | group FFN, one expert, 2560/640 | **B 1.8-2.5x** (see table) |
| 3 | 2026-10-16 | see below | same A | same B | real model, 8192-token prompt, GPU0, ctx 65536 | **B +21.5% PP** (386.8 vs 318.5 tok/s) |
| 4 | 2026-10-17 | see below | threshold 16 (8..15 rows → indexed GEMV) | threshold 8 (8..15 rows → WMMA) | real model, same config, 3 alternating runs/arm | **no gain**: 394.0 vs 392.2 tok/s; kept 16 |
| 5 | 2026-10-17 | see below | device-FFN stream sync per (group,tile) | one sync per wave + wave-batched route tables (`STRATA_EXL3_WAVESYNC=1`) | real model PP, WMMA and baseline regimes | **B +2.2-2.4% PP8k**, +5-6% PP512; kept opt-in |
| 6 | 2026-10-17 | see below | LRU cache (admits decode misses) | frozen warm-decode cache (misses stage via scratch, no eviction) | warm TG, PP->TG, needle and essay, 121 and 32 slots | **A wins 2.9-4.2x**; freeze removed |
| 7 | 2026-10-17 | see below | WMMA separate gate/up calls | paired gate+up launches (fork v4 structure, K=32 pairing excluded) | group FFN microbench + real PP | **no gain**: -1% PP8k; pair removed |
| 8 | 2026-10-17 | see below | single-k-tile mma loop | K=32 mma pairing on the single projection | microbench + real PP8k 6+6 | **B +0.9% PP8k** (12/12 ordered); promoted |

### A/B #2/#3 detail: the gfx11 matrix-core group FFN (M2)

B is nasone32's `exl3_wmma_mul1_routed_v4` technique (MIT; the trellis decode from turboderp's
ExLlamaV3) adapted to Strata's per-(expert-group, tile) calls: the packed tile's tensor-core
interleave is decoded straight into the WMMA B operand (A tiles staged in LDS, 512 threads = 16
column warps, one block per (row-tile, 256-column band)), so no FP16 weight is materialized.
Strata's host route plan already groups each expert's routes, so the fork's GPU CSR route table is
unnecessary. Opt-in `STRATA_EXL3_WMMA=1` at the count>=16 dispatch; falls back to A on non-gfx11
hardware, mcg/3inst codebooks, K outside 3..6 or non-128-multiple shapes. The B-operand slot
mapping was re-verified on this card by solving for the effective matrix (128 basis rows against a
CPU-decoded `What`): the recovered rows match `What` at rel 2.9e-05 exactly through the fork's
shuffle.

Microbench (idle GPU0, one expert group, `exl3_moe_window_bench <rows> 0 group <bits>`, K4 = the
4bpw model's experts; K6 similar):

| rows | A ms/group | B ms/group | B/A |
|---|---|---|---|
| 16 | 0.393 | 0.147 | 0.37 |
| 64 | 0.399 | 0.185 | 0.46 |
| 256 | 0.591 | 0.212 | 0.36 |
| 455 | 0.623-0.671 | 0.266 | 0.41 |

Real model (same config as the M1 baseline: ctx 65536, KV int8, 94 slots/layer, fresh engine per
run, interleaved B-A-B-A, warm page cache):

- 8192-token prompt: B 21150.5 / 21245.1 ms (**386.8 tok/s** median) vs A 25729.5 / 25693.5 ms
  (318.5 tok/s) — **B +21.5%**, spread < 1%. *Artifact note: only the second pair survives —
  `bench/results/2026-10-16-hip-exl3-m2/pp8k-B` (21245.1) and `pp8k-A` (25693.5); the first B-A
  pair was overwritten by the same-label re-run, so 21150.5/25729.5 are historical numbers without
  raw artifacts. The surviving pair, the untracked `pp8k-A-24619` (25882.0) and the 2026-10-17
  default-path control (below) all sit in the same 25.7-25.9 s band.*
- Warm 512-token session (turn 2, decode through the verifier windows): A 1988.6 / 1899.9 ms vs
  B 1955.7 / 1811.4 ms for 37 tokens — B within noise ahead, as expected: decode windows route
  ≤6 rows per expert and take the indexed GEMV path in both arms.
- Correctness: `exl3_moe_multi_parity` now checks the WMMA group FFN against the independent CPU
  reference at K4/K6 × rows 16/40 with token-scattered routes — rel 6.8e-4..7.9e-4, the same
  band as the existing GEMV/GEMM paths; the raw-matrix probe above bounds it at 2.9e-5 before the
  FFN's f16 rounding. Greedy output of a repeat 8192-run is identical between the arms.
- Tests after the change: exl3/moe/hip subset 40/40 pass (2 pre-existing hipBLASLt-table skips).

Verdict: **B kept, opt-in** (`STRATA_EXL3_WMMA=1`; the default path stays byte-identical, per this
repo's opt-in rule for new features). Measured on gfx1100 only; the device gate admits the whole
gfx11 wmma family (gfx1100/1101/1102/1103/1150/1151) but no other card has run it. The gate itself
caches the answer **per device id** (successful queries only; a failed one is retried), so a
mixed-gfx process cannot carry one device's answer to another.

One bug found on the way, now a locked regression test: the down projection's gather must read the
FFN's route-major silu slab directly — indexing it with the token-row table (as gate/up do) reads
out of bounds and faults, which only showed up with prefill groups whose token rows scatter widely
(rows=455 of an 8192-token chunk), not with the identity-row tests.

### A/B #4 detail: the group-count threshold (2026-10-17)

An earlier abandoned experiment had lowered the WMMA entry count to 8 and left a note claiming
"wrong engine output while passing the kernel parity" — with no surviving artifact of any wrong
output. Re-examining the leftovers settled what actually happened: `pp8k-B8` and `pp8k-B16-re`
(untracked, 2026-10-16 00:08/00:12) ran the **same** threshold-8 binary (identical engine sha,
identical token output — both match today's threshold-8 run), so "B16-re" was threshold 8 as well;
its 2.4x slower prefill (51.6 s vs 21.2 s) is the machine's known outlier mode (the M1 A-arm shows
the same shape: `pp8k-A-22621` 56.3 s vs `pp8k-A-24619` 25.9 s, same binary).

The controlled re-test (temporary `STRATA_EXL3_WMMA_MIN` env override, since removed; the fallback
for 8..15 rows was the indexed GEMV, never the reconstruct path; default dispatch untouched):

- Kernel parity first: `exl3_moe_multi_parity` extended to K3/K4/K5/K6 × rows
  7/8/15/16/23/24/31/32/40 (the row_tiles=2 switch at 24 and the 16-row A-tile padding) — 36/36
  PASS at rel 6.1e-4..7.6e-4, the same band as before.
- CLI smoke ("What is 17 + 25?", cache 128): both thresholds answer `42` with identical token IDs.
- 8192-token prompt, idle GPU0, fresh engine per run, alternating th16/th16... 3 runs per arm
  (`bench/results/2026-10-17-hip-exl3-m2b/pp8k-th{16,8}-r{1,2,3}`, engine sha recorded per report):
  **th16 394.0 [393.2..394.7] vs th8 392.2 [391.0..393.1] tok/s PP** — no gain; decode 15.4 both.
  Within each arm the three token streams are identical; between arms the greedy streams diverge
  from token 17 of 32, both remaining coherent — the same FP-path divergence A-vs-B already shows.
- Warm 512-token needle sessions (turn 2, verifier windows): all three keys found at **both**
  thresholds; warm decode 21.1 (th16) vs 21.4 (th8) tok/s — equal.
- Default-path control (no envs, `pp8k-default-r1`): token IDs **byte-identical** to the committed
  A-arm `pp8k-A` output, 323.7 tok/s (the A band).

Verdict: **threshold stays 16**, the temporary override is removed, and the "wrong engine output"
note is retired as unsubstantiated — the claim never reproduced under controlled conditions
(idle GPU, recorded env, unique run dirs). Not measured here: a true cold-page-cache series
(needs root `drop_caches`, not available); all engine runs above are fresh-engine/warm-page-cache
plus the warm-session pair.

### A/B #5 detail: one device-FFN stream sync per wave (2026-10-17)

M1 ranked host-side miss servicing as the top cost; the device-IO path of `run_multi_impl` took a
`cudaStreamSynchronize` after **every (expert-group, tile)** — 15,444 of them for one cold
512-token chunk across 48 layers. B keeps the host route plan, resident-first ordering, wave
pinning and the staging/upload stream exactly as they are, and only changes the barrier
granularity of the compute stream: the wave's route table (input rows, destination rows,
route-major matrices) is built and uploaded once per wave, tiles take pointers into it, and one
sync closes the wave — the same granularity the slot pinning already assumes. The FFN workspace
is pre-reserved to the wave's maximum so no buffer grows (and frees) under queued tiles. Host
route-readback, prefill staging and the host-IO arm (used by decode windows: their counters show
0 device syncs) are untouched; the default per-tile path is unchanged.

Lifetime argument (why the per-tile syncs are removable): all per-tile pageable uploads (the
route-table rows here, and the FFN entry points' own `Exl3Mat` array uploads) read host memory
that now stays unmodified for the whole wave; the next wave's prepare/staging may overwrite
expert slots only after the wave-end sync, exactly as before at tile granularity; tile workspaces
(`batch.*`, the MoE workspace) are reused on the same stream, so stream order serializes them;
prefill consumes `m.Dm` by enqueueing `moe_combine` on the same stream; the existing exception
handler drains both streams before clearing residency, and a drain guard declared after the local
staging vectors syncs the compute stream during unwinding while those vectors are still alive
(pageable-async source lifetime is also covered by the documented runtime contracts - CUDA stages
pageable sources before `cudaMemcpyAsync` returns, and ROCm 7.14's `hipMemcpyAsync` note says a
non-pinned copy "will be performed synchronously" - so the guard is defense in depth; a null
stream handle is the legacy default stream and is drained too).  The FFN scratch is reserved
through narrow seams invoked only by the wave arm, with the call's largest group count: the
group scratch via `Exl3MoeWorkspace::reserve_group_scratch`, and the BASE arrays through the
plain `reserve` - the group entry points call `reserve(rows, ..., indexed=false)` themselves, so
a later, larger tile of a wave must not grow them under earlier queued tiles.  Below 16 no group
tile can run, so neither allocation is made.  Ordinary indexed callers keep the original lazy
`reserve` and never allocate group scratch.  The blocking allocations run once, before the call's
first queued tile; the default arm grows the same buffers lazily mid-dispatch exactly as before
this change, and both converge to the same capacities (VRAM free measured identical, below).
The arm also carries an allocation invariant (`Exl3MoeWorkspace::group_tile_fits`, integer
compares only, checked before every group tile): if a pre-wave reserve ever fails to cover a
tile, the call fails loudly instead of growing - and freeing - under queued work.
`tools/test_exl3_loader.py`'s `WaveSyncTests` writes a synthetic 16-expert one-layer model with
DISTINCT weights and scales per expert and projection and asserts `run_device` output is
byte-identical (memcmp) between the two arms - with explicit `STRATA_EXL3_WMMA` control so an
inherited environment cannot pick the arm, in both kernel regimes (the reconstruct fallback and
the matrix-core arm) - across four waves with slot eviction, the >=16 group arm, the <16 indexed
arm, one 5000-route group split over two successive tiles of the 4096-route tile bound, a
16-row group followed by a 600-row group inside ONE wave (600 exceeds the stream constructor's
512-route base capacity, so a missing base pre-reserve trips the invariant - verified by
removing the reserve and watching the checker fail), and a warm second call; a missing GPU exits
77 (ctest skip), any real failure fails.  Output token streams are byte-identical to the default
arm in both regimes (same kernels, same order - only scheduling changed).

Numbers (idle GPU0, fresh engine per run, warm page cache, alternating, unique run dirs under
`bench/results/2026-10-17-hip-exl3-m3/` and `-m3b/`, engine sha + env recorded per report; same
8192-token essay prompt as A/B #2-4):

- PP8k, WMMA regime (`STRATA_EXL3_WMMA=1`): wave **402.2 [401.5..402.6]** vs default
  393.4 [393.1..393.7] tok/s (3 runs/arm, non-overlapping) — **+2.2%**; re-confirmed at
  **403.1 [402.7..404.9] vs 394.0 [393.6..394.8]** (+2.3%) on the reviewed build, whose sync
  instrumentation does no work unless `STRATA_EXL3_TIME` is set (3 runs/arm,
  `bench/results/2026-10-17-hip-exl3-m3b/`; outputs again byte-identical between arms).
- PP8k, baseline regime (no WMMA): wave **331.7 [331.7..331.9]** vs default 324.0 [324.0..324.5]
  tok/s (2 runs/arm) — **+2.4%**; the WMMA arm's gain stacks (393→402 on top of 324).
- Cold 512-token chunk (needle session, `STRATA_EXL3_TIME=1` diagnostics): syncs **145 vs
  15,444**, host wait in them **2,291 vs 2,929 ms**, pool 7,470 vs 8,048 ms — the pool delta
  matches the removed sync wait.  (The FFN event numbers are NOT comparable between arms: the
  wave arm's interval spans first-tile dispatch through the last scatter, host gaps included,
  while the default arm sums per-tile kernel intervals.)
- PP512 cold prefill (turn 0 of the needle sessions): 8,337 vs 8,853 ms (**+6.2%**);
  cache-0 needle (every expert streamed): 60.1 vs 57.0 tok/s single runs, keys found in both.
- Warm TG: unchanged - decode windows take the host-IO arm (0 device syncs in both); 34.3 vs
  33.5 tok/s single runs, within this config's noise. VRAM identical (1,840 MiB free, 94
  slots/layer, both arms; re-confirmed after each review correction, most recently default 395.3 /
  wave 403.1 tok/s with byte-identical outputs to the baseline,
  `bench/results/2026-10-17-hip-exl3-m3c/` and `-m3d/`).

Verdict: **kept, opt-in** (`STRATA_EXL3_WAVESYNC=1`; default per-tile barrier unchanged, one knob
total). Measured on gfx1100 only; the mechanism is plain CUDA-runtime scheduling with nothing
AMD-specific, but no NVIDIA run has validated it. Follow-up seam if it ever matters: the FFN entry
points still upload their per-tile `Exl3Mat` array slices from the wave-long host arrays - a
device-array variant would remove those small per-tile copies too.

### A/B #6 detail: frozen warm-decode cache vs the LRU (2026-10-17) - negative result

M1 ranked miss servicing ~2/3 of warm TG.  The fork found a frozen decode cache better than its
high-churn adaptive swaps, so the smallest Strata spike froze the cache at the real phase seam -
run_device (prefill) fills the cache under the LRU; the host-IO run_multi windows (the verifier
callback) are decode - and staged decode misses through the existing scratch slots (wave-pinned
residents untouched, RAM tier unchanged, prefill admission unchanged).  STRATA_EXL3_TIME gained
two diagnostics that stay: per-report eviction count and churn misses (a miss for an expert
evicted within the layer's last 16 evictions).

Correctness during the experiment: byte-identical outputs vs the LRU arm across the phase
transitions (window after prefill, second window, second-turn prefill, window after it) on the
distinct-weight fixture, and needle keys found in every engine run of both arms.

Warm TG, ctx 8192 / 121 slots, WMMA+WAVESYNC on both arms, --spec 2 no drafter, needle essay
prompt, 3 alternating sessions per arm (`bench/results/2026-10-17-hip-exl3-m4/`, engine sha +
env per report):

- essay 256 tokens, warm (turn 1): LRU **18.9 tok/s** (13519-13586 ms, spread < 0.5%) vs frozen
  **6.5 tok/s** (39021-39217 ms) - **2.9x slower**.  Turn-0 decode and the warm re-prefill of the
  next turn are slower too (stale set).
- needle 37-token answer, warm: 33.4 vs 8.0 tok/s (4.2x).  Low cache (32 slots, 128 tokens):
  4089 vs 6489 ms.
- Why (STRATA_EXL3_TIME, per token): LRU hits 426-453 of 480 uses (89-94%), 27-54 evictions of
  which only 3-15 are churn, 64-128 MiB uploaded.  Frozen hits 149-165 of 480 (**31-34%**),
  0 evictions, 315-331 experts staged through scratch per token, 744-782 MiB uploaded - the
  decode working set drifts far from the prompt-warmed set, and without admission the recurring
  experts re-stage on every use.  The fork's freeze beat *adaptive swap churn*; against Strata's
  plain LRU at 121 slots the admission is what carries warm decode.

Verdict: **A (the LRU) retained**; the freeze policy and its env knob are removed (the
STRATA_EXL3_TIME eviction/churn diagnostics stay - they produced the explanation above).  The
checker keeps the phase-sequence leg (prefill -> host-IO windows across two turns, byte-identical
across identically-constructed streams - a determinism check, not a cache-policy comparison),
which is also the first coverage of run_multi's host-IO arm.  PP optics unchanged (the prefill
path was never touched); post-removal control: 18.86 tok/s warm, in the LRU band.  Scope: this
compared freeze-at-decode-start against the plain LRU only - prompt-heat admission RANKING and
other admission policies are untested.

### A/B #7 detail: paired gate+up WMMA launches (2026-10-17) - negative result

The fork's v4 runs gate and up in ONE matrix-core launch (projections=2).  Read from its source:
only the launch, the LDS-staging pass and the sync cadence are shared - the two sides' A slabs
(their suh differ) and B fragments (different trellises) are computed separately, and its measured
win rides together with a K=32 mma pairing.  The spike ported exactly that structure (STRATA_EXL3_
WMMA_PAIR=1, temporary): one gather launch (blockIdx.z per side, each applying its own suh), one
routed_wmma_kernel with projections=2 (both slabs staged in one pass, both trellises decoded per
k-tile, no K=32 pairing), one postscale launch (each side applying its own svh); the down
projection unchanged; gate/up must match in bits and codebook or it falls back.  This is a PREFILL
path only (the group arm is device_io, count>=16); warm decode is untouched by construction.

Correctness during the experiment: the paired entry matched the CPU oracle at K3..K6 x rows
7..40 with distinct gate/up weights and distinct suh/svh (36/36 PASS, rel 6.1e-4..7.6e-4, the
established band).

Numbers (idle GPU0, WMMA+WAVESYNC on both arms, unique run dirs
`bench/results/2026-10-17-hip-exl3-m5/`):

- Microbench (one expert group, K4): rows 16: 0.1424 vs 0.1441 ms (+1.2%); rows 64: +0.7%;
  rows 256: 0.2376 vs 0.2134 (**-11%** - at row_tiles=2 the doubled LDS staging and the doubled
  accumulator/B-fragment register pressure cost more than the saved launches).
- PP8k (3 alternating runs/arm): base **403.1 [402.4..406.3]** vs pair 399.2 [398.9..399.4]
  tok/s - **1.0% slower**, non-overlapping; outputs byte-identical.
- PP512 (3 runs/arm, turn-0 prefill): 8973 vs 8976 ms - a wash (overlapping spreads).
- Warm TG control: 13,603 ms / 256 tokens - the LRU band, unchanged as designed.  VRAM 1,840 MiB
  free in both arms.

Verdict: **removed** (negative result; the env knob, the paired entry, its parity block and the
bench arm are gone).  The routed_wmma_kernel keeps the `projections` template dimension - it
compiles to the existing single-projection path - so a retry that includes the fork's K=32 mma
pairing (the piece that plausibly offsets the doubled LDS/registers) is a small change if it is
ever wanted.  Post-removal control: 402.0 tok/s PP8k, in the base band, outputs identical.

### A/B #8 detail: K=32 mma pairing on the single projection (2026-10-17) - promoted

The fork's k loop pairs two k-tiles per step: both B fragments decode up front, then two chained mma
per row tile - half the loop iterations, doubled decode ILP.  Tried as a scratch experiment against
the restored pre-pairing kernel (same signature, separate variant behind a temporary env; the gate/up
launch pairing of A/B #7 was NOT included, per isolation).  Admissible shapes (ki % 8 == 0) make
rows_total a multiple of 8, so kct is always even; the odd-tail step is a guard, not a live path.

Correctness: the whole exl3_moe_multi_parity suite re-run with the variant selected - the K3..K6
boundary-row oracle section passes 36/36 (rel 6.1e-4..7.6e-4); with the variant off, unchanged.

Resource evidence (llvm-readobj on the code object's amdhsa notes, not inferred): the paired kernel
uses vgpr 87-110 / sgpr 34 against the single-step form's 61-90 / 27, with ZERO vgpr spills in both;
the LDS staging array is identical by construction (same `sh_a` declaration).  The isolated
microbench therefore favors the single-step form everywhere (rows 16: 0.1519 vs 0.1465 ms; rows 64:
0.1973 vs 0.1878; rows 256: wash; rows 455: 0.2832 vs 0.2697) - the register pressure costs more
than the saved loop overhead when the kernel has the GPU to itself.  The engine is the decider and
disagrees:

- PP8k, WMMA+WAVESYNC both arms, 6 alternating runs per arm
  (`bench/results/2026-10-17-hip-exl3-m6/`): single-step **403.0 [401.4..404.4]** vs K32
  **406.6 [405.6..408.0]** tok/s - **+0.9%**, and every K32 run is faster than every base run
  (12/12 ordered; within-arm spreads ~0.7%).  In the engine the group kernels interleave with
  gathers, postscales, scatter and the staging DMA on two streams, where the occupancy headroom is
  shared and the shortened k loop pays; the isolated bench gives the kernel the whole GPU, where the
  extra registers cost.  The mechanism reading is consistent with the numbers but the sign of the
  application result is the evidence.
- PP512 control (2+2, ctx 8192 config): 8914/8932 vs 8921/8928 ms - a wash, no small-prompt
  regression.  Warm TG control: unchanged (decode windows take the host-IO path).  Outputs
  byte-identical between arms.

Verdict: **promoted** - the pairing is now the body of `routed_wmma_kernel` (no env knob, no
separate variant left), with the reconstruct+hipBLAS path remaining the overall fallback.  The
dormant `projections` template dimension and `WmmaProjIo` argument from A/B #7's cleanup were
removed; the kernel is back to the simple three-pointer signature.  Post-promotion controls: parity
49 PASS lines, exl3/ple ctest subset 24/24, the wave checker PASS, PP8k 407.3 tok/s with outputs
identical to the trial arm.

### A/B #9 detail: prefill wave staging/compute pipelining (2026-10-17) - kept, opt-in

M1 ranked expert miss servicing as the dominant prefill cost and A/B #5 left the wave-end barrier
as the remaining serialization: a wave's whole upload ran before its FFN, and the next wave's
uploads waited for the previous wave's compute.  B pipelines the waves the way the native prefill
staging ring streams experts (prefill.cpp's `copied`/`used` slot events): while wave W's FFN runs
on the compute stream, wave W+1's `pread` + H2D run on the existing staging stream, into cache (or
scratch) slots wave W is **done reading**.  A slot is released by a `used` event recorded on the
compute stream right after the group's last tile is queued (early release, not wave-end), and the
staging stream waits that event before overwriting the slot - so a later wave may reuse a slot
while earlier waves' other tiles still run, with no cross-wave pinning.  The per-wave host sources
(the pageable route/matrix arrays the FFN entry points upload from) live in a two-bank ring,
reused two waves later behind a `wave_done` event; every call enters and leaves drained (one
barrier per call replaces one per wave).  Implies the wave barrier; the default per-tile path
creates no events and is byte-identical.  One knob: `STRATA_EXL3_PIPE=1`.

Correctness on the synthetic one-layer model (now 80 experts with DISTINCT weights and scales per
(expert, projection), written by `tools/test_exl3_loader.py`): `strata-exl3-wave` compares default,
wave, pipe and pipe+wave arms in both kernel regimes (reconstruct fallback and WMMA) across the
>=16-row group path, the indexed path, one group over two tiles (5000 routes), a 16-row group
before a 600-row group in one wave, a partial final wave (10 groups over a 4-slot cache: waves
4+4+2), warm repeats, a scratch (cache 0) stream whose second wave reuses the first wave's scratch
slots while their readers may still be in flight (80 groups over the 64-slot scratch wave size),
and the prefill -> run() -> decode-window phase sequence - 64 byte-identity comparisons, all
byte-identical, plus finite/non-constant checks.  The single-token `run()` leg is compared at a
scale-normalized tolerance (max |diff| 3.8e-6, norm 3.5e-4): that path's no-input_rows GEMV
accumulates K partitions with atomics, so two runs of the SAME call already differ by that much
(the default-vs-default control shows the identical figure - the pipeline adds no divergence).
`run()` itself is untouched by the change.  Tests: exl3/ple ctest subset 24/24 (includes the
checker and `exl3_moe_multi_parity` 26/26 on both XTXs).

Overlap is event-measured on the real model, not inferred from stream counts:
`STRATA_EXL3_TIME=1 STRATA_EXL3_PIPE=1` records four timestamp events per wave (staging begin/end
on the staging stream, FFN begin/end on the compute stream) and reports their pairing at the
call-end barrier.  One 8192-token chunk (diag-pp8k-pipe, fresh engine, warm page cache): **189
waves, staging 6013.8 ms, FFN 5282.9 ms, staging-overlapping-prior-wave-FFN 3658.2 ms** - 61% of
the staging time runs concurrent with the previous wave's FFN.  A rocprofv3 timeline
(`--kernel-trace --memory-copy-trace`) was captured as an independent check: it confirms the
copies and kernels run on separate queues and identifies the agent (Gpu_Id 34772 =
0000:03:00.0), but its host-side tracing overhead stretches a 6.9 s prefill to 391 s, leaving the
GPU idle 98% and collapsing the measured copy/kernel overlap to 7.6 ms - a traced timeline is NOT
evidence about the untraced regime here; the event counters above are the quantitative record.

Numbers (3.05 bpw model, WMMA+WAVESYNC fixed in BOTH arms, GPU0 = 0000:03:00.0 - confirmed
CPU-attached x16 at 28.2 GB/s pinned H2D; topology/identity evidence in
`bench/results/2026-10-17-hip-exl3-3bpw-prefillpipe/GPU-IDENTITY.txt` - fresh engine per run,
warm page cache, alternating, unique run dirs with engine sha + env per report):

- PP8k ctx65536 (126 slots/layer auto): base **423.8 [422.6..426.7]** vs pipe
  **511.5 [507.8..512.1]** tok/s - **+20.7%**, 3 runs/arm non-overlapping; all six token streams
  byte-identical.  Decode after the prompt: 17.8 vs 17.7 tok/s.
- PP512 (warm-session turn 0, cold expert cache): 8036 [7998..8077] vs [6902..6915] ms -
  **+16.2%**.
- Warm TG control (turn 1, 256 tokens, essay): 23.0 vs 23.0 tok/s - unchanged, as designed:
  decode windows take the host-IO path and the change is device-IO only.
- Low cache (32 slots, 128-token needle): turn-0 prefill 5107/5147 vs 3948/3964 ms - **+29.6%**
  (more, smaller waves create more overlap pairs); warm turn-1 10.2 vs 10.2 tok/s unchanged;
  needle keys 3/3 in every run of both arms.
- VRAM: identical in both arms (6048 slots total, 1884 MiB free at ctx65536; 6000/1822 on the x4
  card).  The per-slot release events cost under 1 MiB and the auto sizing gives their bytes back
  from the slot count (a re-probe after the events are created), so the arm runs under the same
  total VRAM budget; at this granularity the slot count did not change.
- Anatomy (STRATA_EXL3_TIME, one 8k chunk, same bytes both arms - 38545.42 MiB): pool 11127.6 ->
  7782.0 ms; device syncs 189 (4833.7 ms host wait) -> 48 (1147.7 ms); cache 0/21702 with 15654
  evictions in both arms - identical slot assignment, only the scheduling differs.
- PCIe link effect, the same A/B repeated on the chipset x4 card (0000:09:00.0, 6.5 GB/s pinned
  H2D, `HIP_VISIBLE_DEVICES=1` recorded per report): base 319.1 [319.0..319.2] vs pipe
  380.8 [380.3..381.4] tok/s - the pipeline gain persists (+19.3%) and the x4 base is ~24% below
  the x16 base: the link is a real prefill constraint that the overlap hides in part.  The 3bpw
  model's NVMe is CPU-attached (0000:04:00.0), so model reads do not share the chipset uplink;
  residual shared traffic there is the NIC/WiFi/SATA/USB chain and the display.

Verdict: **kept, opt-in** (`STRATA_EXL3_PIPE=1`, one knob; it implies the wave barrier).  The
mechanism is plain CUDA-runtime stream/event scheduling like A/B #5 - nothing AMD-specific - but
it is measured on gfx1100 only; no NVIDIA run validated it (no nvcc on this machine).  Cross-call
pipelining is not done (the arm barriers at each call boundary; the caller's route readback syncs
the stream before the next call anyway). Review closure (`ba9b8c7f`) gates wave completion events
and both host banks under PIPE, restores baseline call-local host arrays, checks slot-release
records and the final staging drain, and prevents a failed slot dependency wait from enqueueing
its upload. Errors take the existing stream-drain/residency-invalidation path; runtime failures
were inspected, not fault-injected. The expanded distinct-expert checker also compares PIPE+RAM
hits/eviction against baseline in both kernel regimes. RAM pins reset only after the previous
wave's workers finish consuming RAM sources into event-guarded pinned staging; those DMA buffers,
not RAM slots, remain live. Real-model PIPE+RAM remains untested. HIP Release and exl3/ple tests
pass 24/24 after closure. PIPE with explicit 24/24 layer split on both XTXs passes a 512-token
needle (3/3 keys, 37-token answer). No decode-side overlap is included.

### Configuration measurements after A/B #9 (no new kernels or defaults)

All runs use this checkpoint's own PLE and tokenizer pack, WMMA+WAVESYNC+PIPE fixed, int8 KV,
context 65536, prefill 8192, no MTP. The prompt's bulk is 8185 rows followed by checkpoint/tail
processing, not a warm seven-token prefix or internal 512-row batching. Physical GPU is
0000:03:00.0 (CPU x16, HIP_VISIBLE_DEVICES=0); other visibility aliases are unset in the review
runs. Loaded hipBLASLt resolves to `/opt/rocm/core-7.14/lib/libhipblaslt.so.1.4`, version 100401
(1.4.1). Exact configs, reports, engine hashes, effective visibility and outputs live in
`bench/results/2026-10-17-hip-exl3-3bpw-prefillpipe/`, latest under `review-closure/`.

These are DIFFERENT configurations, not a single tuning-table result:

| Configuration added to PIPE baseline | PP8k evidence | Result |
|---|---|---|
| User's hipBLASLt table only | one phase diagnostic | 12.21 s, roughly 671 tok/s |
| `STRATA_PF_GEMM=1 STRATA_WMMA_GEMM=1` | 3 alternating pairs | 677.8 [677.2..683.7] vs 504.3 [503.7..504.5] tok/s |
| Above plus `STRATA_HIP_WMMA=1`, explicit **126 slots/layer both arms** | 3 alternating pairs after safety closure | **751.3 [751.0..752.3] vs 510.4 [508.1..510.6] tok/s (+47.2%)** |
| User table plus `STRATA_HIP_WMMA=1`, same126 slots | one phase diagnostic | 741.0 tok/s; simpler alternative, not established equal by repeated A/B |

User table provenance: this checkout's `tools/hip/gfx1100-hipblaslt-100401.txt`,
read-only, 26 rows, bf16 solution IDs603/604 and f16 ID1277. At actual T8185 it accepts **15**
resolved shapes and falls back on **one** uncalibrated shape (bf16 T8185, N10240, K2560,
ldy10240). Table SHA256 is `5a13da8777af122460aa85fdb5eb520c63c359ce8d799734d2d19e98c20d3d48`.
An earlier table from the exl3-perf
worktree loaded the same header/version but accepted0/fell back16 (IDs2106 unavailable,
1035/1036 rejected). This is a solution-ID/library-build mismatch, NOT missing T coverage:
`closest()` selects the nearest T bucket (8185 selects8192), then support is checked against the
actual shape. The user's table is not edited. Adding that table above PF+WMMA changes nothing
because those paths take the calls first.

`STRATA_PF_GEMM` handles f16 before the ordinary WMMA path; `STRATA_WMMA_GEMM` also handles bf16.
`STRATA_HIP_WMMA` changes prompt-attention arithmetic and is explicitly **bit-changing**. All32
saved diagnostic token IDs were actually compared and matched; all six latest A/B32-token streams
also match. The tuned configuration passes an8K needle (3/3 keys) and a coherent256-token essay.
These are sanity checks, NOT broad numerical-quality or long-generation equivalence evidence.
`STRATA_HC_Q8` had no measured prefill effect here; `STRATA_SELECT_WMMA` saves only about17ms;
`STRATA_DENSE_MMQ` did not engage. None is added to the recommended configuration.

Same-cache phase measurements (single diagnostics, ms; GPU-event intervals include host gaps):

| Phase | PIPE baseline | PF+WMMA+attention | User table+attention |
|---|---:|---:|---:|
| Prompt GPU timeline |15792|10454|10632|
| DONE prefill wall |16201|10877|11055|
| HC read |1475|443|319|
| GDN input projections |1913|261|320|
| QSA projections |816|164|184|
| QSA attention |1329|258|258|
| GDN recurrence |327|303|306|
| GDN output projection |756|131|149|
| Router+shared (includes expert pool) |9028|8776|8871|
| Expert pool host duration |7859|7811|7945|

The tuned expert pool is now about75% of the prompt timeline (72% of DONE prefill wall), not a
claim that all remaining time is DMA or GEMM. In its189 waves, staging spans6069ms, FFN spans5277ms,
and3666ms of the staging interval overlaps the preceding FFN interval. These overlapping spans
must NOT be summed as independent savings. The staging interval includes host starvation and
slot-release waits; historical pread throughput does not establish today's bottleneck. The
measured remaining seam motivates evaluating **wave-batched expert computation**, not another
per-expert gate/up fusion. That is a proposal only; no implementation is started.

Both latest arms use6048 total slots (126/layer) and10856MiB cache. Tuned free VRAM is2264MiB vs
baseline1884MiB, so the tuned arm uses LESS total VRAM; it does not gain cache capacity. Earlier
auto-cache full-stack trials chose130 slots and were explicitly followed by these fixed126-slot
runs. Reproducible opt-in configuration is `bench/exl3-3bpw-gfx1100-tuned.json`: same model/PLE,
explicit126 slots, PF+WMMA+attention switches, no defaults changed. To reproduce on this machine:

```sh
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/llvm/lib:/opt/rocm/lib \
  .venv/bin/python tools/exl3/bench.py --engine build-hip-exl3/strata \
  --config bench/exl3-3bpw-gfx1100-tuned.json --prompt-tokens 8192 --max-new 256 --output /tmp/exl3-unique-run
```

Use `--prompt-tokens 8192 --max-new 256` (separate CLI arguments); choose a fresh output directory.
The configuration's paths and126-slot budget are machine/model-specific. Review is required
before recommending bit-changing attention for broader workloads.

### A/B #10: bounded multi-expert compute launches (2026-10-17, provisional opt-in)

`STRATA_EXL3_EXPERT_BATCH=1`, with EXL3 WMMA and wave scheduling enabled, batches up to eight
adjacent compatible expert groups, at most4096 routes total. Existing expert-major routing,
cache assignment and transport are unchanged. Gather, WMMA and postscale use an expert grid
z dimension, followed by one subgroup SiLU; the down postscale scatters directly to original
route destinations. Each expert/projection keeps its own suh/svh. Mixed bitrates/codebooks,
unsupported geometry, small groups, row-tile boundary transitions and single-group tails fall
back without partially enqueueing a batch. This adapts nasone32's read-only v4 expert-dimension
technique to Strata's existing workspace and route plan, not its scheduler/graphs.

Batches release slots after each bounded subgroup, not after the full wave. Immutable host
metadata slices live in the existing two-bank wave ring. All device scratch and host descriptor
capacity is reserved before queued tiles; the subgroup API rejects growth. A bound of eight
appended descriptors per starting group includes rejected attempts, preventing host-vector
reallocation while earlier uploads use it. The default unset/0 path allocates no subgroup scratch;
the initial explicit0 matched-scratch experiment was removed from production.

3bpw, physical0000:03:00.0 x16, cache126/layer, densePF_GEMM+WMMA_GEMM+HIP_WMMA and EXL3
PIPE+WMMA+WAVESYNC fixed in both arms; fresh engines, warm OS page cache, cold expert cache.
Artifacts: `bench/results/2026-10-17-hip-exl3-3bpw-expertbatch/`, exact configs/engine hashes,
BDF/visibility records and outputs. No inference from historical NVIDIA or fork throughput.

- Initial matched-scratch PP8k,3 alternating: **766.1 [764.5..769.4] ->790.5 [789.9..795.4]
  tok/s (+3.17%)**. Prompt-only median10692.5 ->10363.6ms. Wall including32decode tokens
  12.489 ->12.187s; those are distinct intervals, not independent savings.
- Production-unset126 controls (no artificial preallocation): **764.7 [763.9..769.0]** tok/s,
  consistent with the matched baseline. These controls are separate sessions, not a paired exact
  27tok/s claim. Candidate same-cache remains a modest, provisional workload-specific gain.
- PP5123alternating:84.0 ->85.3tok/s (+1.48%); fresh-engine/cold-expert essay, NOT warm prompt-KV
  hit rates. PP16ksingle pair847.0 ->902.0tok/s (+6.5%, not repeated evidence).
- WarmTG256:20.62 ->20.71tok/s (single sessions, unchanged within noise). Needle8k3/3; coherent
  256-token essay. All sixPP8k andPP51232-token streams match byte-for-byte.
- Histogram on both current arms:21710expert groups (not tile calls):6621<16,1277 at16..23,
  3773 at24..63,5859 at64..255,4180>=256. Prior21702 was under different dense/attention
  arithmetic; no forced identical routing across different arithmetic paths is claimed.
- 12036groups form3375subgroups;9674remain standalone (6621small and3053eligible tails).
  Known executed-chain dispatch accounting:10125gather+10125WMMA+10125postscale+3375SiLU=33750.
  This is host counting of fixed dispatched chains, NOT profiler/API tracing. Stop attempts:
  small11011, route cap64, row-tile boundary1598, eight-group cap252; attempts overlap across
  retries and are NOT a disjoint group taxonomy. Real-model metadata rejects0.
- Geometry: WMMA grid `(ceil(max_rows/(16*row_tiles)), ceil(outputs/256), groups)` with x capped32,
  z<=8,512threads. Gather/postscale grid over bands/rows/experts. No measured occupancy claim.
- Diagnostic pool7561 ->7387ms, FFN5281 ->4285ms, staging5836 ->6049ms,
  staging-over-prior-FFN3682 ->3078ms. About1s less FFN span loses about0.6s overlap because
  subgroup releases are later than per-expert releases. These intervals overlap; wall is the
  decision metric. This is why fewer launches do not imply a proportional application gain.

Memory accounting: INFO is a fresh startup hipMemGetInfo sample before lazy first-request buffers,
not a peak/settled measure. Live engine snapshots after DONE+1s on the same BDF show used bytes
23464620032 baseline vs23477960704 candidate (extra13340672bytes, **12.72MiB**). Device total
25753026560 leaves **2182.39 vs2169.67MiB free**, BOTH above the configured2048MiB reserve.
Requested workspace capacities after dispatch:230540288 vs230542336bytes (2048extra descriptor
bytes in the measured16-capacity prototype; final capacity is8,1024bytes). Existing4096base arrays
and largest-group scratch dominate subgroup needs. No exact runtime-allocation-granularity cause
is asserted for the observed plateau. Peak allocation/headroom was not sampled.

An overconservative112-slot candidate vs126baseline lowered settled use by~1.11GiB but gave only
+1.43%PP and **-7.63%warmTG**. That is cache-capacity loss, NOT intrinsic batching regression;
it is not the acceptance comparison. The same-cache candidate fits the original reserve and its
small extra settled cost is disclosed rather than sacrificing14cache slots/layer.

Validation: HIP Release, EXL3/PLE24/24. CPU oracle uses distinct signed weights/scales for each
expert and projection, K3..K6, unequal16/23 and24/31groups, nonidentity input/scatter; expected
fallbacks include mixedK/cb and small groups. On supportedgfx11 a rejected valid arm fails rather
than silently skipping. Stream checker covers PIPEon/off, WMMAoff fallback, partial waves,
cache0/cross-wave eviction, repeated calls and groups>4096. Final reviewed512needle3/3.
CUDA/SYCL builds unavailable; no portable performance claim, fault-injection or peakVRAM test.

Verdict: retain as **provisional opt-in**, defaults unchanged; no subgroup sweep, transport packing
or new routing. Next proposal, not implemented: measure staging fragmentation/read-enqueue gaps
before deciding whether to pack transport. Review this bounded experiment first.

### Arena-only preparation for future phase borrowing (2026-10-17)

`STRATA_EXL3_PREFILL_BORROW=1` currently selects **experimental arena backing only**. No lending,
refill, cache resizing, phase coordinator or callback allocation changes are enabled. Default/unset
and0 keep the original nine device allocations PER LAYER (three planes per projection, each plane
spans all slots); this was not nine allocations per individual expert slot.

One stable allocation per device/layer range is rank-major: `[rank][layer][nine planes]`. Every
plane/rank starts on256-byte alignment. Checked geometry, cumulative sizes and capacity reject
zero planes, overflow and slots outside0..512. Valid ki/nj multiples of8 make every scale plane
`32*ki`/`32*nj` and trellis plane`32*bits*ki*nj` divisible by256; an explicit invariant checks that
rank bytes equal the raw auto-cache price. Mixed K/hidden dimensions do not add hidden padding.
The arena is owned exactly once by the stream; layer allocation-owner lists stay empty, while
slot descriptors hold interior pointers. Destruction uses the owning device, drains arena streams
and frees once. A contiguous allocation failure reports the runtime error and falls back to the
original backing without claiming borrowing support. Fallback allocation failure and post-arena
allocation unwinding were inspected, not fault-injected. The missing-layer constructor test fails
before arena allocation and must not be described as a post-allocation cleanup test.

HIP Release and EXL3/PLE24/24 pass. Distinct-expert tests compare arena/original across PIPE+batch,
cache0, eviction, partial waves, repeated calls and groups>4096; mixed layerK4/hidden640 and
K3/hidden128 plus nonzero layer range; pure alignment/nonoverlap/end/rank isolation and invalid
bounds; destructor restores another current device. Phase-sequence parity explicitly covers
run_device ->run() ->run_multi ->second-turn prefill. run() keeps its existing atomic-rounding
tolerance; deterministic paths compare byte-for-byte. CUDA/SYCL compilation remains unavailable.

Runtime sizing, real3bpw, tuned8K, context65536, CPU-x16 device0000:03:00.0, own PLE/tokenizer:

| Quantity | Bytes | MiB |
|---|---:|---:|
| Actual arena,126 slots/layer |11263795200|10741.992|
| One rank,48 expert slots |89395200|85.254|
| Prefill borrowed-region counter |3472925696|3312.040|
| Prefill owned-allocation counter |3514826752|3352.000|
| Actual Alloc requested sequence, GEMM scratch included |3380626432|3224.017|
| Runtime free-memory delta during carve only |3307208704|3154.000|
| Native expert staging ring96, unused by EXL3 |132710400|126.563|

Counters, requested bytes and runtime allocation deltas are DIFFERENT quantities: counters include
slack/conservative accounting, owned count models page rounding; carve delta excludes GEMM scratch
allocated immediately before it and is not a peak. No allocation granularity claim follows from
the delta. The shared MoE region still reserves GGUF-only Xs400MiB, GU400MiB and Hh100MiB at8K;
these are not consumed by EXL3 but overlap the same region as attention, so their sum is NOT
independently reclaimable memory. Native ring126.563MiB and two dequantized expert pairs18.75MiB
are separate unused EXL3 allocations. Nothing is removed/refactored in this stage.

The owned reserve3352MiB could, by bytes, fund39 extra per-layer ranks (126 ->165). A39-rank tail
is3486412800bytes and covers the counted borrow with~12.86MiB slack. This is a capacity estimate,
NOT a measured speedup or permission to reduce reserve now. Verifier/decode workspace remains
permanently preallocated (~220MiB requested in prior diagnostics).

One paired PP8K control: original782.4 vsarena769.2tok/s (-1.7%, single pair; no speed claim),
32 output IDs identical. WarmTG1286159.9 vs6058.4ms (single sessions, not an established win).
Arena512needle finds3/3 keys. Startup free VRAM2172 vs2182MiB at identical6048 cache slots.
Raw configs, engine hashes/BDF identity, outputs and sizing logs:
`bench/results/2026-10-17-hip-exl3-3bpw-arena/`.

Next-stage design remains unimplemented and needs review: extend the SAME PfPart/lend/refill
coordinator with an EXL3 backing adapter. Eligibility for batch/head-alias/no-prefill/owned fallback
must be decided BEFORE reducing the cache reserve. Initial Prefill::init carving itself acquires
the tail (inactive even at startup), not merely the first request's lend call. Reduced active ranks
must not inherit native global128-slot floor semantics.

Duplicate-safe restore options, not a locked policy: (a) exclude saved-tail IDs from prefix admission,
which needs mixed scratch/cache per-group ownership/events and may hurt PP; (b) keep valid hot-prefix
duplicates, skip their saved records, compact remaining restored experts sequentially into the tail,
and publish map/ticks/filled only after all copies complete. Option(b) can preserve a dense occupied
prefix without a free-list. Both need loan epochs, cancellation, duplicate IDs, partial-refill failure
and drained-stream transaction rollback tests; never blindly overwrite cache.at[expert] or restore
an entire old map. No host residency is published before copy completion. No integration is started.

### Stream loan/refill primitives, implemented and validated (2026-10-17; review fixes same day)

Option (b) above is IMPLEMENTED at the stream level, scoped to the experimental arena
(`STRATA_EXL3_PREFILL_BORROW=1` with a cache; original backing and cache 0 report no loan and are
untouched).  No engine call site exists yet: `generate.cpp` does not lend; these are library
primitives plus tests.  Public API on `Exl3ExpertStream` (`include/strata/kernels/exl3_experts.hpp`,
full contract there):

- Queries: `loan_available()`, `loan_rank_bytes()` (exact, 256-aligned), `loan_max_ranks()` (keeps
  one active rank), `loan_active()`, `loan_active_ranks()`, `loan_epoch()`.
- `lend_tail(ranks) -> Loan{address, bytes, ranks, epoch}`: one contiguous tail range ending at the
  arena end.  While open, every run-path fill/evict/LRU scan is clamped to the active prefix; a
  call needing more live experts falls back to the per-call scratch.  Growth re-lends a superset
  (address moves down by `rank_bytes` per rank); same-size repeat idempotent; shrink/zero/negative/
  no-active-rank-left rejected with NOTHING changed.  Exception safety: the snapshot and every
  allocation complete before the first map mutation; the commit (erase/append/clamp) cannot throw.
- DENSE INVARIANT, every owned layer, between calls: the mapped slots are EXACTLY [0, filled) (a
  bijection), filled <= active.  The stock dense-fill/evict paths preserve it; lend clamps
  `filled` per layer; restore extends each layer's own prefix.  Exposed as
  `loan_dense_invariant()` and asserted by the tests after every phase - including on the
  original-backing oracle, so the clamped scans keep the stock semantics.
- `restore_tail() -> bool`, a transaction.  Per owned layer (deterministic order: layer order, then
  the record's original slot) each saved expert NOT already resident is re-read from the model
  (existing pread/pinned staging) and placed densely, extending THAT layer's own occupied prefix
  into ITS OWN tail - per-layer slot spaces, never a shared counter (the first review found a
  global counter that would walk a 48-layer restore out of bounds).  An expert saved more than once
  (a growth can re-lend a re-admitted expert) is placed exactly once; a resident duplicate keeps
  its hot prefix copy - overwritten tail bytes are never trusted.  ADMISSION IS BOUNDED,
  deterministically: one loan's records all come from the DISJOINT rank increments of its lends
  (their union is exactly [cache - ranks_final, cache)), so their count - and the unique experts
  among them - never exceeds the lent ranks; with filled <= active = cache - ranks the survivors
  always fit, and a valid growth can never overflow the restore.  Should that bound ever be
  violated anyway, the surplus records in the deterministic order are DROPPED (not resident;
  ordinary misses re-admit them) rather than failing the restore.  Planning precomputes the next
  maps, fill watermarks AND placement lists - every allocation - before staging; publication
  happens only after the staging stream drained, via swaps and in-bounds writes that cannot throw
  or allocate: no allocating operation runs after publication begins.  Failure contract:
  a FAILED PRE-drain rejects with no state change and the loan stays OPEN (no loan on a context
  whose syncs fail, no recovery claim for one); a failure after uploads began drains the partial
  uploads, publishes nothing, and invalidates every layer's device AND host-RAM-cache residency
  exactly like the run error paths, while still returning the capacity.
- Validation seams (narrow, read-only, documented for tests and the future coordinator):
  `loan_debug_saved(layer, out)` - the active loan's saved ids per layer, unique ascending;
  `loan_debug_restored(layer, out)` - the LAST restore's placements per layer (a lend
  clears it; a restore that ran and failed clears it; a REJECTED call - no loan, or a
  failed pre-drain - leaves the previous value untouched).

**Correctness evidence (`exl3_wave_check` loan battery; PIPE+BATCH+WMMA on, and all opt-ins off;
GPU0 gfx1100):** the lending stream records every call it makes - including calls whose ids were
CAPTURED from the seams - and an original-backing oracle replays the identical list; every output
matches (deterministic paths byte-identical, `run()` within its established atomic tolerance), and
the replay stream keeps the dense invariant after every call.  Phases: rejections that change
nothing; a cold lend (empty saved tail, still an active loan) with run() inside it; a
partial-prefix lend; the PURE REFILL (no run during the loan - `restored == saved` asserted, then a
hit-only call reads exactly the refilled slots BEFORE anything can evict or repair them); a
run_multi window inside a loan (mixed duplicates/survivors); the ALL-DUPLICATES case made explicit
(route exactly the saved ids during the loan so every one is resident at restore; `restored` EMPTY
asserted; then a hit-only read of the same ids proves the poisoned tail was never published);
growth mid-loan with churn (duplicate records across growth deduplicated, survivors <= ranks,
address arithmetic checked), the maximal one-active-rank loan, run()'s scratch fallback, a second
full turn, cache-0 and original backing reporting no loan, and broad churn only AFTER the direct
hit reads.  Overwrite evidence throughout: the borrowed range takes a 0x5A memset that is read
back (the borrower owns the exact bytes), then 0xFF - an fp16 NaN pattern - so any read of a lent
or unrestored byte surfaces as a non-finite output.

**The multilayer battery (both fixture layers owned - K4 2560->640 and K3 ->128 - each with a full
4-slot cache):** lend captures BOTH layers' tails at once, poison, churn on both layers during the
loan, growth to the maximal loan, restore with per-layer `restored <= ranks` bounds, per-layer
hit-only reads immediately after restore, broad churn after, and a second full turn - all
capture-replayed against an original-backing oracle.  This is the regression for cross-layer
placement/bounds (the first review's blocker) - a shared placement counter walks layer 1's slots
out of range here.

**The repeated-growth battery (six-slot cache, two growth steps, re-admission churn between
them):** one saved expert is deliberately never routed again after its lend, so it is absent from
the final prefix; the restore must still succeed (valid growths cannot overflow - the bound
above) and that old-saved expert must come back as a placed survivor (`loan_debug_restored`
asserted, then a hit-only read before any churn).  The saved-union bound is asserted at every
growth (unique saved <= lent ranks), with re-admission churn between the lends exercising
duplicate records across growths.

**A real pre-existing race the battery found and fixed:** `run()`'s cache-0/scratch arm enqueued
`exl3_moe_rows` on the compute stream BEFORE the shared `stage_done` wait that orders the FFN after
the staging stream's copies (the cache split path was correct; only the scratch arm was
mis-ordered).  Seen as one NaN/wrong-expert row in ~2 of 5 checker runs before the fix, 0 after;
the fix moved the wait ahead of the FFN enqueue and consumed the jobs so the shared tail cannot
re-stage them.  Isolated in its own commit.

**Validated:** HIP Release build; EXL3/PLE ctest subset 24/24 with `HIP_VISIBLE_DEVICES=0,1`
(includes the extended wave checker via `exl3_loader_validation`), checker run 3x consecutively
green; real 3bpw smoke on this tree labeled ENGINE-UNCHANGED (arena-only backing, no lending in
the engine): needle 3/3 keys passed (prefill 90.11 tok/s, decode 20.35 tok/s - single runs,
consistent with the accepted arena artifact's 88.45/19.84, no speed claim; final engine sha256
907766ab..., recorded in the artifact) and a two-turn warm session (128 tokens/turn); artifacts:
`bench/results/2026-10-17-hip-exl3-3bpw-loan-primitives/`
(original checkout).  `ple_parity` (test #66) fails there for a missing GGUF shard, unrelated.

**Inspected, not fault-injected:** the two restore failure modes (failed pre-drain: loan stays
open; failed staging/drain: invalidate-everything) are code-reviewed, not triggered - no production
fault injection exists solely for tests; concurrent lend/restore from multiple threads is
serialized by the run mutex but not stress-tested; a borrower writing the range after restore is a
contract violation the stream cannot detect.  CUDA (no nvcc) and SYCL (no icpx) builds not
performed; the new code is plain host C++ on the same headers the file already uses.

**INTEGRATED (2026-10-17, correctness-first): the engine's serve lifecycle now borrows.** The
smallest coordinator step below is implemented, additive and narrow:

- ELIGIBILITY, decided once in generate.cpp BEFORE the prompt path is laid out (so the loan opens
  before Prefill::init carves anything): EXL3 model, an operator-NAMED `--prefill N` (a new
  `prefill_explicit` flag; `--prefill auto` collapses as before and stays owned), `--prefill`
  positive, no `--no-prefill-borrow`, no `--prefill-head-alias`, single GPU (no split), no
  `--batch`, and `loan_available()`.  The loan is `ceil(Prefill::bytes_needed(g, ss, chunk,
  src=false) / loan_rank_bytes())` ranks, bounded by `loan_max_ranks()` (one active rank stays;
  the native 128-slot floor is NOT applied).  Any refusal logs why and keeps today's owned
  buffers, byte for byte.  Measured on the 3bpw rig: 39 of 126 ranks (3.25 GiB) for an 8192-token
  chunk; 12 ranks (1.00 GiB) for 2048.
- THE RESERVE AND THE CACHE ARE UNCHANGED (126 slots/layer): the cache is still sized as if the
  owned buffers were allocated.  The borrowed arm simply never allocates them - those bytes stay
  free - so this stage claims NO capacity or speed win, only lifecycle correctness.  Sizing the
  cache for the loan instead is the explicitly deferred next stage.
- LIFECYCLE: the SAME PfPart serve machinery, one participant wide.  The loan is FIXED at the
  configured chunk (documented choice: no per-request sizing, no shrink path, so `lend_tail`
  reopens the identical range and `Prefill::relayout` re-carves the per-run metadata; a shorter
  request uses less of it, a longer one reads in chunks of it).  `refill` - before every window
  segment, before decode, and on the final/cancelled exit - drains the borrower first through a
  new narrow `Prefill::sync_streams` seam (relayout's own compute/copy/KV waits; a cancelled read
  can leave copy/KV work that run()'s end-of-call syncs never reached) and then `restore_tail()`s.
  An active loan with an EMPTY saved tail (the startup cold-cache case) is handled by
  `loan_active()`, never the saved list.  A failed restore closes the loan with every layer's
  residency invalidated and the request loop exits (ERR) - decode never continues on it.  A
  failed relend/lend fails the request the same way.  If `init` steps the chunk down, the loan
  keeps its ranks and `relayout` follows the smaller chunk.  The verifier callback workspace and
  doorbell are permanent separate allocations, untouched.  GGUF (any flag state) and EXL3 with
  the flag off execute the identical code path as before (the lend/refill branches key on the
  loan participant existing).
- NOT covered by the initial scope (explicit owned fallbacks, checked by their log lines in the
  regression): `--batch`/multi-slot, layer split, `--prefill-head-alias`, `--prefill auto`, a
  cache too small to lend (e.g. `--expert-cache 1`).  Untested: the nonfinite/yield/checkpoint
  batch-slot conditions (unreachable single-session; the cancel path IS covered via STOP), a
  restore that fails at runtime (fault-injection only), CUDA/SYCL builds.

**Validation of the integration** (all on the 3bpw rig, GPU0 x16, cache 126, tuned env):
- `tools/exl3/borrow_lifecycle.py` (new; single-session --serve, explicit `--prefill 2048`): the
  borrowed arm and the owned arm run the same five-request script - a 3000-token read across the
  chunk boundary, a follow-up whose ~1350-token suffix re-reads ONLY the suffix (the PP progress
  count proves it) through a re-lent loan, a ~20-token window request (nothing lent, nothing
  restored), a ~300-token single-chunk read, a STOP mid-prefill (cancel; the loan restores on the
  cancelled exit, re-reading the saved experts) and a valid request after it - and every token
  stream is IDENTICAL to the owned arm's.  The eligibility fallbacks are checked by startup log
  lines.  The four traced restores in that script: 0.5 ms (the cold startup tail) and 123-145 ms
  (warm tails decode had filled) - the loan's real per-request refill bill on this rig.  Raw logs
  and the check list: `bench/results/2026-10-17-hip-exl3-3bpw-borrow-integration/lifecycle/`.
- Paired owned-vs-borrowed A/B (single pairs, no speed claim): PP8k + 32 output IDs identical,
  needle 3/3 both arms with identical IDs, a two-turn warm session (128 tokens/turn, prefix
  reused) with identical turn streams; full PP wall, PP tok/s, decode tok/s and the traced
  `EXL3 loan restored N ranks in X ms` lines recorded separately in `summary.json` (the restore
  cost is a REFILL cost, measured on its own - never summed into a throughput claim):
  `bench/results/2026-10-17-hip-exl3-3bpw-borrow-integration/`.

**Integration boundary (next task, NOT started):** the smallest coordinator step is an EXL3 branch
in `generate.cpp`'s existing `PfPart`/`lend`/`refill` lifecycle (borrow selection ~7714, init ~7818,
`refill_issue`/`refill_wait` ~11639, `lend` ~11699, cuts/final refills ~11992/12046): compute the
loan's rank count from `Prefill::bytes_needed / loan_rank_bytes()`, `lend_tail` before the startup
carve writes the tail (Prefill::init already writes into the borrowed region, so the loan must open
BEFORE `reserve_for` reduces the reservation - eligibility decided first), `relayout` into
`Loan.address`, and `restore_tail` in place of `fill_slot_queued` loops.  Native global-128-slot
floor rules and GGUF defaults stay untouched; EXL3 keeps `loan_max_ranks()`' one-active-rank floor.

## The 3.05 bpw checkpoint: identity, validation and a separate M1 baseline (2026-10-17)

**Model** (an EXL3 3.05 bpw export of Qwen3.8-Flash-Next, `Qwen3.8-NEXT-exl3-3bpw`, actual path
has `LMstudio`, not `LM Studio`, at the first level): `Qwen4ExpForConditionalGeneration` /
`qwen4_exp`, quant_method `exl3`, **bits 3.05, codebook mul1**, `head_bits`/`vision_bits` in its
96 MB `quantization_config.json`.  7 safetensors shards, 48.4 GiB of weights, index complete.
Shard headers: **every one of the 48 layers' 512 routed experts is K3 (2560 -> 640, cb=2)**; the
dense linears (shared expert, linear attention, self-attention) are **K5**, the MTP indexer K3.
The loader validates and reconstructs (`strata-exl3-model` on `layers.0.experts.7.gate_proj`:
`trellis[160,40] K=3 cb=2 -> 2560x640`, fnv1a `5a3ad5386333a301`, 304,105 tensors visible).

- **PLE**: the checkpoint carries its own `ngram_embedding.safetensors` (32.6 GB) in the ORIGINAL
  128-shard layout (`shard_N.trellis`, each [2500012, 51]) - the form Strata's loader has always
  accepted; the 4bpw file (39 GB, merged single-trellis, 61 words) is a different bitrate and is
  NOT interchangeable.  No downloads were made.
- **Tokenizer**: merges are byte-identical to the 4bpw's (modulo one trailing newline) and the
  full tokenizer vocab is the same 248,320, but the HF `vocab.json` alone (248,044) is incomplete -
  so a dedicated pack was built from this checkpoint's HF files
  (`tools/exl3/tokenizer_to_pack.py --hf <dir> --out .../packs/exl3-3bpw/tokenizer --check`:
  vocab 248320, merges 247587, round-trip 16/16) instead of assuming the 4bpw pack.
- **MTP**: the checkpoint's native MTP layer weights are IN the shards (`mtp.layers.0.*`) plus
  `mtp_hyper_connection_mixer_patch.safetensors` (13 MB).  Strata cannot load these (M7 deferred);
  the packed family drafter (`Strata-data/mtp/rt`) is used below and works.
- **Fork context (read-only, NOT an A/B)**: the fork's own anchors were measured on this same
  3.05 bpw model in the GGUF container on gfx1100 x2 - PP ~891-1550 t/s and TG ~37 t/s (MTP d3)
  under its engine and configs (`EXL3_HANDOFF.md` "Performance anchors"); different container,
  engine and settings, so these are context, never a comparison.

**Baseline on the final build (`2cc2f6a3`), GPU0 idle, explicit
`LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:...:/opt/rocm/lib` and `HIP_VISIBLE_DEVICES` throughout,
never the gfx1036 iGPU.  Arms: default (no env) vs opt (`STRATA_EXL3_WMMA=1 STRATA_EXL3_WAVESYNC=1`;
K32 is inside the kernel unconditionally).  All artifacts under
`bench/results/2026-10-17-hip-exl3-3bpw-m1/` (engine sha + env per report):**

| configuration | slots/layer | metric | default | opt | note |
|---|---|---|---|---|---|
| PP8k ctx65536, 3 alternating/arm | 126 | tok/s | 339.5 [308..340] | **431.3 [422..434]** | **+27%**; outputs byte-IDENTICAL |
| PP512 warm page cache (essay turn 0) | 163 | ms | 269-278 | 301-313 | ~1.7-1.9k tok/s; see residency note |
| warm TG 256 tok | 163 | tok/s | 25.5 | 24.0 | a real observed outcome of the prefill-policy difference: FP-induced trajectory divergence meeting the slot boundary; see note |
| low cache 32 slots, 128 tok | 32 | ms | 11593 | 11235 | wash |
| layer split 24/24, 512 needle | 347 (both) | PP tok/s | 36.1 | 47.5 | single runs in a marginal-page-cache session: INCONCLUSIVE, not a controlled win |
| MTP packed drafter, needle 37 tok | 163 | tok/s | 62.1 | 62.1 | 27/27 accepted, keys found |

- **Page-cache residency is a first-class variable on this model**: weights+PLE = 81 GB against
  93 GB of RAM.  The first-ever touch of the shards cost 20.8 s for a 512-token prompt; later
  warm-cache runs prefill the same prompt in ~0.27-0.31 s.  The split runs, interleaved with other
  sessions, landed in between (11-14 s) - PP512-scale wall times are only comparable within a
  session, which the alternating pairs above are.
- **Workload-by-workload output comparison (not a blanket claim)**: the PP8k essay streams are
  byte-identical between arms (checked token-by-token, 3/3 pairs).  The WARM-TG essay streams
  DIVERGE - turn-0 prefills round differently, and even same-arm repeats differ (verifier
  checkpoint counts vary run-to-run), so trajectory divergence here is not deterministic
  arm-fingerprint.  The 25.5 vs 24.0 tok/s is nonetheless a real, 3/3-consistent outcome of the
  prefill-policy difference: STRATA_EXL3_TIME shows default at 470-473/480 hits (98%, 7-10
  evictions) vs opt at 456-461/480 (95%, 19-24) - the diverging essays' expert mixes sit
  differently near the 163-slot boundary, and the pool difference (17.5-18.6 vs 22.5-23.4 ms;
  FFN ~12.0 ms/token in both, same indexed kernels) is that miss servicing.  Correctness evidence
  on this checkpoint: needle keys found in every needle run of both arms (MTP arms, split runs
  within their token budget, sanity), PP8k outputs identical, essays coherent (stop=length).
- **Prefill anatomy, first 8192-token chunk (STRATA_EXL3_TIME)**: 21,702 distinct (layer,expert)
  uses - K3 routing spreads wider than 4bpw's 15,444 - all compulsory (cache 0/21702).  Upload
  38.5 GiB / 5.37 s in both arms.  The FFN numbers are NOT comparable kernel measurements: the
  default arm's 9.49 s is a SUM of per-tile kernel intervals, the opt arm's 5.07 s is the wave
  interval (dispatch through last scatter, host gaps included) - different semantics, so no
  "halving" claim and no additive decomposition with the sync counters (device syncs 21,726 /
  9.66 s host wait default vs 189 / 4.89 s opt, which overlap the wave interval).  The controlled
  statement is the wall-clock one: 3/3 alternating PP8k pairs, +27%, non-overlapping.
- VRAM: 1,884 MiB free at ctx65536, identical arms; split leaves 2,886 MiB.
- Revalidation of the promoted selections on K3: the K3 decode path is covered by the standing
  oracle (K3 x boundary rows in `exl3_moe_multi_parity`, 36/36 within the 6.1e-4..7.6e-4 band on
  this build), and the real-model greedy streams above are byte-identical between arms.

**Hotspot ranking on 3bpw**: (1) cold-prefill compulsory miss servicing - upload+pool is ~60% of
an 8k chunk and page-cache state dominates short prompts; (2) the routed FFN kernels - even after
WMMA, 5.1 s of the 8k chunk (the default's 9.5 s is why opt wins by 27%); (3) warm-decode
hit-rate at the slot boundary (95-98% hits, 12 ms FFN + 17-23 ms pool per token); everything else
behind these.

**Feasibility of the remaining plan items on this evidence (assessed, not promised):**
- *M5 (resident decode / host round trips)*: the per-token decode cost is ~12 ms FFN + ~5-11 ms
  pool-over-upload of miss servicing; a graph-side win is bounded by that pool term.  The smallest
  useful spike is overlap, not graphs: stage the next window's predicted misses while the current
  window's resident FFN runs, on the existing staging threads.  Recommended as the next bounded
  task; full in-graph resident compute stays high-complexity/deferred.
- *M4 (staging/cache)*: on 3bpw the page cache already plays the RAM-tier role (81/93 GB); the
  40 GiB managed RAM tier measured NEGATIVE on 4bpw and the freeze measured negative; prompt-heat
  admission ranking remains untested and is the only cheap remaining cache idea.
- *M3 proper (GPU CSR routing)* - PENDING task, not judged complete: the wave barrier already
  removed ~99% of the sync count; the remaining 4.9 s host wait per cold 8k chunk looks
  staging/barrier-bound rather than route-table-bound, so the expected ROI is low - an assessment,
  to be revisited only if a later profile says otherwise.
- *Dense trellis prefill arm* - PENDING task: dense is K5 here and runs through the existing
  native path; unprofiled in Strata (the fork's 26%-of-PP-frame anchor is their engine).
- *M6 (GPU1 helper tier)* - PENDING task: layer split is functional (its single-session PP
  comparison was inconclusive above); a helper expert tier is untested and bounded by GPU1's x4
  PCIe.
- *Native MTP (M7)*: weights are present locally (in-shard MTP layer + mixer patch); Strata lacks
  the loader.  Availability alone does not close this: a later task must inspect the weights'
  compatibility with the drafter interface and attempt the smallest adapter if feasible.  The
  packed drafter already delivers 62 tok/s meanwhile.  Status: PENDING, not deferred-away.

### M5 feasibility probe on 3bpw: the resident-decode seam, measured (2026-10-17)

All numbers: 3bpw, WMMA+WAVESYNC fixed, no MTP, GPU0 idle, explicit library path and
`HIP_VISIBLE_DEVICES` (`bench/results/2026-10-17-hip-exl3-3bpw-m5/`).

**The seam, traced precisely - and it is GGUF-only where it matters.**  The verifier's all-resident
zero-doorbell window graph (`all_resident_` / `exec_nr_`, `include/strata/core/verify.hpp:403-464`)
and its device residency table are REJECTED for EXL3: `verify.cpp:587-589` refuses
`external_experts` together with `d_res`/`cache_base`/`h_res`/`blob` ("external experts cannot use a
GGUF residency table"), so `all_resident_` can never engage on this path.  What EXL3 windows
actually do (`verify.cpp:1639-1642`): each (layer, group) graph node runs `wait_flag_ge(m_flag_,
ring)` - a device spin-wait on a mapped host flag, which DOES work on this HIP stack - and then
`copy_from_mapped` pulls the window's ENTIRE expert-FFN output from host mapped memory
(`m_ymiss_`), the output of the host callback (`drive_exl3_pool_multi` -> `run_multi`'s host-IO
arm).  Resident and miss compute BOTH happen host-side today; there is no GPU hit arm for EXL3.
M5 for EXL3 is therefore "split the callback's output: run the resident share on the GPU and copy
only the miss share" - the fork's mailbox split, on top of the doorbell that already exists.

**Full residency is infeasible on this hardware at 3bpw - measured, not estimated.**  48 layers x
512 K3 experts = ~45.6 GiB of slot data; split 24/24 across the two XTX needs ~22.9 GiB per card
before dense, KV and staging.  The attempt (`allres-split.json`, `--expert-cache 512
--layer-split 24 --split-device 1`, 512 MiB reserve) aborts during the second stream's slot
setup with no error line - an allocation failure.  The zero-doorbell graph therefore cannot
engage on this model+hardware, and "all-resident upper bound" must be read as the mixed-case
bound below, not as a reachable operating point.

**Trajectory stabilization plateaus below full residency.**  Four-GEN sessions repeating the
same prompt (warm_session `--turns 4`, 3 repeats + 1 STRATA_EXL3_TIME diagnostic) stabilize at
**93% hits** (pool 26.2-26.9 ms, upload 10.1-10.8 ms, FFN 12.6 ms per token; turns 3-4:
5342-5543 ms / 128 tokens = 23.5-24.0 tok/s, i.e. no gain over the 2-turn sessions) - the K3
decode working set genuinely exceeds the 163 slots/layer the ctx-8192 config affords.

**The mixed-case bound, from controlled measurements (labeled, no causal decomposition):**
- Engine warm decode, stable 93%-hit state: ~41.7 ms/token wall, of which the STRATA_EXL3_TIME
  medians attribute ~26.5 ms pool (10.4 of it upload) and ~12.6 ms "FFN" - the FFN counter here
  is the host-issued per-window launch sequence, not a pure kernel interval.
- Pure kernel cost of the same resident work, isolated (`exl3_moe_window_bench`, now taking the
  bitrate; K3, 10 experts/token/layer, idle GPU): **0.743 ms per layer-window at 40 routes
  (4-token window), 0.926 ms at 60 routes (6-token)** - i.e. ~7.4-8.9 ms/token of FFN kernels.
- These two numbers are NOT an upper bound of each other's work: the isolated bench differs from
  the engine's resident share in routes-per-window pattern, input data, descriptor residency and
  the launch cadence around it.  The honest statement is directional only: the engine spends
  ~12.6 ms/token issuing the FFN from the host and ~16 ms/token of pool time beyond uploads, while
  kernel-shaped work of this flavor costs single-digit milliseconds - a resident-in-graph path is
  worth evaluating, with no numerical claim of reclaimable time.  Nothing here is "M5 assessed".

**Precise barriers to the smallest native seam (source-cited), for the orchestrator's go/no-go:**
1. The route->slot plan is host state (`Exl3RoutePlan` + the per-layer `at` map,
   `exl3_experts.cu`); the zero-doorbell graph already plans from the device residency table
   (`h_res`/`d_res`, verify.hpp:74) - the table plumbing to reuse exists.
2. `exl3_moe_rows_indexed` consumes HOST-built `Exl3Mat` arrays uploaded per call
   (exl3.cu:636-638); an in-graph resident path needs the slot-table-driven variant (device-side
   route->slot gather feeding the indexed kernel) - the concrete kernel-side work item.
3. Misses need host pread/H2D and cannot suspend a graph - hence the existing doorbell publish;
   the fork's in-graph wait (`hipStreamWaitValue`-class) was rejected on HIP by its own work, and
   nothing in this probe contradicts that.
4. Window graphs are captured per stage; slot base pointers are stable, but the per-window
   hit/miss split varies - the doorbell graph's serve-the-rung-layers dynamism is the mechanism
   to extend.

The smallest useful next step, if approved, is the kernel-side item (2) behind the existing
doorbell graph, evaluated first on the wave checker + a needle A/B; the current-route overlap
idea (stage the CURRENT window's misses while its resident FFN runs - no prediction of future
routes, which are unknown until the routers) is a separate, later M4-side follow-up.

### Decode-profile refresh on the final build (2026-10-17): sustained TG, attribution, M5 inputs

Final build `54ef5882` (engine sha `612d5978...`, recorded per report), GPU0 = 0000:03:00.0
(`HIP_VISIBLE_DEVICES=0`), warm OS page cache (71 GiB buff/cache before the runs; a needle sanity
run absorbed the session's first touch), tuned env fixed in every arm (`STRATA_EXL3_WMMA=1
WAVESYNC=1 PIPE=1 PF_GEMM=1 WMMA_GEMM=1 HIP_WMMA=1`), borrowing OFF except one labeled control.
All artifacts, configs, per-run env, DONE lines, token streams and raw engine logs:
`bench/results/2026-10-17-hip-exl3-3bpw-decode-refresh/` (original checkout).

Sustained warm essay TG (`warm_session.py --essay`, two GEN turns in one engine; turn 1 is the
warm number; same deterministic essay prompt as every prior session):

| arm (slots/layer = engine INFO) | turn 0 / warm turn 1 tok/s | notes |
|---|---|---|
| tuned cfg 126 slots, ctx 65536, `--spec 4` (INFO: spec=6 mtp_max=4 - the suffix drafter extends the window) | 19.63 [19.61..19.66] / **20.11 [20.08..20.12]** | 3 sessions, 512 tokens each turn, spread < 0.3%; stop=length |
| same + `STRATA_EXL3_PREFILL_BORROW=1` | 19.66 / 20.14 | **both turns' token streams identical** to the un-borrowed arm; startup free VRAM 5528 vs 2264 MiB (the 3.25 GiB loan) |
| ctx 8192 / `--expert-cache auto` -> **167** slots, `--spec 2`, tuned env | 22.83 / **25.11 [25.07..25.15]** | the M1-matched shape; M1's 3bpw row was 24.0 on the older build at 163 slots |
| 32 slots, ctx 8192, spec 2 (low cache) | 10.51 / **10.55 [10.53..10.57]** | 128 tokens/turn |
| tuned cfg + packed drafter `--mtp rt --spec 4 --spec-min-p 0.5` | 21.12 / **21.91 [21.89..21.92]** | drafts accepted **139/231 = 60.2%**, 2.19 tokens/round over 117 rounds (~99.8 ms/round); streams identical across the two sessions |
| needle sanity (37-token answer) | 20.5 | 3/3 keys found; 18/29 suffix drafts accepted (predictable text) |

Two structural facts the short-prompt M1 rows could not show: (1) on essay text the suffix/PLE
drafter is dormant - **2 windows of ~511** carried a draft (warm turn: 1 of 2 accepted) - so the
tuned config's decode is effectively **T=1 windows, one token each**; `--spec 4` costs nothing
there but also nothing is gained. (2) The ctx8192/167-slot arm decoded 25.11 vs the tuned arm's
20.11 tok/s - an OBSERVED CONFIGURATION DIFFERENCE, not a causal slot-count claim: that arm
differs from the tuned one in slots AND context AND spec (2-session pair), so no "41 slots are
worth 5 tok/s" attribution follows without a matched ctx/spec, fixed-slot-count control.  (The
matched control now exists - A/B #11 below measured it: 167 vs 126 slots at FIXED ctx8192/spec2
is +4.6 tok/s warm.)

**Attribution of one warm decode token** (tuned cfg, stable state; `STRATA_DECODE_TIMING=1` +
`STRATA_VERIFY_PROFILE=1` session for the host/GPU decomposition, `STRATA_EXL3_TIME=1` session
(3 turns) for the pool counters, `exl3_moe_window_bench` idle-GPU for the pure kernel anchor;
each instrument run has an unprofiled wall control within ~1-5%):

| component (per T=1 window = per token) | ms | instrument and boundary |
|---|---:|---|
| window wall | 49.55 | decode-timing line; DONE line gives the same per token (20.1 tok/s) |
| GPU dense/attention/router/shared/head, host waits for the GPU to ring | 13.61 | "GPU-reach wait" (host view); GPU stamps sum ~15.3 incl. gaps |
| host expert pool, exposed behind the doorbell (48 layers) | 34.99 | "per-layer host"; equals EXL3_TIME pool 35.29 and the GPU's `copy+combine` stamp interval 35.03 - the three instruments agree, and the doorbell interval = the pool wall: **no graph work overlaps the host service on a single GPU** (G=1) |
| - miss staging: pread(page cache) -> pinned buffer -> H2D | 18.16 | EXL3_TIME "upload" (staging-stream event span) for 95.91 MiB = 5.5 GB/s effective; 87.3% hits |
| - FFN issue + kernels (48 x indexed GEMV + f16->f32) | 13.35 | EXL3_TIME "FFN" event interval; pure kernel alone is 10.8-11.8 (window bench 0.226-0.245 ms/layer-window, K3, 10 routes) -> ~2 ms host issue/sync overhead |
| - route plan + LRU + D2H output + host scatter + input upload | ~3.6 | pool minus the two above (remainder, two instruments) |
| commit/stage/draft | ~0.2 | decode-timing |

The intervals above are NOT independent savings: staging and FFN both live inside the doorbell
interval; the wall is dense (13.6) + pool (35.0) + ~1. At 32 slots the same anatomy shifts to
pool 80.77 (staging 61.88 for 351.67 MiB at 5.96 GB/s, 59.2% hits; FFN 12.97 unchanged) on a
94.6 ms wall - the FFN is cache-size-invariant, staging scales with misses. The staging path is
already pinned-buffer pread + DMA (exl3_experts.cu's Stage workers), and the 5.5-6.0 GB/s
effective span sits well below the 28.2 GB/s pinned link. WHY it sits there was not isolated in
this profile: page-cache `pread` and the 9-planes-per-expert transfer granularity are plausible
contributing hypotheses, not a measured causal breakdown.  (The decode-isolation bench below
has since measured the pieces: the 9-plane pattern costs ~2.8 ms/window more than one
contiguous copy of the same bytes IN ISOLATION, the isolated current-pattern transport of a
window's misses is 7.1-7.8 ms, and most of the engine's 17-19 ms span is the pread path's
engine-context machinery rather than bytes-moving - see A/B #12.)

**M5 inputs (read-only; nothing implemented).** The seams on the final tree are unchanged from
the probe above: `verify.cpp:587-589` still rejects `external_experts` with a residency table;
the EXL3 window node is `wait_flag_ge(m_flag_, ring)` + `copy_from_mapped` of the WHOLE expert
output (`verify.cpp:1639-1641`); the host side is `drive_exl3_pool_multi` -> `run_multi`'s
host-IO arm, which stages misses and then issues ALL routes (hits and misses together) in one
indexed launch after the staging joins - unlike the single-token `run()`, which already splits
hit FFN from miss staging. The device slot-table hit-arm primitive
(`exl3_moe_rows_slot_table`, `include/strata/kernels/exl3.hpp:108-125`) was re-validated on this
exact binary (GPU0): byte-identical vs host arrays, invalid-slot throws before launch, and
graph capture/replay byte-identical, K3 and K4 (`slot-table-parity.log`, 63 PASS, exit 0).
MODELED ESTIMATES for the candidate paths (warm tuned state, per token) - arithmetic on the
diagnostic intervals and the synthetic kernel anchor above, NOT measured upper-bound benchmarks.
They assume the hit-share kernel cost is proportional to the hit-route fraction of the
window-bench figure, and leave real contention, launch and sync behavior under concurrency
unmodeled; each number is a hypothesis to test, not a result:

- **In-call hit/miss overlap in the host-IO arm** (issue the hit routes' FFN before staging
  joins, merge after - `run()`'s existing pattern, no graph change, one doorbell): pool
  ~ max(18.2 staging, ~9.8 hit kernels) + ~1.4 miss FFN + ~3.6 glue -> wall ~37 ms ->
  **modeled ~27 tok/s (+34%)**.
- **Full M5 in-graph resident hits** (E-6-style flagA early publish + slot-table kernel; misses
  still host-staged): wall ~ 13.6 + max(18.2, 9.8) + ~2 -> **modeled ~29.6 tok/s (+47%)** - only
  ~2.6 ms/token better than the simple overlap in the model, because staging (18.2 ms) then
  binds.
- **All-resident ideal** (no misses at all): 13.6 + 11.3 + ~2 -> **modeled ~37 tok/s (+85%)**;
  infeasible here: ~45.6 GiB of slot data against a 24 GB card (the earlier `allres-split`
  attempt aborted in allocation).

Recommendation for the next bounded implementation (NOT started): the in-call overlap first -
it reuses a proven in-file pattern, needs no new host/GPU protocol, no graph changes, and its
fallback is the current order; validate with the wave-checker battery, needle/essay A/B, MTP
acceptance unchanged, and STOP if it measurably serializes worse than the model above (for
instance if a mid-call drain of the hit pass is needed and its cost eats the overlap).
M5's graph work stays second and only if staging is attacked afterwards; full residency stays
out of scope on this hardware. Limitations of this profile: single-session pairs for the
secondary arms (the primary arm has 3 repeats, spread < 0.3%); the diagnostic runs perturb
(EXL3_TIME ~+1%, VERIFY_PROFILE/DECODE_TIMING disables fusions, ~+1-5% wall) and every wall
claim above comes from unprofiled runs; the window bench is synthetic weights at the model's
geometry; no cold-page-cache series (needs root), no CUDA/SYCL build (toolchains absent).

### A/B #11: host-IO hit/miss overlap, `STRATA_EXL3_OVERLAP=1` (2026-10-17) - kept, opt-in, modest

Implemented at `aba99fc3` (timing-accounting fix `2aac2b31`; engine shas and raw logs in
`bench/results/2026-10-17-hip-exl3-3bpw-decode-overlap/`, original checkout).  The host-IO
wave (run_multi - the verifier windows; run_device/prefill untouched) classifies its groups
hit/miss in the existing admission loop (hit slots were already pinned before any miss could
evict them), uploads the permuted route table (hits first) and enqueues the HIT pass' indexed
GEMV BEFORE `stage_parallel()` joins the miss staging; the MISS pass follows the existing
`stage_done` wait and writes the `[nh, total)` partition of the SAME buffers.  A partition, not
a copy: no mid-call drain, no extra device memory, one D2H and one barrier per wave (the
original cadence).  Both passes' workspace geometry is pre-reserved before the first enqueue
(the entry points' grow-only reserves would otherwise free buffers the queued hit kernels
read).  All-hit, all-miss, cache-0 and wider-than-one-tile waves keep the original order
exactly; unset/0 allocates nothing new and enqueues nothing differently.

Correctness: the wave checker's overlap battery (all-miss, all-hit, a genuinely mixed wave,
duplicate ids in one token, tokens 1/2/4/8, a 512-route 20-wave window with eviction and
cross-call workspace growth, cache 0, a nonzero layer range with the fixture's other K/FFN
geometry, and lend_tail -> window inside the loan -> restore_tail -> window) requires
BYTE-identical outputs vs the original order in both kernel regimes and asserts the split
counters prove the split ran (`overlap_debug`; no silent fallback); the phase-sequence leg grew
to six arms (default/pipe/arena x overlap on/off).  HIP Release: exl3/ple 24/24 on both XTX,
the checker 3x on GPU0 and once on GPU1.  On the real model: needle 3/3 keys both arms with
identical 37-token streams, warm essay 512 streams byte-identical across arms, and the packed
MTP arm's acceptance is IDENTICAL (139/231) in every session - the four original ones via
their preserved DONE lines (`runner-all-output.log`; the directory-reuse mistake below cost
those sessions' raw streams) and the four closure sessions with full raw streams and
byte-identical outputs.  CUDA/SYCL not built
(no nvcc/icpx); the change is plain host C++ on headers the file already uses.

Numbers (3bpw, GPU0 = 0000:03:00.0, tuned cfg fixed in both arms, alternating order-varied
pairs, warm page cache, unprofiled walls):

| workload | base | +OVERLAP | verdict |
|---|---|---|---|
| warm essay 512 (3 pairs) | 20.22 [20.19..20.26] | **20.68 [20.67..20.71]** | **+2.3%**, ranges do not meet, streams identical |
| same, turn 0 (cold expert cache) | 19.77 | 20.23 | +2.3% |
| low cache 32 slots, 256 tok (2/arm + 2 closure pairs) | 10.57 / 10.71 / 10.68 / 10.60 | 10.78 / 10.77 / 10.71 / 10.66 | a wash within noise (<=1%) |
| packed MTP 256 (2/arm + 2 closure pairs) | 21.99 / 21.99 / 21.93 / 21.90 | **24.14 / 24.14 / 24.04 / 23.98** | **+9.6..+9.8%**, acceptance identical |
| needle 37 tok (single runs) | 20.3 | 22.7 | noisy short sample; keys 3/3 both |
| turn-0 prefill ms (smoke) | 6023-6064 | 6023-6080 | overlapping bands: prefill unchanged |

VRAM identical (INFO 2264 MiB free both arms; the split partitions existing buffers).

**Event-measured evidence, and why the win is 2.3% and not the modeled +34%.**  The
STRATA_EXL3_TIME evidence line (events, not env inference): ~56% of the layer-waves split; the
hit pass spans 6.8-7.5 ms per window and **99% of it executes inside the staging span** (the
overlap mechanism works exactly as designed); the pool drops 35.50 -> 33.59 ms.  An earlier
draft of this paragraph attributed a "remainder growth from ~3.9 to ~5.5-9.5 ms" to the second
pass's host-side issue cost; BOTH parts of that were wrong.  The 5.5-9.5 figure came from the
pre-`2aac2b31` diagnostic (the FFN events were read before the barrier, understating FFN and
overstating the remainder - a measurement error), and the isolated two-pass issue cost is now
MEASURED at ~0.010 ms per call set (the transport bench's part B, below), i.e. ~0.5 ms per
window - it cannot explain any multi-ms remainder.  With the corrected accounting the overlap
arm's FFN interval spans the staging by construction, so pool-minus-parts is not a meaningful
decomposition there.  The decode-refresh model was wrong for a different reason: it assumed
kernel cost proportional to route fraction and that hiding it shortens a host-bound pool; the
honest number is the measured +2.3%.  The MTP arm wins ~9.6-9.8% because its T=4 windows hide
a 4x wider hit pass behind the same staging.  (First diagnostic run's FFN figure was wrong -
the elapsed read preceded the barrier; fixed in `2aac2b31`, walls unaffected, control in
band.)

**The matched slot control** (the decode-refresh correction, now measured): ctx8192/spec2
fixed, ONLY the slot count differs - 167 slots 25.30 [25.20..25.40] vs 126 slots 20.67
[20.58..20.76] warm.  The 41-slot difference alone is +4.6 tok/s (+22%) on essay decode at
that ctx/spec - capacity, not the overlap, is the bigger measured lever, and the tuned config's
126 slots leave 2264 MiB free against a 2048 MiB reserve (a capacity decision is a separate,
explicitly not-taken step).

**Review closure (same day).**  (1) The overlap arm's two profiling events are destroyed with
the Impl's other events (`e7d9e43c`; constructor partial creation unwinds through ~Impl, the
`profile[]` pattern).  (2) The original secondary-arm loops reused output directories, so each
low32/mtp arm's FIRST session kept only its DONE line (all of them survive in
`runner-all-output.log` - the original four MTP sessions' 139/231 acceptance and walls come
from there); the closure reran both controls as two alternating pairs each in UNIQUE dirs on
the FINAL binary (`6bb84c1f...`, post `e7d9e43c`): MTP base 21.93/21.90 vs overlap
24.04/23.98 with acceptance 139/231 in every session and byte-identical streams across arms
and across same-arm sessions; low32 base 10.68/10.60 vs overlap 10.71/10.66, byte-identical
streams - within the combined evidence the low32 effect is a WASH within noise (the original
2-session +1.4% did not hold up), and the table above carries both.  One unprofiled warm
control on the same binary: 20.82 tok/s, in the arm's band.  Checker + exl3/ple 24/24 re-run
on this exact binary (`closure2-ctest.log`).  (3) Exception lifetimes, stated explicitly: the
split's only new host arrays are the wave-local `group_hit` (never a GPU source) and the
permuted reuse of the wave bank's vectors plus `ov_dest` (function-scope, declared BEFORE the
StreamDrain guard, so they outlive its compute-stream sync during unwinding); the pageable
uploads the split enqueues on the compute stream read exactly those bank vectors, the staging
stream's only host sources remain the Impl's persistent pinned Stage buffers (the change adds
no Job fields and no work-stream source), and the existing catch still drains both streams and
invalidates residency.  Stage-failure behavior is inspected, not fault-injected.

Verdict: **kept, opt-in** (`STRATA_EXL3_OVERLAP=1`, one knob, host-IO only; the recommended
tuned configuration does NOT include it yet - a workload-dependent +2.3% on T=1 essay decode,
+9.6-9.8% with the packed MTP drafter, wash at 59% hits).  Next measured bottleneck if decode
is to move further: the ~18.5 ms staging span (staging-span-bound, with the host-read-vs-DMA
fragmentation split unresolved - the decode-refresh hypotheses) and the unattributed
remainder above - a follow-up must beat the measured 33.6 ms pool, not a modeled figure.
Limitations: diagnostics perturb ~1%; no cold-page-cache series; the dirty decode-overlap
sketch in the audit worktree was READ as a reference only and is untouched.

### A/B #12: decode-isolation - the staging span decomposed, and the pread-path finding (2026-10-17)

A NON-PRODUCTION diagnostic (`strata-exl3-transport`: `src/kernels/exl3_transport_bench.cpp`
plus the doorbell-poll kernel in `src/kernels/cuda/exl3_transport_spin.cu`; no engine code
changed, no default-path timers) isolates, on the REAL 3bpw miss geometry (one warm window's
distinct misses = 54 experts, 9 planes each, 95.91 MiB, layer-uniform 1.78 MiB/expert), what
the engine's ~17-19 ms staging span is made of.  Engine binary unchanged (`6bb84c1f`,
tree `70a687a9`); artifacts `bench/results/2026-10-17-hip-exl3-3bpw-decode-isolation/`
(identity, all raw harness/engine logs, DERIVED.json, cost definitions in the harness header;
cpu_* sums are CPU work across threads, never wall; every GPU interval elapsed only after its
stream synced; events destroyed; the mislabeled first-batch filenames are explained in
NOTE-FILES.txt and superseded by the `-2-` runs).

Isolated transport arms (one window's bytes, warm page cache, idle GPU):

| arm | wall ms | dma span ms | note |
|---|---:|---:|---|
| (a) engine worker pattern, 15 workers | 7.11 | 7.39 | cpu_pread sum 13.3, in-lock enqueue 7.2, lock-wait 5.3 (CPU work, not wall) |
| (a) same, 1 worker | 15.5 | 15.6 | |
| (a+spin) + doorbell-style PCIe poll | 8.20 | 8.52 | **the poll is NOT the engine gap** (equal within noise at K=1/2/5) |
| (b) pinned source, same 9-plane pattern | 6.33 | 6.45 | isolates DMA+enqueue from reads/workers |
| (b-one-big) one contiguous H2D | - | 3.54 | 28.4 GB/s, the bandwidth UPPER BOUND, not an engine prediction |
| (c) pack + one H2D per expert | 3.88 | 3.85 | pack CPU 0.048 ms/expert (~2.6 ms/window) timed separately |

Per-layer batches (the engine's granularity) cost 0.058 ms (1 expert) to 0.44 ms (5) - 48
layers of the engine's actual mix extrapolate to ~5-7 ms, matching the batched arm.  **The
engine's span (17.4-19.2 ms) is therefore ~10 ms MORE than every isolated measure of the same
bytes**, and it does not move with the worker count (STRATA_EXL3_WORKERS=1/4/15 diag sessions:
upload 19.2/17.8/17.4, pool 34.1/32.9/32.5, wall 19.9/20.8/21.3 tok/s - MORE workers is
better, refuting wakeup-churn).  Untouched layers (30, 47) measure the same as layer 0 - with
the caveat that the harness's warmup touches its own pages first, so a first-touch cold
effect is not excluded by that arm.

The split's pure issue cost (part B, device-resident weights, no reads, workspace
pre-reserved, outputs byte-identical): one all-routes indexed call vs two offset passes costs
0.013 -> 0.023 ms host-issue (T=1, 9+1) and 0.014 -> 0.022 (T=4, 35+5) - **~0.010 ms per
call set, ~0.5 ms per window**; the GPU spans are equal within noise (0.343 vs 0.350 T=1).
The prebuilt slot-table primitive (the test-only `exl3_moe_rows_slot_table`, READONLY probe)
is byte-identical, costs 0.051-0.052 ms host-issue (it validates ids by readback each call)
and 0.255 ms GPU span at T=1 vs 0.350 for the host-array path - a device-side descriptor path
is not free today and not obviously cheaper at T=4 (0.696 vs 0.671).

**The finding: `STRATA_EXL3_NO_PREAD=1` - a pre-existing A/B env arm, no code change - wins
+22.5% warm sustained decode.**  The diag session with it drops the staging span 17.4 ->
10.43 ms and the pool 32.5 -> 25.5; the paired UNPROFILED A/B (warm essay 512, two
order-varied pairs, tuned cfg + OVERLAP fixed in both arms) measures pread 20.23/20.20 vs
NO_PREAD **24.78/24.78 tok/s**, and the warm-cache cold-expert turn-0 prefill IMPROVES from
6035/6044 to 4802/4831 ms.  Outputs are byte-identical (both essay turns, both arms, and the
needle's 37-token stream matches the pread arm's; 3/3 keys; needle decode 31.3 tok/s).
ATTRIBUTION, corrected: switching this env changes a BUNDLE - no pread syscalls, no pinned
worker buffers, no double-buffer event reuse, and pageable-runtime copy semantics from the
mmap - so the result identifies the read-path bundle as the lever, not any single mechanism
inside it.  And it does NOT "retro-explain" A/B #11's remainder: that remainder was invalid
accounting (the pre-fix FFN read), and an invalid quantity has nothing to explain.

CONDITIONALITY, stated plainly: all of today's runs ride a warm page cache (71 GiB
buff/cache; the model + PLE are 81 GB against 93 GB RAM).  The in-file cold-case note
(`exl3_experts.cu`: mmap reads at 0.3 GB/s against 10.4 GB/s for `preadv` when pages fault
from disk) is a HISTORICAL measurement that was NOT reproduced here; today's warm-cache
numbers and that note describe different, unverified-to-be-comparable conditions, and a
controlled cold measurement is still missing (`drop_caches` needs root, not available).
Two decomposition caveats also corrected: the harness "wall" (host clock, ends when the last
worker reports) and "dma_span" (first-enqueue event to last-copy completion) have DIFFERENT
boundaries and are never subtracted; and the transport arms' ~2.8-3.7 ms spread is an
isolated microcost extrapolation, NOT a proven engine-side ceiling.

**Recommended next bounded step (review-shaped, no production change yet):** reproduce the
warm gain (one order-varied pair) and measure the packed-MTP combination and a low-cache
sensitivity pair; measure the cold side with a bounded, self-created scratch fixture
(deterministic bytes, fsync, posix_fadvise(DONTNEED) on THAT FILE ONLY, mincore-verified
residency before every arm, multi-trial) - never by pressuring the machine's global cache.
A runtime adaptive selector is NOT proposed: page-cache state changes after startup, one
cheap probe cannot track it, and the existing env knob already provides the opt-in.  The
recommendation to carry is an EXPLICIT, documented warm-serving choice
(`STRATA_EXL3_NO_PREAD=1` for this machine's warm-cache serving) unless the cold fixture
produces a new fact; a dynamic policy would need its own review.  Transport work (packing,
fewer planes) is unattractive under the current evidence - with the extrapolation caveat
above.  M5 graph work, GPU helper, capacity and native MTP remain out of scope.

### A/B #12 review follow-up: the read-path knob verified, and the cold fixture (2026-10-18)

This executes the bounded step A/B #12 recommended.  Engine binary unchanged (`6bb84c1f`).  Two
artifact sets, and the difference matters: `bench/results/2026-10-17-hip-exl3-3bpw-decode-readpath/`
holds nine runs whose env was NEVER captured (their report.json/run.txt do not record it and the
knob is silent in the engine logs) - the run NAMES label `STRATA_EXL3_NO_PREAD`'s arms, but a
label is not evidence and inferring the env from the measured speed difference is circular, so
everything measured there is **PROVISIONAL (historically labeled, unverified env)**;
`bench/results/2026-10-18-hip-exl3-3bpw-readpath-verified/` holds the **CERTIFIED** closure: six
runs whose effective env is captured explicitly BEFORE each engine spawn (scrubbed shell env +
copied config snapshot = the child's full env, per-run run.txt), the two arms proven to differ
only by `STRATA_EXL3_NO_PREAD` (baseline UNSET - presence arms the knob, "=0" would arm it).
All walls unprofiled (`STRATA_EXL3_TIME` unset).  DONE semantics (src/program/generate.cpp):
`prompt_ms` = prompt read + expert-slot refills; `decode_ms` = the FIRST verify window (first
token INCLUDED) through the last window's exit, excluding the trailing commit wait; net tok/s =
produced / decode_ms.  Essay prompt 512 tokens, temperature=0 seed=7319, warm page cache.

**Certified (2026-10-18, verified env, GPU0 = 03:00.0, config snapshots in the run dirs):**
packed MTP (256 tokens, `--spec 4 --spec-min-p 0.5 --mtp rt`), TWO order-varied pairs - pair1
25.08 -> 31.20 (+24.4%), pair2 25.03 -> 30.97 (+23.7%), mean **+24.1%**, 2 runs per arm; warm
control (512 tokens), one pair, 21.30 -> 25.48, **+19.6%**, 1 run per arm.  Draft acceptance
INVARIANT at 139/231 (60.2%) in all four MTP runs, both turns; all four MTP token streams
identical (512 each) and the warm pair's identical (1024).  Turn-0 cold-slot prefill improves
~18% (warm pair 5797.6 -> 4769.5 ms; MTP pair2 5829.7 -> 4746.7 ms).  One outlier, reported not
hidden: `v-mtp-p1-nopread` ran first after the machine had sat idle and shows a partially cold
page cache (prompt 33.1 s vs 4.7-5.8 s elsewhere, inflated turn-0 decode); its turn-1 (31.20)
matches its pair-mate sibling (30.97), and the certified comparison is turn-1 (VRAM-warm,
cache warm within the run).  Do NOT compose across workloads: MTP 256-token and warm 512-token
runs are different trajectories and window mixes - the two gains are each a within-workload A/B,
and any MTP-vs-warm number is a cross-workload reference, not a composed gain.

**Provisional (2026-10-17, env unverified) results, kept for the record:** warm one pair
(nopread-first, 2x512 tokens) 20.25/20.71 vs 25.01/25.57 (+23.5%), turn-0 prefill 6024 -> 4731 ms;
packed MTP two order-varied pairs + one unpaired base run 24.06/24.09/24.03 vs 31.04/31.05
(+29.0%), acceptance 139/231 in all five; low32 one pair 10.69 -> 15.84 (+48.1%).  All streams
byte-identical within each pair.  The extra fifth MTP run (base-only, `mtp-rp-base-p3`) is
reported with the rest, not dropped: the run.txt timestamps show (nopread,base),(base,nopread),
then one base-only drift sample (24.03, within 0.3% of its siblings) before the low32 pair; the
one-line run.txt does not record why it exists.  These are HISTORICAL LABELS ONLY - env was
never captured, the +23.5/+29.0/+48.1% cannot be attributed to the knob from these runs, and
the earlier "+49.9% composed" reading (MTP-only +16.2%, NO_PREAD-only +23.5%, both +49.9% vs the
warm arm) is withdrawn as a composition claim: MTP 256-token and warm 512-token runs are
different trajectories and window mixes, so that comparison is cross-workload context at best.
Replicates, stated as they are: 2 runs per arm (MTP), 1 run per arm (warm, low32).  The
low32 pair is provisional historical (env unverified) and was not re-run.  The certified
numbers above supersede these for citation; the provisional ones bracket them (day-to-day
absolute drift a few percent, gains +23.5%/+29.0% provisional vs +19.6%/+24.1% certified).

**Cold fixture** (NON-PRODUCTION `strata-exl3-transport --fixture`): a self-created 95.91 MiB
scratch file (54 experts x 9 planes = one warm window's real miss geometry), deterministic
non-zero bytes, fsync'd; per-file `posix_fadvise(DONTNEED)` + munmap/remap, mincore residency
verified before EVERY warm arm (30/30 checks full) and every cold trial (0 pages resident), never
the model file, never root or global cache pressure.  The review fixed a real race in the first
build's pread arm (one pinned buffer overwritten while its async H2D still read it - torn source,
extra accidental overlap): the fixed arm ping-pongs 3 pinned buffers with per-copy events, so the
source lives until its copy drains.  Race-free results (`fixture-cold-2.txt`): WARM mmap-copy
6.970 ms vs pread+pinned 12.580 ms (both arms verified to move the file's exact bytes), COLD
(mincore-verified, n=8/7) pread 198.173 vs mmap 201.240 ms, read_bytes 1508597760 = ~15 x the
file, major faults 7.  Read plainly: warm, the mmap-copy arm is cheaper in this fixture -
DIRECTION-consistent with the engine's warm gain, not a prediction of its magnitude (one reader
thread, no workers, no engine context); cold, the arms are equal within 1.5% at ~0.48 GB/s (this
volume caps both) - the historical in-file note (mmap 0.3 GB/s vs preadv 10.4 GB/s faulting from
disk) is NOT reproduced and stays unverified-to-be-comparable.  **The fixture does not establish
real-model cold TG**; it measures the two read arms under controlled residency.

**Wall vs span, defined** (the old 7.11 wall < 7.39 dma_span question): the transport bench's
"wall" is a host-clock MEDIAN ending when the last WORKER reports (before the DMA tail drains);
"dma_span" is an EVENT MEAN, first-enqueue to last-copy completion, read after sync.  Different
boundaries AND different statistics; never subtracted - wall < span is expected when the final
copies outlast the worker join.  The fixture's wall instead brackets the whole arm through its
final `cudaStreamSynchronize` (both boundaries on the host clock, DMA drained).

**Correctness (read arms)**: `tools/test_exl3_loader.py::test_read_path_arms_byte_identical`
runs the full wave battery - duplicate ids, hit/miss mixes, cache 0, eviction, PIPE/OVERLAP/BATCH
arms, the loan interplay, the phase sequence - as TWO PROCESSES (`STRATA_EXL3_NO_PREAD` unset vs
present; the knob is a once-per-process static, so a within-process setenv cannot switch it),
each folding every compared output into a SEQUENCEHASH (FNV-1a).  Coverage, stated precisely:
the byte-identity folds cover the checker's `compare()` outputs - the run_device/run_multi
windows, prefill legs and loan/overlap battery - while the single-token `run()` legs go through
`compare_rel`, which folds LENGTHS only (that path is tolerance-based by design because of
atomic-accumulation noise); cross-process VALUE parity for `run()` is therefore NOT asserted -
no blanket byte-identity claim over atomic outputs.  Both processes print
`SEQUENCEHASH 395bca77f2a837a2`, PASS, on GPU0 and on GPU1 (same hash on both).  The 24 exl3 ctests (HIP Release) pass 24/24 on both GPUs.  The
diagnostic harness itself closed out: the transport bench's doorbell-poll spinner releases
through every path including exceptions (a spinning kernel would hold the GPU), the fixture's
cleanup guard owns its GPU/CPU/file resources, and the two diag targets stay inside the
`STRATA_ENABLE_HIP OR STRATA_ENABLE_CUDA` gate.  CUDA/SYCL still unbuildable here (no
nvcc/icpx), so no claim about them.

**Recommended next step (one), executed after the certified runs above**: docs/EXL3.md gained a
small opt-in warm-serving section for the EXISTING `STRATA_EXL3_NO_PREAD=1` knob - the certified
3bpw/7900 XTX numbers, the packed-MTP acceptance, the exact `STRATA_EXL3_OVERLAP=1` companion
settings, HIP-only testing, the presence-based semantics (UNSET to disable, "=0" still enables),
the real-model cold/cache-pressure unknown, and an explicit warning that the ~31 tok/s is this
model and hardware, not a promise.  No automatic selector (a startup residency snapshot - mincore
observes residency fine - does not predict later residency or performance; a dynamic policy would
need its own review), no default change, no
engine code touched - the default path stays byte-identical (engine binary hash unchanged).

### Fast-config attribution + the M5 bounded experiment (2026-10-18): M5 deferred on judgment, not measurements

Follow-up to the review of the section above.  Accepted fast config (the six tuned kernel flags +
`STRATA_EXL3_OVERLAP=1` + `STRATA_EXL3_NO_PREAD=1`), env captured before every spawn, artifacts
`bench/results/2026-10-18-hip-exl3-3bpw-decode-m5-fastpath/` (`SUMMARY-M5-FASTPATH-2026-10-18.txt`
has every number and boundary).  Unprofiled controls: warm essay 512 **24.93 tok/s** net, packed
MTP 256 **30.60 tok/s** net (acceptance 139/231 in both), streams identical to their diagnostics;
the event-stamped diagnostics run 3-6% slow and are used for decomposition only.

Per warm window (1 token, 48 layers, 42.77 ms): host side = GPU-reach wait 13.69 + per-layer
host 28.12 (the verifier service loop - the [plan actq jobs CPU] sub-counters are the GGUF
pool's and read 0 on the EXL3 path, they are NOT "no host work") + input staging/PLE 0.25 +
commit/emit 0.01; device side total 42.77, of which **copy+combine 27.4 ms is the EXL3 expert
SERVICING INTERVAL, not a staging measure** (CORRECTED interpretation; the earlier "staging
critical path 64%" reading is withdrawn).  What the stamps actually cover on this path, from the
sources: `prof_sum[i]` is the interval `stamp[i-1] -> stamp[i]`, and a skipped stamp folds into
the next fired one; the `external_experts` branch of `post()` (verify.cpp ~1639) fires NONE of
the stamps 19-23 (`waitA`/`VRAM hits`/`waitB`/`PCIe grp`/`waitCPU` are the GGUF and ar_on
branches' stamps), and instead does `wait_flag_ge(m_flag_)` + `copy_from_mapped(m_ymiss_)`
INSIDE the interval that prints as `copy+combine` (stamp 18 -> 24).  So `copy+combine` = the
per-layer doorbell wait for the host's expert service + the mapped miss-output copy + the
combine kernels, and the zeros of waitA/waitB/waitCPU are VACUOUS here - they do NOT show the
host hidden; the graph stream is blocked per layer on the doorbell while the expert stream's
kernels run.  The three views of the same service family CONCUR and are never summed: GPU
interval 27.4 ms/window, host service wall (`ms_pool`) 28.12 ms/window, and the EXL3 call-wall
counter (`pool`, 26-45 ms/window, ~35 typical) with its own boundaries.  What did NOT fire at
all on this path: the expert FFN kernels never appear in the verifier stamps (they run on the
expert stream inside the service interval); the ~10.8 ms of named stages that did fire are the
window's NON-expert kernels (attention/GDN/QSA/head/hc) plus router+plan+bell 1.8 and
shared+quant 1.6 - label names must not be read as the GGUF path's semantics.  Only two
sub-counters of the service interval are separately measured (EXL3_TIME events): the upload
span (an event span over the staging interval INCLUDING host enqueue gaps - not a pure-DMA
number) and the FFN counter, which is INCLUSIVE (both OVERLAP passes including the staging
wait, per the source) and therefore never subtracted anywhere.  MTP (T~3, 2.19 tok/window): the service interval is 53.8 of 71.1 ms.  The
EXL3 staging lines show churn inside the service: warm windows re-upload 97-335 MiB with 48-223
evictions each (cache 352-432 of 480 lookups warm; MTP 263-668 MiB, 148-376 evictions) - the
126-slot cache churns.

M5 (`strata-exl3-transport --m5`, kernel/protocol microbench, NOT engine TG): 16 pinned
resident experts at the real 2560->640 K3 geometry, k=10, T=1/T=4, 200 reps; slot-table vs
host-array byte-identical over 3 route sets (one with duplicate slots), invalid id outside
capture throws before launch, one captured graph replayed x6 over changed route inputs stays
byte-identical.  Costs (host-issue | gpu_span ms): host-array 0.025|0.272 (T1) 0.014|0.717 (T4);
slot-table public 0.058|0.304 / 0.058|0.752 (its slot-id readback validation is 0.031 host and
runs OUTSIDE a capture by contract); graph replay+inputs 0.003|0.260 / 0.003|0.721.  These
numbers bound ONLY the kernel-call issue path of the primitive: the synthetic graph replay
validates the primitive's contract, it does NOT measure the savings of removing the host
protocol (the callback roundtrip or the graph-side hit/miss overlap headroom are not in this
microbench at all).  What the microbench does establish: the GPU kernel diff between the two
dispatch forms is ~0.03-0.04 ms either way at T1/T4.  The discarded 27/37 ideal extrapolations
(slow pread arm, invalid remainder) are not reused.

Engine-hook feasibility (inspection; `verify.cpp` untouched): the seams exist - HitFn
Launch/Combine, `VerifyHits::external_experts` + mapped doorbell + `run_multi` (stays the
host-miss fallback, misses merged exactly once), E-6 device planning, and the zero-doorbell
all-resident window graph.  The gap: `all_resident_`/E-6 need the GGUF residency table and
verify.cpp:587 rejects `external_experts` from carrying one, so the EXL3 path never arms the
zero-doorbell graph.  Smallest integration, should this ever proceed: an EXL3-side residency
mirror satisfying the same check, the slot-table resident call captured into the window graph
writing straight into `parts`, the host arm kept for misses, commit/rollback graphs and MTP
acceptance gates untouched, slot pinning = the existing wave rule.

**Verdict on graph integration: DEFERRED as engineering judgment, not as measured-no-headroom.**
The earlier "NO-GO because the host protocol is measured hidden" is WITHDRAWN (wrong: the
EXL3 doorbell wait sits inside `copy+combine`, see above) and with it the "staging-bound
64-76%" and "0.03 ms on 0.89 ms therefore NO-GO" arguments.  What is actually established: the
per-layer expert service interval is 27.4 ms (GPU view) / 28.12 ms (host view) per warm window,
its internal split (doorbell wait vs expert-stream compute vs staging vs host CPU) is NOT
decomposed by any existing stamp, and the primitive microbench covers only the dispatch issue
path.  M5's real headroom lives inside that undecomposed interval and is therefore unquantified;
deferring the integration is a complexity judgment (a residency mirror, capture lifecycle and
miss-merge semantics on a path where the split is unknown), not a measurement verdict.

**Proposed ONE next experiment (proposal only - not implemented, awaiting review):** a matched
cache-capacity control on the FAST config - `--expert-cache 126` vs ~165-167 slots/layer, both
arms otherwise identical (OVERLAP=1 + NO_PREAD=1, same flags, ctx 65536, spec/MTP as measured).
The capacity lever is chosen because the churn evidence (48-223 evictions/window re-uploading
97-335 MiB) is compatible with a set-size problem as well as an LRU-order problem, and capacity
is the first to price.  Prerequisites BEFORE any run (do not just force an OOM): a budget and
borrow-eligibility analysis - 126 -> ~165 slots/layer is +~3.2 GiB of expert cache (6048 ->
~7920 slots at ~1.78 MiB), against ~2.2 GiB free VRAM at the measured watermark with the
2048 MiB reserve kept unchanged; the STRATA_EXL3_PREFILL_BORROW mechanism works and leaves the
reserve untouched, so the eligibility and the headroom (what the larger cache would evict or
borrow from) must be computed first and recorded.  If the capacity influence on the churn
numbers cannot be established from the existing counters, the fallback is a bounded churn trace
(ids of the re-evicted experts per window on the byte-identical stream) before any policy idea.
No policy implementation, no M5 integration, no selector, no default change.  Validated after
the bench change: 24/24 exl3 ctests and 12/12 loader/wave tests (incl. the two-process read-arm
hash) on GPU0; CUDA/SYCL untested here (no toolchains).

### Matched cache capacity on the fast config (2026-10-18): +12-14% measured, opt-in budget proposal

MEASUREMENT ONLY - no policy, no default change, no engine code (nothing rebuilt).  The one
experiment proposed above, run as an explicit operator A/B: `--expert-cache 126` vs `165`
slots/layer (configs differ at one position), fast config + `STRATA_EXL3_PREFILL_BORROW=1` in
BOTH arms, ctx 65536 spec 4 prefill 8192, `--vram-reserve-mib 2048` unchanged, env captured
before every spawn.  Artifacts `bench/results/2026-10-18-hip-exl3-3bpw-decode-capacity-fastpath/`
(`SUMMARY-CAPACITY-2026-10-18.txt`, per-run VRAM samples at 100 ms).

Budget first, from measured allocations: the arena rank is 89,395,200 B; the startup loan is
39 ranks (3,486,412,800 B) - logged as "borrows 39 of the arena's 126 ranks ... 87 active while
lent" vs "... of 165 ... 126 active", the lend BEFORE the prompt-path carve and the refill
before decode both confirmed from `STRATA_TRACE` (and `prompt_ms` INCLUDES the refill - request
wall = prompt_ms + decode_ms everywhere here).  Settled in-run free VRAM 5505 MiB at 126
(min over the run, sampler) projects 2181 MiB at 165 (+3325 MiB) - measured 2179-2180 MiB,
above the 2048 MiB target by ~132 MiB.  The MTP drafter costs ~890 MiB more and 165+MTP dips to
1289 MiB free: functional in this workload (identical streams, no OOM) but BELOW the headroom
target - a budget rule must price the drafter.

Results (turn-1 net tok/s; raw token streams identical across ALL runs of each workload):
warm essay 512, three alternating pairs - 25.19/25.24/25.25 -> 28.41/28.39/28.19 (+12.8/+12.5/
+11.7%, mean **+12.3%**); packed MTP 256, two order-varied pairs - 30.62/30.53 -> 34.77/34.67
(**+13.6%** mean) with draft acceptance INVARIANT at 139/231.  Prompt side is a coupled effect
of the loan split (126 vs 87 active ranks during PP) and measured as-is: turn-0 prompt_ms
4706 -> 4836 ms (+2.8% at 165), turn-1 prompt 232 -> 206 ms.  Loan lifecycle correctness at
165: 3-key needle 3/3, two-turn warm streams identical in all six warm runs.

Mechanism (matched short diag samples, the ACTUAL per-window distribution, 254 decode windows
each; paired unprofiled controls show the profile costs ~2.5%): misses/window 58.2 -> 37.7
(-35%), hit rate 90.0 -> 93.8%, upload 103.4 -> 67.0 MiB/window (-35%), evictions 55.4 -> 34.2,
staging event span 12.0 -> 8.4 ms (an interval incl. host enqueue gaps, not pure DMA), service
wall 27.3 -> 23.4 ms.  The larger cache removes churn; it does not change the FP kernel path
(outputs identical).

**ONE proposal (NOT implemented, for review): a minimal OPT-IN auto-budget** - when the arena
loan is decided early (that decision already runs before the prompt-path carve) and confirmed
active, let auto-sizing grant +the loan's ranks of cache (the loan-saved bytes), pricing the
drafter's ~900 MiB and the 2048 MiB reserve as today; on any refusal or non-eligible mode the
fallback is SAFE and pre-decode (decline/trim the extra tail ranks before the first decode
window; the existing owned-buffers path stays byte for byte; never a mid-session eviction storm,
never a forced OOM).  The explicit `--expert-cache` operator path stays exactly as it is, and
nothing becomes default without review.  Limits: explicit config only, one machine/model, ctx
65536, thin ~132 MiB margins verified live; other contexts/chunk sizes and non-eligible modes
untested at 165.

### Peer tier feasibility (GPU1 computes) + minimal EXL3 fixture (2026-10-18): GO, adapter not built

MEASUREMENT + DESIGN ONLY - no engine integration.  The user's next target is a "peer" second
card that CACHES experts on GPU1 and COMPUTES their activations/outputs there (per-expert FFN,
not a layer split, no weight pulling per inference).  The existing machinery already defines the
contract (source-inspected): `PeerExperts` (include/strata/core/peer_experts.hpp) is a disjoint
adaptive tier - `open()` fills it with ranked pairs the primary does NOT hold, `launch()` groups
the kind-2 entries and computes them on the peer (q8_1 + `native_expert_grouped`, native packs
ONLY), `finish()` lands the rows in the same pinned portable mapped rows as CPU rows (kind
routing in `expert_source.cpp`: primary 0 / peer 2 / CPU -1; peer launch before the CPU work,
finish before publish - one merge, by construction), `adapt()` swaps neither-card-held experts
per layer between windows.  P2P is probed, not required (decode traffic is host-buffer based).
Constraints kept: `--peer-device` needs `--expert-profile` + the expert cache - the EXL3 path is
currently REJECTED ("needs --expert-profile and the expert cache") and that reject stays until a
real implementation exists; SECOND_GPU.md's remote helper and MULTI_GPU.md's layer split are
DIFFERENT mechanisms.  The native blob API cannot host EXL3 trellis data - no fabricated pack
conversion; an EXL3 adapter uses Exl3Mat/stream backing only.

Fixture (`strata-exl3-transport --peer`, diagnostic mode; artifacts
`bench/results/2026-10-18-hip-exl3-3bpw-peer-feasibility/`, review-fixed runs -3/-4; the first
build's -1/-2 are superseded where the review found errors): 4 resident experts/layer over 2
layers (cache 14,899,200 B EXACT; stream bytes() 128.8 MiB INCLUDING scratch; measured
permanent VRAM 348.0 MiB per stream - ~334 MiB of workspace/staging/events beyond the cache),
k=10, T=1/T=4, 100 reps, two fixed runs agreeing within 2%, P2P 1/1 present but unused.
Correctness: GPU1 rows BYTE-IDENTICAL to GPU0 rows for duplicate and mixed route sets at both
shapes and both layers (finite+non-constant checked); invalid residency refused by a REAL
launch-wrapper pre-check with the stream call counter asserted unmoved (fixture-local
prototype, the production guard is not built); device context restored after every call and at
exit (verified, not assumed); all allocations in setup; every run_multi return and memcpy
checked, failures propagate to exit 1; 6 negative-arg cases exit 2.
Wall per layer-window (run_multi is SYNCHRONOUS - SERIAL roundtrip only, no async adapter is
demonstrated): T1 primary 0.287-0.288 vs peer 0.284-0.288 ms (equal); T4 0.745 vs 0.795-0.796
(+7%); link-only floor 0.055/0.100 ms.  Traffic from the exact calls: T1 = X 10,240 B + rows
102,400 B = 112,640 B (110 KiB) per layer - all 10 rows paid even with 4 unique experts; T4 =
450,560 B (440 KiB); per 48-layer window 5.16 / 20.6 MiB, serial floor bounds 2.64 / 4.8 ms.
The COLD first call on a fresh stream (7,449,600 B fill + compute) is 15.5-15.8 ms - a
fill+compute number, not a pure fill time (the first build's "0.5-1.0 ms fill" was a warm hit
and is withdrawn).  A fixture, NOT engine TG; no theory claim that the roundtrip hides.

Budget (modelled only, corrected arithmetic): usable = 24 GiB - 2 GiB reserve = 22 GiB
(22,528 MiB) BEFORE workspace; minus the measured ~334 MiB permanent stream overhead ->
~22,194 MiB of cache = **~12.5k (layer,expert) slots** at 1,862,400 B each (the model has
24,576 pairs; the earlier ~12.9k was optimistic).  With the primary at 126/165 the disjoint
coverage models at ~18.5k/20.4k pairs - on paper only.  FILL: 22 GiB at ~6 GB/s is a DMA-only
lower bound of **~4 seconds** (the earlier "60 minutes" was an arithmetic error and the
lazy-admission deduction drawn from it is withdrawn); the actual fill cost (host reads +
enqueue) is unknown at scale - the only datapoint is the cold first call above.  Steady
activations cost 2.64/4.8 ms per window if fully serialized (measured floors x 48 layers).
165+MTP's primary headroom finding (1289 MiB) stands; the peer is a separate budget.

**GO for a bounded adapter - ONE initial step proposed, NOT implemented (awaiting review):**
the EXL3 peer adapter's STATIC subset path for DECODE only, opt-in.  Shape (the async contract,
explicitly - `run_multi` is synchronous and is NOT the adapter): a minimal PeerExperts VARIANT
that owns EXL3 device descriptor tables + ONE stream + pinned portable host buffers on the peer
device; its `launch()` only ENQUEUES on the caller's thread - H2D x, `f32_to_f16_bulk`,
`exl3_moe_rows_indexed`/slot-table, `scatter_rows_f32`, all stream-ordered async, exactly
PeerExperts::launch's shape - and returns; `finish()` spins on `cudaStreamQuery` like
PeerExperts.  No new thread subsystem, no allocation in any callback (sources and workspaces
pre-allocated in open()); the EXL3 worker pool stays on the host-IO miss path.  The kind-2
entries are claimed in the EXL3 drive's per-layer queue exactly where `expert_source` does;
the residency pre-check refuses non-resident ids with the existing host-IO path as the fallback
owner; disjointness filters in primary admission and loan refill (no duplicate uploads);
commit/rollback and MTP acceptance gates untouched; gates = existing wave battery + engine A/B
outputs byte-identical + the 2-GPU ctest suite (24/24 both cards after the diagnostic change).
No new host-GPU protocol and no new subsystem is required - the adapter is a client of the
existing async kernel primitives and kind routing (the M5 measurements bound its issue path).
The `--peer-device` reject scope for EXL3 (needs `--expert-profile` + the expert cache) is
EXTENDED by such an implementation, never loosened on its own.  CUDA/SYCL untested here (no
toolchains).

### The async EXL3 peer primitive, implemented with tests (2026-10-18): no engine integration

The proposed adapter shape now exists as a standalone primitive - `include/strata/kernels/
exl3_peer.hpp` + `src/kernels/exl3_peer.cpp` (in `strata_kernels`, existing CUDA/HIP kernel
path) with its battery `src/kernels/exl3_peer_test.cpp` (ctest `exl3_peer_test`).  **Nothing
integrates it**: no generate.cpp change, no external-drive hook, no knob/CLI, the `--exl3`
`--peer-device` reject and the native GGUF path are untouched, budget/capacity policy deferred.
The full contract is the header comment: a static resident (layer, expert) subset with exact
checked geometry uploaded at `open()` (everything preallocated for the worst request; a failed
open unwinds), `launch()` validating the FULL request before ANY enqueue (a non-resident kind-2
id rejects the whole request with nothing enqueued - no quiet clamp; the host-IO fallback stays
the future engine's job) and then enqueueing only (H2D, f32->f16, `exl3_moe_rows_indexed`, D2H)
- no sync, no alloc, no file IO; ONE pending launch (nested rejected, empty is a no-op);
`finish()` spin-waits and host-scatters unweighted rows at unique destinations (non-peer rows
never touched); `close()` drains before any free; every entry restores the caller's device.
Inputs are copied into preallocated pinned buffers before launch returns; a future direct-scatter
variant would require portable-registered out.  A mid-flight enqueue error marks the tier
failed and nothing invalid is ever advertised.  (The `Exl3MoeWorkspace` binds to the device
current at construction - it is created under the peer device guard.)

Tests, CORRECTED record (the first commit's record is withdrawn: review found a partial-upload
double free in open(), non-transactional state on allocation failure, pageable descriptor
arrays, missing open pre-check/fault tests, a same-class oracle that could not catch a common
route/row error, and a hardware-dependent timing assertion; all fixed - planes are owned the
moment they allocate and freed exactly once, open is transactional including exceptions and
injected faults, the device switch is checked, descriptors are pinned, dims/cb/routes caps are
validated before any pointer use, valid()/has() report false on a failed tier, and an
enqueues() seam counter replaces the counter claim).  Artifacts
`bench/results/2026-10-18-hip-exl3-3bpw-peer-async-primitive/` (`peer-async-tests-5.txt` is the
accepted record, full raw `ctest-full.txt`): **207 checks PASS / 0 FAIL, two runs, exit 0**,
every failure path exit-nonzero.  Synthetic: K3+K4 mixed, two geometries (2560->640,
1536->384), layers 1/5 and 2/3, tokens 1/2/4/8 x k 1/10, duplicate and mixed kinds.  Oracle is
INDEPENDENT: per-route `exl3_moe_rows` (k=1) on GPU0 from its own device planes - a wrong
route->row mapping cannot pass.  Cross-path comparison uses the EXISTING wave_check compare_rel
rule (5e-3 norm; the GEMV accumulates with atomics and the two call shapes differ in the last
ULPs - observed norms 7e-8..4e-4 synthetic, 5e-6..4e-5 real); byte-identity is NOT claimed
cross-path and no new tolerance was invented.  Each computed row is finite+non-constant checked
first; non-peer rows keep their sentinel exactly.  Real model: explicit loop over ALL of layers
0/7/23 x T=1/4/8 (the earlier tokens-based layer pick is fixed), mixed kinds included.  Contract
matrix: zero peer rows (enqueues() seam unmoved), non-resident rejection (seam unmoved, out
untouched), invalid dims/caps/nulls/layer (seam unmoved), input lifetime (clobbering x/ids after
launch changes nothing), null-finish semantics, 20x repeat, nested pending, close-while-pending,
caller-device restoration (0 and 1).  Open pre-check matrix (empty/duplicate/geometry/bits/cb/
negative/overflow dims/null plane/caps/routes limit) with VRAM-equality on every reject.  Fault
matrix (STRATA_EXL3_PEER_FAULT): ALL 155 injectable allocation/copy positions fail open and
unwind exactly (VRAM back to baseline, empty state, reopen works); a fault inside launch's
enqueue chain leaves a sticky failed tier that advertises nothing until close().  Enqueue-only
evidence is OBSERVATIONAL (launch 0.49 ms vs finish 45.36 ms, busy() true after launch; not a
pass/fail criterion - hardware dependent).  Suite: 25/25 `ctest -R exl3` on both GPUs, full raw
output archived.

Second corrective pass (same day, review of 4d44fe8f): the `ki%8/nj%8` H128 alignment checks
were missing - a coherent unaligned entry passed open and only threw after enqueue (BLOCKER,
fixed and tested with fake non-null plane pointers, proving rejection before any read/allocation);
plane sizes are now ALL precomputed with checked products/sums before the first allocation or
source read (the per-plane cap moved out of the upload loop); the fault state is per-instance
(no globals; one instance is a single-owner object, like the native tier); the test oracle's own
plane upload had the same partial-free pattern and is fixed; `finish(nullptr)` is ALWAYS an error
and that is documented; every device switch in launch/finish/close is CHECKED (a failed switch
cannot enqueue or free on the wrong device - close drops rather than cross-frees, best-effort
restore only, nothing stronger claimed); the upstream descriptor upload in exl3.cu:743-745 is
now CHECKED and throws into the tier's sticky failure (error-path-only change for existing
users); cross-path tolerance is PER COMPUTED ROW with sentinels compared exactly and oracle rows
checked finite+non-constant too.  Fault evidence, stated exactly: the injectable seams are the
tier's OWN operations (stream + 4 device buffers + 6 pinned + 9 planes x alloc/copy per entry +
4 launch seams) - workspace MoeBuffer allocations, container allocations and the upstream
descriptor upload are NOT injected; the double-free evidence is 2 x 155 cycles surviving the
allocator per geometry; empty-state introspection is exact; device-VRAM leak evidence is
repeat/pass stability within the runtime allocator's observed 32 MiB caching quantum (it
oscillates in BOTH directions - exact equality is not a valid invariant against it) and
pinned-host/heap release is by code inspection of free_all, not directly observed.  Accepted
record (`peer-async-tests-8.txt`, full raw `ctest` inside): 219 checks PASS / 0 FAIL, two runs,
25/25 suite; four additional stability runs passed before recording.

**Next integration, proposed and NOT implemented:** extend the `--peer-device` gate for EXL3
behind an explicit opt-in (extending, not loosening, the current reject); fill the subset from
ranked pairs the primary does not hold; claim kind-2 entries in the EXL3 drive's per-layer queue
exactly where `expert_source` does (launch before the CPU work, finish before publish) with the
host-IO path as fallback owner; disjointness filters in primary admission and loan restore
(`has()` is the exposed hook; the primary's dynamic LRU and loan refill must never duplicate a
peer-resident upload).  Gates: engine A/B outputs byte-identical, wave battery, MTP acceptance
invariance, commit/rollback graphs untouched.  No new host-GPU protocol or subsystem required.

### EXL3 static peer ENGINE integration + lifecycle smoke (2026-10-18): gates green, no perf runs

The peer primitive is now wired into the engine behind an EXPLICIT opt-in (`STRATA_EXL3_PEER=1`
+ `--peer-device` + `--peer-slots`; default and the current EXL3 reject unchanged; native GGUF
peer untouched; `--batch`/layer-split refused).  Static ranked subset (profile or round-robin),
ownership exclusion in `Exl3ExpertStream` (peer-owned experts never admitted to the primary
cache - proven absent after churn AND loan lend/restore; prefill computes them into a bounded
scratch without admitting), and the pool callbacks route kind-2 rows to the peer: launch before
the primary service, finish into their own destinations before publication; zero-peer windows
are no-ops; any peer failure falls through to the FULL primary service (never partial rows).
Integration bug found by bisect and fixed: fallback-computed (excluded) rows released their
scratch slot through the CACHE blocks' used-events under the prefill wave-arm - releasing the
wrong cache slot early and corrupting queued reads (engine streams diverged; unit tests missed
it: host-IO waves never take that release path).  After the fix, every gate passes on the final
binary (`bench/results/2026-10-18-hip-exl3-3bpw-peer-integration/`): peer-vs-oracle lifecycle
(short / 2048-token batched prefill / long / after-cancel streams IDENTICAL, STOP mid-request
cancels, the following valid request identical, tier line present, no fallback line), packed-MTP
A/B (streams identical, acceptance invariant 139/231 both turns), exl3_peer_test (primitive
batteries + the engine seam: mixed/duplicate/zero-peer split == full primary exact by
destination; ownership exclusion after churn + lend/restore; scratch-vs-cache rows
byte-identical), 25/25 exl3 suite and 8/8 gguf/native.  Correctness/lifecycle milestone ONLY -
no performance runs yet (the peer's decode gain is unmeasured; the padded complement recomputes
the few peer rows as dummy rows - a known small cost).  CUDA/SYCL untested (no toolchains).

Correction pass on the integration (same day, review of ba77a4e6/ee5216de; record in
`peer-integration/lifecycle-v3-run.txt` + full-raw ctest files): the required
`include/strata/kernels/exl3_experts.hpp` (the API declarations) had been left out of the
integration commit - committed now (7c155148, checkout builds); the scratch-growth path is
rollback-safe on partial allocation and throwing appends, `set_owned_elsewhere` is transactional
(the table publishes last; any resident owned entry is erased - run() consults residency before
the exclusion, so a retained entry would duplicate the owner's weights) and documented
setup-only; the production debug scaffolding (PEER_TRACE/STRATA_EXL3_PEER_BISECT, per-group
getenv, the unconditional wave_scratch allocation) is removed - the default hot path is
unchanged when no ownership table is set.  The complement is now a TRUE compact call in bounded
8-route chunks (correct token indexing; padded-all-rows was a baseline, not an optimization;
the earlier compact hang was the release-event bug, now fixed and the compact form verified);
caps come from the verifier window bound (<= 8 tokens x top-k <= 64) with safe fallback for
larger windows.  Subset selection deduplicates, skips pairs the primary holds, refuses a
peer_device equal to the primary's, and honors a coarse byte budget (peer free VRAM minus
max(--peer-reserve-mib, 2 GiB) display-peer headroom) under the explicit --peer-slots bound.
Peer counters (rows, mixed/all/zero windows, fallbacks) print at shutdown and the smoke ASSERTS
real peer work (725 rows computed; mixed and zero-peer windows exercised).  Injected launch
failures (a dedicated launch seam) fall back to the FULL primary service with oracle-identical
streams, no error latch, and the tier's exclusions cleared on close.  The lifecycle now includes
a real 8k-chunk-boundary request (the earlier "long" was a prefix-reusing follow-up, labeled as
such), asserts exact output counts with "length" finishes, and the suites are archived full-raw
(`ctest-exl3-full-raw.txt` 25/25; `ctest-gguf-native-full-raw.txt` 8/8 - GGUF compile/unit-level
suites, not a generic native-runtime claim).  MTP A/B re-verified on the final engine: streams
identical, acceptance 139/231.  Still correctness/lifecycle only - no performance runs.

### Peer tier sustained benchmark closure (2026-10-18): +11.5% warm / +16.6% MTP (padded strategy)

Preflight correctness first (review): warm-then-exclude now INVALIDATES the affected layer's
whole cache (the dense [0, filled) slot layout and the loan machinery stay consistent; enabling
the exclusion during an active loan is rejected; grow_scratch is complete-or-nothing) - focused
tests cover warmed-then-exclude, dense state and lend/restore.  The peer budget fit expression
is corrected (bytes <= budget - used; the old form accepted oversize entries at used==0) and the
budget keeps the configured reserve PLUS a coarse 256 MiB workspace allowance.

Measured (`bench/results/2026-10-18-hip-exl3-3bpw-peer-performance/`, matched single-GPU vs
peer, fast knobs identical, all token streams byte-identical, MTP acceptance 139/231 invariant;
turn-1 net tok/s):

| workload | single | peer8k chunked-8 | peer8k PADDED-1call | peer256 |
|---|---:|---:|---:|---:|
| warm 512 (2 pairs) | 25.49 | 27.78 (+8.9%) | **28.42 (+11.5%)** | 24.53 (-3.8%) |
| packed MTP 256 (2 pairs) | 31.10 | 29.98 (-3.6%) | **36.23 (+16.6%)** | 27.41 (-12.1%) |
| code prompt (1 pair) | 23.15 | 25.79 (+11.5%) | - | - |

Peer share is ~33% of routed rows in BOTH workloads (8192 slots = 14.21 GiB, filled in 5.2-6.4
s; 256 slots gives ~1% and loses), so share does not explain the sign flip: the chunked
complement issues ceil(complement/8) primary calls per layer-window (warm: 1 call, wins; MTP:
2-3 calls, loses), and the ONE targeted competing strategy - opt-in `STRATA_EXL3_PEER_PADDED=1`,
one (tokens,k) call per window with peer rows padded and overwritten by finish - removes the
extra calls and flips MTP to +16.6% (warm +11.5%) with identical outputs.  The per-call
staging-join/barrier cost is the mechanism.  A whole-window compact (c,k=1) call is UNSAFE: it
hangs in the staging join above 8 tokens (demonstrated; <=8 fine) - left open at the kernel
level; the default stays the bounded chunked form and PADDED is the RECOMMENDED default
candidate (not promoted without review).  VRAM (200 ms samples): primary peak 22.3/23.3 GiB of
24 (approximately full, no spill), peer peak 17.5 GiB (14.2 GiB tier + ~3.3 GiB runtime ->
~6.5 GiB free, above the ~2 GiB target).  Prompt-side (prompt_ms includes prefill +
restoration): unchanged within noise; the 8k-chunk-boundary pair matches (short 32-token decode
sample).  Owned scratch on the primary ~218 MiB (one wave).  Round-robin subset (no hot-set
profile yet); primary miss counters need the perturbing STRATA_EXL3_TIME and were not
collected.  CORRECTED record (closure confirmation pass): the padded column was ONE exploratory
run per workload vs the same-day control mean - the "2 pairs" claim applied to the chunked runs
only.  Fresh matched order-varied confirmation pairs give warm 25.56 -> 28.42 (+11.2%) and MTP
31.16 -> 36.20 (+16.2%), tokens identical (exact logs in `confirmation/`).  The analyze script's
gain formula was inverted in the first recording (fixed: peer/single-1; the table above was
computed correctly by hand).  Replicate counts: chunked 2 pairs/workload, padded 1 exploratory +
1 confirming pair, 256-slot sanity 1 pair, chunk8k/code prompts 1 pair each.  "No spill" is NOT
established: no VRAM exhaustion or slowdown was observed (primary peak 22.3/23.3 GiB of 24, peer
17.5 GiB) and GTT was sampled in the confirmation runs (peer peak 35 MiB of mapped/pinned
accounting - nonzero by construction, not a spill measure); spill remains unverified at 200 ms
sampling.  The ~3.3 GiB peer runtime overhead is inferred from peaks, not attributed; the 256
MiB budget allowance is coarse.  The compact (c,k=1) hazard is narrowed to the tried
whole-window configurations (an unsupported call shape), not a universal >8-token claim.  No
default promotions, GGUF unchanged; MTP/prefill optimization untouched (fresh executor's task).

### Matched MTP diagnosis (2026-10-18): +24.7% prose / +41.9% code at matched length; the row-dependent window service cost is the strongest measured association, draft quality NOT excluded

The peer numbers above are a DECODE-transport speedup, not an MTP algorithm uplift; the old
warm-512 vs MTP-256 ratio was an unmatched comparison.  This pass measures the MTP algorithm
itself, matched: same prose prompt (the bench.py essay prompt, 512 tokens) and the same code
prompt (the peer closure's merge-bug prompt, 134 tokens) in every arm, 512 generated tokens in
every arm, greedy `temperature=0 seed=7319`, two turns per session (turn 1 measured, warm page
cache and warm expert cache), cache 126 / context 65536 / prefill 8192 / kv int8, the six fast
kernel flags + `STRATA_EXL3_OVERLAP=1` + `STRATA_EXL3_NO_PREAD=1`, primary GPU0 only (HIP
gfx1100, binary `aadf1ce3` built at `aca8c372`; exl3/ple/spec/draft ctests 31/31).  Two
order-varied replicates per arm.  All 28 runs exited 0, no nonfinite/error latch, every stop
`length`, and all 512-token turn-1 outputs are BYTE-IDENTICAL across every arm and replicate
(the verify window makes speculation output-invariant under greedy sampling).

Turn-1 net tok/s, mean [min,max] of the two replicates (`nospec` = plain one-token decode: no
`--mtp`, `--suffix-draft 0`; `suffix` = the accepted warm config; depth N = `--mtp` with at most
N MTP drafts per window (`--spec N+1` + `--spec-min-p 0.5`; the accepted config is depth 3,
`--spec 4` + min-p 0.5, whose effective window is T<=6 with a 2-token suffix headroom):

| arm                 | prose            | code             |
|---------------------|------------------|------------------|
| nospec              | 25.49 [25.44,25.54] | 23.83 [23.81,23.86] |
| suffix only         | 25.51 [25.46,25.57] | 24.49 [24.38,24.60] |
| MTP depth 1         | 31.00 [30.97,31.04] | 30.60 [30.36,30.84] |
| MTP depth 2         | 31.99 [31.98,31.99] | 32.39 [32.35,32.44] |
| MTP depth 3 (accepted config) | 31.79 [31.77,31.80] | 33.83 [33.72,33.94] |
| MTP depth 4         | 30.83 [30.67,30.99] | 34.15 [33.98,34.31] |

Gains (on/off - 1, pair means): prose MTP/nospec **+24.7%**, MTP/suffix +24.6%, suffix/nospec
+0.1% (the suffix drafter is dormant on this essay: 0-1 windows/turn).  Code MTP/nospec
**+41.9%**, MTP/suffix +38.1%, suffix/nospec +2.7% (28 suffix windows, 49 of 95 drafts accepted
turn 1).  So the algorithm uplift is real and roughly 2x bigger on code than prose - but below
the +50%/+100% experience expectation, and the matched prose number (+24.7%) is only slightly
above the old unmatched ratio (+21.9%), so the mismatch was not the main gap.

What the prefix statistics say and what they do NOT say (corrected 2026-10-18, profile
milestone): the histogram below is end-to-end prefix SURVIVAL under the min-p 0.5 policy - it
conflates confidence truncation (min-p cuts the window: deeper drafts are never offered, never
tested) with rejection (offered and mismatched), and the depth-conditional rates condition on
both earlier acceptance and the window having been long enough to offer that depth.  It is NOT
a measurement of draft-model quality, and "draft quality is not the limiter" was an unsupported
claim: draft quality REMAINS a possible factor (the drafter's routed experts are Q2_0 and its
dense projections Q8_0; no higher-precision drafter has been tested).  Accepted-prefix
histogram over the MTP-drafted windows (the `STRATA_SPEC_DEPTH` counters; `P(a>=k)` =
ok[k-1] / windows-with-drafts):
prose depth 3 (191 windows) P(a>=1/2/3) = .749/.487/.262 with P(acc|depth) = .749/.795/.685;
prose depth 4 (184) = .761/.489/.255/.125 (.761/.789/.662/.590); code depth 3 (156) =
.891/.699/.564 (.891/.858/.854); code depth 4 (136) = .890/.721/.551/.463
(.890/.883/.833/.887).  Aggregate acceptance 286/448 (63.8%) prose and 343/444 (77.2%) code at
depth 3.  Per round (turn 1, depth 3): prose 1.98 proposed / 1.27 accepted / 2.27 committed
tokens per window; code 2.61 / 2.02 / 3.01.  Code's acceptance stays ~.85-.89 even at depth 3 -
deeper drafting pays there and the depth-4 arm is +0.9% faster than the accepted depth-3 cap;
prose decays from .69 at depth 2 and wants the shallower depth-2 cap (+0.6%).  The accepted
depth-3 cap sits within ~1% of both workloads' optima; these numbers do not justify a default
change on their own.

The strongest measured association is the verify window's cost growth with window width (causal
attribution INSIDE the service call is the profile milestone's job, not claimed here).
`STRATA_DECODE_TIMING` per-window phase intervals (INCLUSIVE - the sub-buckets are parts of
verify and are NOT summed with it; GPU work overlaps the host phases):

| arm (turn 1)        | ms/window | verify | GPU-reach wait | per-layer host | draft | commit |
|---------------------|-----------|--------|----------------|----------------|-------|--------|
| prose nospec (T=1.00, 1.00 tok/win) | 39.1 | 39.1 | 12.0 | 26.2 | 0.0 | 0.01 |
| prose depth 1 (T=1.86, 1.68)        | 54.4 | 52.7 | 12.6 | 39.1 | 1.7 | 0.01 |
| prose depth 2 (T=2.51, 2.07)        | 64.8 | 61.9 | 13.2 | 47.7 | 2.9 | 0.01 |
| prose depth 3 (T=2.98, 2.27)        | 71.3 | 67.5 | 13.7 | 52.8 | 3.8 | 0.01 |
| prose depth 4 (T=3.44, 2.39)        | 78.0 | 73.5 | 14.2 | 58.2 | 4.5 | 0.02 |
| code nospec (T=1.00, 1.00)          | 41.9 | 41.9 | 11.9 | 29.1 | 0.0 | 0.01 |
| code depth 3 (T=3.61, 3.01)         | 89.3 | 85.1 | 14.2 | 69.8 | 4.2 | 0.01 |
| code depth 4 (T=4.31, 3.44)         | 100.2 | 94.9 | 14.9 | 78.9 | 5.3 | 0.01 |

The per-layer host bucket is `ms_pool`, the INCLUSIVE elapsed span of the pool callback (the
external EXL3 expert service call) - host elapsed, NOT pure CPU work: it contains the GPU
service of the callback's own kernels as well as host route planning, expert preparation,
staging and worker joins, and its internal split was NOT measured here (claiming "CPU expert
compute for misses" for it was an inference, corrected here; the native pool counters
plan/actq/jobs/CPU read exactly 0.00 because that path bypasses them - zero native-wait CPU
counters do not show what is hidden).  MEASURED: this bucket grows ~13-15 ms per window per
added draft row (marginal over T=1 on prose: 15.0/14.2/13.4/13.1 ms/row at T=1.86/2.51/2.98/3.44),
while the drafting itself is cheap (~1.3-1.5 ms per draft) and GPU-reach wait nearly flat
(12 -> 14 ms).  At depth 3 prose the window costs 1.82x the plain window and yields 2.27x the
tokens; the marginal draft row costs ~half of the first row's bucket time.  HYPOTHESIS
arithmetic (not a measurement, and the "avoidable vs unavoidable" split is unmeasured): IF the
row-dependent part is avoidable repeated per-row work (decode/launch/transfer), halving it
would put prose depth 3 near +50-60% over nospec and removing it near +100%; IF it is
unavoidable per-row arithmetic, no such headroom exists.  The verification-profile milestone
exists to split these before any fix.

Peer-padded matched confirmation (1 fresh pair, prose depth 3 at 512 tokens, same instrumented
env): single 31.43 -> padded-1call 36.62 tok/s (**+16.5%**), turn-1 outputs byte-identical,
acceptance and histogram identical (286/448, .749/.487/.262) - consistent with the closure's
+16.2% pair at 256 tokens and with "the peer does not change drafting".  On the padded arm the
per-layer host bucket drops 52.8 -> 43.3 ms/window (the peer computes its routed-row share off
the primary), the same bucket the depth sweep implicates.  VRAM (200 ms samples): primary peak
23,208-23,332 MiB of 24,576 (~22.7 GiB of 24) in every arm (no exhaustion or slowdown observed;
GTT 25-79 MiB mapped/pinned accounting, nonzero by construction, not a spill measure - spill
remains UNVERIFIED); peer peak 17,977 MiB (~17.6 GiB) at 8192 slots.

Instrumentation overhead: the matrix runs carry `STRATA_SPEC_DEPTH` + `STRATA_DECODE_TIMING`
(both presence-gated, both pre-existing; no new instrumentation was needed - the histogram and
per-round counts derive from the existing counters).  One uninstrumented control run per key
arm (ran last in the matrix): nospec 25.06 vs 25.49 instrumented (-1.7%), depth-3 31.33 vs
31.79 (-1.4%).  The controls ran SLOWER, so the two flags' overhead is not detectable above the
~2% run-to-run spread; it is not resolved below that.  `STRATA_EXL3_TIME` was NOT set
(perturbing), so the host bucket's internals are not decomposed here - that is the proposed
next step.  Limits: one machine/model/binary; HIP/gfx1100 only (CUDA/SYCL untested); 2
replicates per arm and 1 pair per depth-sweep point; the uninstrumented controls ran last
(order/thermal drift possible); DraftPolicy adapts to wall-clock round times, so MTP-vs-suffix
window choices (and offered counts) shift between replicates while outputs do not; the depth
sweep caps MTP drafts but keeps suffix drafting and the 2-token suffix headroom in place; GPU
event/upload spans (not collected here) include host enqueue gaps and are not pure DMA; the M5
synthetic-graph bounds remain upper bounds, not callback-removal proof.  No default changed,
no peer adaptation, no capacity tuning, GGUF/native paths untouched.

Next step recommended for review (now authorized as the profile milestone; the original wording
is kept with one neutral fix - the sub-bucket names were themselves assumptions): minimal opt-in
counters INSIDE the external EXL3 expert host service - per layer-window: distinct routed
rows/experts, host-elapsed phase splits (route planning/preparation, read/stage worker join,
launch enqueue) against the GPU event spans, preallocated, nothing allocated in the callback,
default path byte-identical - then two matched runs to split the ~13 ms/row, and size one
targeted fix on the largest sub-bucket (row-batched host service is the candidate the numbers
point at, but the split decides).  A zero-code side probe is also cheap: depth 6 on code only (2 runs) to see
whether the flat acceptance tail continues past the tested depth 4.

### Verification-window service profile (2026-10-18): M=1..5 always runs the indexed GEMV; the row cost is staging + per-route work, and ~half the rows sit in multi-row expert groups

The diagnosis entry above left the service call a black box ("ms_pool is inclusive, internals
unmeasured").  This pass splits it with `STRATA_EXL3_PROF=1` (opt-in, host counters only: a fine
per-group row histogram, dispatch-arm route counts and host-elapsed phase spans; no GPU events
added, no allocation anywhere, default path unchanged) alongside `STRATA_EXL3_TIME=1` (its event
spans are REUSED, not duplicated) and `STRATA_DECODE_TIMING=1`.  Matrix: {prose, code} x
{nospec, MTP depth 3} x {primary, peer-padded}, 512 tokens every run, turns 2, greedy, cache 126
/ context 65536 / prefill 8192 / kv int8, the accepted fast flags; 8 instrumented runs + 3
UNINSTRUMENTED twins.  All 11 exit 0; all turn-1 outputs byte-identical across every run.
Window lines below are turn-1 decode windows; every span is labeled with its interval type.

**Architecture facts (code + counters agree):** the route plan is an expert-major counting sort -
one group per DISTINCT expert per layer call, group rows = that expert's routed rows this window
(top-K = 10 routes/token; a decode window at M tokens is 48 layers x M x 10 routes).  Groups are
therefore == distinct (layer, expert) pairs and route reuse = (M x 10)/groups.  The window
service = one run_multi call per layer (48 per window), all counters below per WINDOW.  Dispatch
thresholds: the group arm (gfx11 WMMA, else reconstruct+GEMM) needs `device_io && count >= 16`;
the expert-batch arm needs group rows >= 16 on the device-IO wave arm; `wave_arm`/`pipeline` are
device-IO-only.  Decode windows are host-IO - so **every decode route at every M runs
`exl3_moe_rows_indexed`, the SAME indexed GEMV M=1 uses** (empirical: arm routes indexed = 100%
of decode routes in every run, primary and peer; the group arm runs only in the prompt prefill,
e.g. M=505: group 179,992 vs indexed 62,408 routes).  The small-M kernel crossover the user
remembered is real and UNEXPLOITED: the indexed path gives each ROUTE its own descriptor entry
and decodes that expert's trellis per route - routes sharing an expert re-decode it (the
reference implementation extracts once per expert and reuses across match rows).

**Actual verifier M and per-expert rows.**  Sequential = M=1 (480 routes, 480 groups, reuse
1.00).  MTP depth 3 mixes M=1..4 (prose: 35/44/37/110 windows at M=1/2/3/4; code reaches M=6
via suffix extension: 11/14/9/136/1/3 at M=1..6).  Per-expert routed rows are NOT M: at M=4
(primary prose) the (layer,expert) groups are 1 row:183,928, 2:55,906, 3:23,684, 4:13,902 -
reuse 1.53, and 56.5% of ROWS sit in multi-row groups (M=2: 43.7%, M=3: 52.0%, M=5: 60.2%; code
similar, 52.5% at M=4, reuse 1.47).  On the PADDED peer arm the distribution is bimodal by
construction (M=4: 1-4-row complement groups PLUS 9-15/>=16-row PAD groups - every peer row
redirects to one dummy expert): reuse 2.32, groups 836 vs 1,261 primary.

**Window service, M-binned (per window = 48 layer calls; HOST ELAPSED = wall time in the phase,
not CPU; the GPU event spans include host enqueue gaps and are NOT pure DMA; pool/callback are
INCLUSIVE of their own GPU service - the columns overlap and are NOT summed across types):**

primary prose MTP (windows at M=1/2/3/4 = 35/44/37/110):

| M | routes | groups | reuse | multi-row rows | plan | prep | join | pool | upload span (MiB) | FFN span | hit% |
|---|--------|--------|-------|----------------|------|------|------|------|-------------------|----------|------|
| 1 | 480 | 480 | 1.00 | 0% | 0.09 | 0.86 | 9.6 | 26.2 | 11.0 (92) | 22.6 | 89.3 |
| 2 | 960 | 750 | 1.29 | 43.7% | 0.12 | 1.50 | 17.0 | 40.1 | 19.5 (172) | 35.6 | 87.1 |
| 3 | 1440 | 1,014 | 1.43 | 52.0% | 0.15 | 1.89 | 24.0 | 52.9 | 26.9 (243) | 47.5 | 86.5 |
| 4 | 1,920 | 1,261 | 1.53 | 56.5% | 0.18 | 2.79 | 38.0 | 70.8 | 41.8 (394) | 63.9 | 82.4 |

peer-padded prose MTP (same workload; M=1/2/3/4 = 35/44/37/110):

| M | routes | groups | reuse | plan | prep | join | pool | upload span (MiB) | FFN span | hit% |
|---|--------|--------|-------|------|------|------|------|-------------------|----------|------|
| 1 | 480 | 317 | 1.52 | 0.07 | 0.31 | 2.7 | 18.8 | 3.4 (24) | 15.7 | 95.7 |
| 2 | 960 | 499 | 1.94 | 0.08 | 0.52 | 4.6 | 28.2 | 6.1 (48) | 24.5 | 94.6 |
| 3 | 1,440 | 671 | 2.16 | 0.10 | 0.68 | 6.8 | 37.9 | 8.7 (66) | 33.6 | 94.5 |
| 4 | 1,920 | 836 | 2.32 | 0.11 | 1.07 | 12.4 | 49.3 | 15.0 (120) | 44.3 | 91.9 |

code MTP primary mirrors prose (M=4: pool 77.7, join 45.3, FFN 70.7, hit 79.3%, reuse 1.47);
nospec M=1 windows match the MTP runs' M=1 bins (pool 27.7 vs 26.2 - a within-arm control).

**Attribution of the row-dependent growth (primary prose, M=1 -> M=4: pool +44.6 ms/window):**
join (the staging workers' pread + H2D wait) +28.4 ms = 64%; residual (enqueue + the D2H/finish
wait + host scatter) +14.2 ms = 32%; prep (admission + prepare_expert) +1.9 ms = 4%; route plan
+0.09 ms ~ 0%.  The GPU spans grow in parallel and overlap the host phases (NOT additive): the
staging-stream upload span +30.8 ms (92 -> 394 MiB - the miss experts' rows; hit rate FALLS
89.3% -> 82.4% as more distinct experts are touched: 51 -> 227 misses/window) and the
compute-stream FFN span +41.3 ms.  The join and upload spans track each other (~10.5 ms per
added row): the row cost is mostly the extra DISTINCT experts' staging on the critical path -
already deduplicated per expert (uploads are per group) - plus per-route GPU work.  The
per-request verifier split (STRATA_DECODE_TIMING) cross-checks: ms_pool 55.5 ms/window at avg
T 2.98 == the M-weighted prof pool 55.0; GPU-reach host wait (ms_wait) stays 13.6-13.7 ms.

**Unavoidable vs avoidable (evidence-ranked, not measured in kernel):** (1) UNAVOIDABLE: each
route row needs its own FFN matvec arithmetic and its own output row.  (2) AVOIDABLE CANDIDATE -
per-route trellis decode: 52-60% of rows at M=2..5 are in multi-row expert groups, and the
indexed arm decodes each route's expert independently; a trellis-once-per-expert multirow kernel
(the reference's approach) would decode once per group.  The padded arm's >=16-row pad groups
run through the SAME per-route indexed path - the extreme case of the duplication.  How much of
the FFN span is trellis decode is NOT measured here (that is the micro-experiment's job).
(3) AVOIDABLE CANDIDATE - staging serialization: the miss staging sits on the critical path
(join 38 ms at M=4); the overlap arm only hides the HIT pass behind it, and wave pipelining
(`STRATA_EXL3_PIPE`) is device-IO-only - host-IO windows get none.  Launch host time is small
(plan+prep 3 ms at M=4) and not a lever.

**Peer spans (host elapsed per layer-call, NONADDITIVE as GPU time):** callback 0.92 ms =
peer launch 0.024 (enqueue only; the peer's GPU work then overlaps the primary service) +
primary service 0.824 + peer finish 0.060 (CONTAINS the wait for the peer's remaining GPU work
- the peer's tail is small here, i.e. the overlap works) + other 0.011 (classification/copies).
Per window (48 calls): launch 1.2 + primary 39.6 + finish 2.9 ms.  Primary-only callbacks:
0.59-1.48 ms/call (M-dependent), matching pool.

**Instrumentation impact** (turn-1 tok/s, instrumented vs its uninstrumented twin run
immediately after): prose MTP 30.64 vs 31.33 = -2.2%; code MTP 32.89 vs 33.06 = -0.5%; peer
prose MTP 36.37 vs 36.72 = -1.0%.  Mean -1.2% (bound: -0.5%..-2.2%, n=1 twin per arm; the
diagnosis's replicate spread was +-0.5%).  All profile numbers carry this bound; the uninstrumented
twins (31.33 / 33.06 / 36.72) also sit within ~1.5% of the diagnosis's throughput figures.
The `syncs` counter reads 0 on host-IO windows (it counts the device-IO `sync_compute` barriers;
the host-IO D2H+finish sync is inside the pool residual).

Limits: one machine/model/binary (HIP/gfx1100, 3bpw Qwen3.8-NEXT; CUDA/SYCL untested), 1
instrumented run per cell + 1 twin per key arm, 2 turns pooled shapes from turn 1 only, event
spans include enqueue gaps, no kernel-internal decomposition of the FFN span (trellis decode vs
matvec share unmeasured), no CUDA graph work, no default change, no capacity tuning, the peer
subset static.  Default behavior unchanged (counters behind `STRATA_EXL3_PROF=1`; 31/31 focused
tests pass on the instrumented build).

Smallest candidate experiment for review (NOT implemented): a GPU-only micro-benchmark of a
trellis-once-per-expert multirow FFN against the current per-route `exl3_moe_rows_indexed`, on
the MEASURED group shapes (M=1..5, rows 1-4 per expert, 3bpw geometry), to price the avoidable
per-route trellis decode share of the FFN span.  Second candidate (larger): extend host-IO wave
pipelining to the decode path so miss staging overlaps the window's compute as it does on the
device-IO arm.  Both are sized by this profile; neither is assumed to win.

### Final status against HIP_EL3_PLAN.md (2026-10-17)

- **M0 (HIP baseline)**: complete - build, 24/24 exl3/ple ctests, real-model CLI/serve/split/MTP
  smoke, GGUF regression reference.  CUDA-side compile of the touched .cu files not possible on this
  machine (no nvcc); SYCL unaffected (no EXL3 sources in `sycl/`).
- **M1 (baseline + bottleneck)**: complete - the ledger above, with STRATA_EXL3_TIME anatomy
  (warm TG ~2/3 miss servicing / ~1/3 FFN; cold PP dominated by compulsory misses).
- **M2 (routed MoE arithmetic)**: complete for gfx1100 - the matrix-core group FFN opt-in (A/B #2/#3,
  +21.5% PP8k; threshold 16 kept, A/B #4; gate+up launch pairing rejected, A/B #7; K=32 mma pairing
  promoted, A/B #8).  Dense trellis WMMA for prefill NOT tested (M1 ranked dense behind miss
  servicing and the routed FFN).
- **M3-lite (per-tile syncs)**: complete - the per-wave barrier opt-in (A/B #5, +2.2-2.4% PP8k).
- **M4 (cache policy)**: one measured negative (A/B #6, freeze-at-decode-start vs the plain LRU).
  Prompt-heat admission RANKING, budget-limited admission and the RAM-tier freeze variants are
  UNTESTED, not ruled out.  M4 transport is done: A/B #9 pipelines the prefill waves' uploads
  behind the current wave's FFN (+20.7% PP8k on 3bpw, `STRATA_EXL3_PIPE=1`).
- Explicitly untested / deferred, not silently skipped: GPU CSR route planning (M3 proper, the host
  plan was kept); in-graph resident decode (M5) and hipStreamWaitValue-type graph waits; the GPU1
  helper expert tier and any dual-GPU scheduling beyond the validated layer split (M6); native EXL3
  MTP weights vs the packed drafter (M7); cross-call wave pipelining and real-model RAM tier
  combined with `STRATA_EXL3_PIPE` (synthetic RAM and real-model layer split now checked in A/B #9
  review closure); anything on NVIDIA or non-gfx1100 AMD parts.
- Final validation on the promoted tree: exl3/ple ctests 24/24; GGUF needle serve smoke PASS
  (198.4 tok/s cold prefill, 34.1 decode); EXL3 single-GPU needle smoke PASS; EXL3 layer-split
  (24/24, both XTX) needle smoke PASS; EXL3+packed-MTP serve smoke all keys found both turns
  (35 tok/s warm).  Not performed: cold-page-cache series (needs root drop_caches), CUDA/SYCL
  builds (toolchains absent), multi-model or non-Qwen EXL3 geometries.

### Coverage and toolchain notes (2026-10-17)

- The M0 grid.y-limit question: covered. `exl3_gpu_parity` case `run(8, 4104, 2, 2)` reconstructs a
  128 × 65664 weight — just past the 65535 grid.y limit that the vocab dim must not sit in — and it
  passes on gfx1100 (part of the 24/24 exl3/ple subset).
- `exl3_moe_window_bench` group mode now also times the indexed GEMV arm and zeroes the whole token
  buffer (it previously zeroed only `routes` of `tokens` rows, reading uninitialised rows when
  `argv[5] > argv[1]`). At 8 rows K4 idle: indexed 0.164 vs wmma 0.147 vs gemm 0.389 ms/group —
  the isolated win that does not survive in the engine (A/B #4).
- Backends of the touched files: HIP built and tested (above). CUDA: no `nvcc`/CUDA toolkit on this
  machine — not buildable here; the touched dispatch code is plain host C++ using only headers the
  file already used, and the gate/mutex code sits inside `STRATA_WMMA_GFX11` (HIP-gfx11-only), so
  the CUDA path compiles the same stub as before. SYCL: needs `icpx` (not installed), and the
  `sycl/` tree carries no EXL3 sources, so these files are not in that backend at all.

```text
Date / Strata SHA / candidate commit / ROCm version / GPU(s) & PCIe
Model + format + bitrate/K/codebook / PLE / MTP + depth / KV / context
Command or script + prompt/workload / PP or TG / cache residency (cold/warm)
A median [spread] / B median [spread] / VRAM / key counters / verdict
Reason for winner or inconclusive outcome; known correctness caveat
```

## Small-M verification-window kernel comparison: grouped trellis-reuse prototype vs the indexed GEMV (2026-10-18)

**SUPERSEDED by the review-fix rerun section directly below (same date): keep this
section as history, but four of its claims were wrong and are corrected there.**  (1) Its mix-M2
and mix-M3 fixtures contained IMPOSSIBLE groups (3 rows at M=2, 4 rows at M=3 - a token never
repeats an expert under top-10 routing, so no group can exceed M rows); the valid fixtures flip
mix-M2 from a win to a LOSS and move the mixed-distribution crossover from M2 to M3.  (2) Its
"byte-identical on every scenario and trial" was measured on 2 trials per scenario, not the 30
the timing used; the rerun checks all 30.  (3) Its "chain is the closer model of the engine's
pipelined window cadence" claim is withdrawn: the 48-call batch is a queue diagnostic only, and
per-call host wall through the sync is the primary cost.  (4) Its "register tax" explanation of
the fixed-width-6 M1 cost is withdrawn: the code object's AMDGPU.metadata shows the width-1
instantiation at the indexed kernel's exact register budget and no spills.

The candidate experiment proposed at the end of the verification-window service profile, now run.
Question: the M=1..6 decode/verification windows route top-10 experts per token and different tokens
share experts (measured prose M4: 40 routes over ~26.3 distinct experts per layer-window, reuse
1.53); the indexed arm (`exl3_moe_rows_indexed`, the path every such route uses today) launches one
GEMV block per ROUTE and decodes a shared expert's trellis once per route. Does an expert-GROUP
kernel that decodes the trellis once and applies it to every row of the group win, with the
per-call route-plan and descriptor uploads included, on resident weights? This replaces the
profile entry's "crossover unmeasured" with numbers; it does not claim the engine gets them.

What was built (all additive, no engine caller, default paths untouched - the indexed entry points
and their kernels are not modified):

- `exl3_moe_rows_grouped` prototype in `src/kernels/cuda/exl3.cu` (+`reserve_small_m`): the group
  is the blockIdx.z unit of the GEMV, prescale and postscale also run per group (the shared
  expert's suh/svh load once), everything else (silu, buffers, output layout, route-major rows)
  is the indexed arm's. The float arithmetic is NOT changed - unlike the ggml reference kernel
  this idea comes from (`exl3_mul1_rows_body`), there is no activation prequantization and no
  DP4A; per row the k-chunk boundaries, FMA order, two-level reduction order, parts order and
  Hadamards match the indexed kernels exactly, so every row's output is byte-identical.
- The GEMV is instantiated per group width 1..6 and the entry picks the window's widest group
  (host-validated). This mattered: a fixed 6-wide build measured M1 at +25% gpu_span - the
  24-extra accumulator registers are a real occupancy tax. With width specialization the 1-row
  body has the indexed arm's register budget; the +25% figure is kept here as the measured
  evidence for that choice.
- `strata-exl3-smallm` diagnostic (`src/kernels/exl3_small_m_bench.cpp`): 192 real 3bpw experts
  (layers 0/12/24/36 x 48, 341 MiB) resident on the primary; per window M=1..6 a homogeneous
  case (every expert takes all M rows - the reuse upper bound) and a mixed case whose
  rows-per-group histogram matches the measured prose profile (M4: 17/5/3/1 groups of 1/2/3/4
  rows, reuse 1.54; M2/M3/M5 matched to the measured 44/52/60% rows-in-multigroup fractions;
  M6 extrapolated, no measurement). 30 interleaved order-varied trials per arm and scenario.

Cost boundaries (printed on every line of the raw log): "gpu_span" is the event interval around
the whole enqueued per-call work - the input_rows upload, the API's own descriptor/plan uploads and
all kernels - with a stream sync per trial; "chain" is 48 back-to-back calls (12 windows x 4
layers) synced once, per call - the closer model of the engine's pipelined window cadence, and it
hides the queue-idle gaps that an isolated small call's pageable H2D staging leaves inside
"gpu_span". "issue" is host wall until the call returns; "e2e" is host wall through the sync.
The weight upload (341 MiB, 226 ms) is setup, outside every timed region; the host route-plan
build measured separately at 0.1 us/window for BOTH arms (40 routes; the counting-sort plan is
shared infrastructure, the delta is 3xgroups vs 3xroutes descriptor copies) - noise-level, not a
factor. Hardware: RX 7900 XTX (0000:03:00.0, primary only), ROCm 7.14, HIPRelease gfx1100.
Artifacts: `bench/results/2026-10-18-hip-exl3-3bpw-small-m-kernel/` in the audit tree.

Correctness first (this is the "same EXL3 math" gate): arm B is byte-identical to arm A on every
scenario and trial (memcmp over all routes x 2560 fp16 outputs), outputs finite and non-constant,
and an independent CPU oracle (trellis reconstructed on the CPU from the same mmap, projection in
double, the same fp16 rounding points) gives rel 7.3e-4 (homo-M1), 6.6e-4 (mix-M4), 7.7e-4
(homo-M6) for both arms - same numbers, because they are the same bytes. No tolerance was relaxed
anywhere; the initial FAIL (rel 1.79 on mix-M4) was a bug in the harness's oracle probe (wrong
group picked for a mid-group route), not in either kernel, and was fixed before any timing was
recorded.

Medians [iqr] per call, indexed -> grouped, gpu_span / chain:

| scenario | routes/groups | indexed ms | grouped ms | span delta | chain delta |
|---|---|---|---|---|---|
| homo-M1 (control) | 10/10 | 0.302 | 0.328 | +8.5% | +4.5% |
| mix-M1 (control) | 10/10 | 0.268 | 0.286 | +7.0% | +3.7% |
| homo-M2 | 20/10 | 0.397 | 0.343 | -13.5% | -15.2% |
| mix-M2 | 20/15 | 0.421 | 0.390 | -7.4% | -8.5% |
| homo-M3 | 30/10 | 0.502 | 0.298 | -40.6% | -44.2% |
| mix-M3 | 30/20 | 0.529 | 0.458 | -13.6% | -14.3% |
| homo-M4 | 40/10 | 0.638 | 0.328 | -48.6% | -52.7% |
| mix-M4 | 40/26 | 0.759 | 0.665 | -12.4% | -27.2% |
| homo-M5 | 50/10 | 0.740 | 0.378 | -48.9% | -55.4% |
| mix-M5 | 50/30 | 0.820 | 0.723 | -11.8% | -19.0% |
| homo-M6 | 60/10 | 0.958 | 0.481 | -49.9% | -55.9% |
| mix-M6 | 60/36 | 0.978 | 0.867 | -11.4% | -19.5% |

Reading, with limits stated plainly:

- The crossover is REAL on this hardware at M>=2: even at the measured mixed distributions
  (mostly singletons, reuse only 1.33-1.67) the grouped arm wins 7-14% per-call gpu_span and
  8.5-27% chained. The homogeneous upper bound (reuse = M) wins ~50%. The win is much smaller than
  the naive traffic model (1 - 1/reuse would predict ~-40% at reuse 1.67), which says most of the
  indexed arm's re-reads were already L2 (Infinity Cache) hits at this 26-36-group working set -
  the resident regime this bench isolates.
- The M=1 control regresses +4..8%: the 1-row instantiation still pays the per-call group-table
  uploads and the per-block group indirection. An engine integration must therefore stay on the
  indexed arm for M=1 windows (or pay this), and this bench does NOT claim anything about which
  one the engine should do - dispatch is engine work, out of scope here.
- Per-call cost structure: arm B's issue time is ~1.5-2 us higher (two extra tiny plan uploads),
  arm A uploads 3xroutes descriptors vs arm B's 3xgroups + 2 int tables - both are noise against
  270-1000 us spans at this scale. The dominant per-call work is the kernels.
- What this does NOT show: any end-to-end MTP gain. The engine's window service overlaps staging
  waits, miss compute and the wave pipeline with these kernels; the profile's ffn span includes
  hit compute + staging wait + miss compute, and a -12% resident-kernel win at M4 does not
  translate one-to-one. Whether the resident hit arm (or the peer PADDED arm, where the
  artificial repeated pad expert makes reuse HIGHER than natural) should adopt the grouped kernel
  is the follow-up decision; the numbers above price only the kernel + its per-call plan cost.
- Backends: HIP/gfx1100 built and measured. CUDA and SYCL toolchains are absent on this machine;
  the changes are additive C++/template code in a file both backends compile, plus one
  HIP-only compat-header define, but neither backend was built here.

Validation on this change: focused exl3/ple ctests on the HIP build - with HIP_VISIBLE_DEVICES=0,
22 of the 24 tests in the `-R "exl3|ple_parity|ple_exl3" -E "loader_validation|tokenizer_validation"`
filter pass (exl3_parity k1-k8 with their fixtures, exl3_gpu_parity, exl3_gemv_parity,
exl3_moe_multi_parity, exl3_route_plan_test, ple_exl3_parity, ple_reader_selftest,
native_dense_ple_key_test); exl3_peer_test SKIPS without the peer visible and PASSES with both
GPUs visible (run separately); ple_parity fails at fixture open - its Q2_0 GGUF
(`../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`) does not exist anywhere on
this machine, so the test exits before any compute; this change touches no PLE code. No
default-path change exists in this commit set, so those results double as the no-regression
check; the new code paths are behind the prototype entry that nothing in the engine calls.

## Small-M kernel comparison, review-fix rerun: valid fixtures, end-of-row barriers, honest cost definitions (2026-10-18)

Bounded fix-and-rerun of the section above, after review of bd07e2b4/306efc93.  Same prototype
(`exl3_moe_rows_grouped`, still additive, still no engine caller, default paths still untouched),
same hardware (primary RX 7900 XTX 0000:03:00.0, ROCm 7.14, HIPRelease gfx1100), same 192 real
3bpw experts resident (341 MiB, layers 0/12/24/36 x 48).  What changed, and what it did to the
numbers:

**Correctness fixes (all verified by rerun):**

- End-of-row `__syncthreads()` added to `prescale_grouped_kernel` and `postscale_grouped_kernel`:
  both reuse `__shared__ b[]` across the group's rows, and the trailing global store reads `b[t]`
  after the Hadamard's last barrier, leaving the next row's write formally racy against a slower
  warp's store-operand read.  One barrier per row (rows <= 6) closes it; the GEMV's reduce loop
  already had the end-of-pass barrier.  Byte-identity was re-verified on ALL 30 trials of every
  scenario after the change.
- Every window is now a VALID token-major top-10 assignment: each token picks 10 DISTINCT
  experts (largest-remaining-demand construction), the groups are derived with the engine's own
  `Exl3RoutePlan` counting sort, and the harness asserts 10 routes per token, no repeated expert
  within a token, group rows <= M and <= 6, distinct token rows per group, contiguous coverage,
  and the achieved size multiset equal to the target.  The impossible fixtures of the first run
  are gone.
- Parity scope: byte-identity on ALL 30 trials per scenario (was 2).  The independent CPU oracle
  (trellis reconstructed on CPU, double projection) still covers EXACTLY 2 probes on EXACTLY 3
  scenarios - homo-M1, mix-M4, homo-M6 - rel 6.6e-4 / 8.5e-4 / 7.3e-4; that is the whole oracle
  coverage, stated on every log line.
- Negative precheck: 7 malformed-plan variants (first-offset, non-contiguous, not-covering,
  zero-row, width-exceeding, mixed-geometry, null table) each THROW before launch, leave the
  stream EMPTY after every rejection (`cudaStreamQuery` clean), and the workspace still produces
  byte-identical output on the valid plan afterwards.  PASS every run.
- Codebook coverage: synthetic probes at the real 2560/640/2560 geometry for K4/mul1, K3/mcg and
  K3/3inst are byte-identical vs the indexed arm with finite outputs (the indexed arm itself is
  CPU-validated for those codebooks elsewhere).  The real-weight scenarios remain K3/mul1 only.

**Cost definitions (corrected):** per-call host wall through the sync ("e2e") is the PRIMARY
reported resident-API cost; "gpu_span" is labeled as including queue-idle gaps from pageable
enqueue staging (not pure kernel time); the 48-call back-to-back batch is now arm-order
alternated AB/BA (6 batches per arm) and reported as `batch48_ms` - a QUEUE DIAGNOSTIC ONLY, not
engine cadence (the engine consumes each layer's activations before the next window), with no
end-to-end inference drawn from it and no preference over e2e.  The host route-plan build was
re-measured with a full-array checksum the compiler cannot elide: indexed 0.65 us vs grouped
0.57 us per M4 window, plan included in both - still noise against 0.3-1.0 ms calls.

**Compiler register evidence (vgpr-counts.txt in the rerun artifact dir, parsed from the code
object's AMDGPU.metadata):** `gemv_batched_kernel<4,true>` sgpr=38 vgpr=90;
`gemv_grouped_kernel<4,1,true>` sgpr=38 vgpr=90 - the IDENTICAL budget; widths 2..6 sit at
sgpr 45..52, vgpr 89..102, zero spills everywhere, and on gfx1100 (16384 VGPRs per SIMD, wave64)
both 90 and 102 VGPRs mean the same 2 waves per SIMD.  The first run's "register/occupancy tax"
explanation of the fixed-width-6 M1 cost is therefore WITHDRAWN: that +25% was inner-loop
instruction overhead (guards, per-row addressing), which the constexpr single-row path and the
hoisted row pointer removed; the residual M1 cost is the per-call group-table uploads and
per-block group indirection.

**Corrected medians [iqr], per-call e2e primary, indexed -> grouped:**

| scenario | routes/groups (reuse) | indexed e2e ms | grouped e2e ms | e2e delta | gpu_span delta |
|---|---|---|---|---|---|
| homo-M1 (control) | 10/10 (1.00) | 0.295 | 0.307 | +4.0% | +5.2% |
| mix-M1 (control) | 10/10 (1.00) | 0.289 | 0.309 | +7.0% | +7.3% |
| homo-M2 | 20/10 (2.00) | 0.424 | 0.364 | -14.1% | -15.7% |
| mix-M2 (40/50% arrangements, reuse 1.25) | 20/16 | 0.433 | 0.472 | **+8.9% LOSS** | +9.3% |
| homo-M3 | 30/10 (3.00) | 0.518 | 0.309 | -40.4% | -41.7% |
| mix-M3 (53/50% arrangements, reuse 1.43) | 30/21 | 0.550 | 0.498 | -9.4% | -9.8% |
| homo-M4 | 40/10 (4.00) | 0.654 | 0.357 | -45.4% | -49.0% |
| mix-M4 (measured histogram, reuse 1.54) | 40/26 | 0.771 | 0.681 | -11.6% | -12.9% |
| homo-M5 | 50/10 (5.00) | 0.791 | 0.409 | -48.3% | -49.4% |
| mix-M5 (60%, reuse 1.67) | 50/30 | 0.842 | 0.749 | -11.0% | -11.1% |
| homo-M6 (extrapolated) | 60/10 (6.00) | 0.987 | 0.522 | -47.1% | -48.4% |
| mix-M6 (extrapolated, reuse 1.67) | 60/36 | 1.030 | 0.920 | -10.7% | -10.9% |

**What the corrections changed in the conclusion:** the crossover on realistic (valid) mixed
distributions now starts at M3, not M2 - mix-M2 (12 singles + 4 doubles alternating with 10 + 5;
75% of its groups are singletons) LOSES 8.9% per-call e2e, where the impossible fixture had shown
a 7.4% win; the grouped arm's fixed indirection overhead dominates when nearly all groups hold
one row.  M1 (the no-MTP decode window) still loses +4..7%.  At M3..M6 the mixed wins hold
(-9 to -12% e2e; -14% at the measured M4 histogram) and the homogeneous upper bounds (reuse = M)
stay around -14% (M2) to -48% (M4-M6).  No end-to-end engine gain is claimed from any of this:
the engine's window service overlaps staging waits, miss compute and wave pipelining with these
kernels, staging is untouched by this experiment, and M1/M2 windows would have to stay on the
indexed arm - a per-window-shape dispatch decision that is engine integration work, not done
here.

The smaller-than-traffic-model mixed wins remain UNEXPLAINED by direct measurement: an L2 /
Infinity-Cache share of the indexed arm's re-reads is a plausible hypothesis (no cache counters
were collected), not a result.

**Validation on the final build (exact command):** `ctest -R "exl3|ple_parity|ple_exl3" -E
"loader_validation|tokenizer_validation" --output-on-failure` in build-hip-exl3 with
HIP_VISIBLE_DEVICES=0: 22 of 24 pass, `exl3_peer_test` skips without the peer visible and passes
with both GPUs visible, and `ple_parity` fails at fixture open because its Q2_0 GGUF
(`../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`) is absent from this machine -
the test exits before any compute and this change touches no PLE code; it is NOT counted as a
pass above.  Backends: HIP/gfx1100 built and measured; CUDA and SYCL toolchains are absent on
this machine, so neither backend was built (the changed code is additive standard C++/templates
plus one HIP-only compat-header define).

Artifacts: `bench/results/2026-10-18-hip-exl3-3bpw-small-m-kernel-rerun/` in the audit tree
(raw 30-trial log, notes-attempts.md with the intermediate states, vgpr-counts.txt, config,
implementation diff, run script, ctests log); the predecessor artifact dir is preserved untouched.

## Small-M primary integration: STRATA_EXL3_SMALL_M in the host-IO expert service (2026-10-19, primary-only milestone)

The bounded opt-in engine integration of the grouped small-M FFN, per the review-approved plan:
`exl3_moe_rows_grouped` replaces the indexed GEMV inside `run_multi_impl`'s HOST-IO arms only -
both passes of a split (STRATA_EXL3_OVERLAP) wave and the default arm's whole-group tiles - on
windows whose shape measured a per-call win: window tokens >= 3, every group of the pass at most
6 rows (the prototype's compiled width), and the pass's route reuse >= 1.4.  M1/M2-shaped
windows, wider groups and cut tiles stay on the indexed GEMV; the device/prefill arms, every
no-flag default and the peer tier are untouched; an active ownership-elsewhere tier (peer or
remote helper -> `own_table` set) bypasses the new path for the whole primary service, preserving
those baselines until the peer milestone decides otherwise.  One transparent threshold, no
heuristic grid; every stay is counted (window<3 / group>6-or-cut / reuse<1.4) and printed under
STRATA_EXL3_TIME, and STRATA_EXL3_PROF gained a `grouped-smallm` arm-route counter so coverage
is observed, not assumed.

**The one real bug this milestone found, and its fix (kept as the load-bearing constraint):**
the first version reserved the grouped arm's device buffers per call at `run_multi` entry.
Both code-prompt flag-ON runs STALLED DETERMINISTICALLY at the first M=4 verify window (3
committed tokens, the engine's 60 s no-progress watchdog stopped it; "the GPU rang 1; flags:
plan (A) 0").  Root cause via a gated breadcrumb trace (STRATA_EXL3_SMALL_M_TRACE, kept as a
diagnostic knob): the host thread sat inside `reserve_small_m`'s `cudaMalloc`.  The verify
window's host service runs WHILE the captured GPU graph spins on mapped host flags; a device
allocation whose internal path waits for GPU quiescence deadlocks against an endless spin -
the allocator waits on the GPU, the GPU spins on host flags, the flags are only published
after run_multi returns.  Prose survived because its post-prefill heap state served the same
allocation without the blocking path - prompt-dependent, not random.  Fix: the flag's ENTIRE
device budget is pre-reserved at Exl3ExpertStream construction (setup, before any window or
graph exists) for the envelope of every graph-coexisting window - 8 verify tokens x k<=max_k
routes, at the layer range's WIDEST FFN so mixed-width models grow nothing later either; the
per-call reserve remains only for prompt-scale host-IO calls that exceed the envelope and can
never run under the verify graph.  The first benchmark round (pre-fix) is preserved verbatim in
`bench/results/2026-10-18-hip-exl3-3bpw-small-m-primary-integration/failed-attempt-1-stall/`.

**Correctness validation (all on the fixed build):**
- `strata-exl3-smallm-seam` (new, wired into the exl3_loader_validation ctest via
  tools/test_exl3_loader.py): four flag arms (OFF / SMALL_M / OVERLAP / both) over 19 windows -
  cold all-miss, warm repeats, a primed guaranteed all-hit window, mixed thrash, M1/M2
  stay-indexed regression, M3..M6 grouped mixes (valid token-major top-10 assignments), an
  M=7 seven-row-group fallback, duplicate ids inside one token, zero routes, both FFN
  geometries of the two-layer fixture, repeated windows - all BYTE-IDENTICAL across the four
  arms, with dispatch-coverage assertions (58 grouped calls / 456 routes on the SMALL_M arm,
  63 on SMALL_M+OVERLAP, all three stay reasons fired, flag-off arms never dispatch) and a
  no-workspace-growth-on-repeated-shapes check (the pre-reserve contract).
- `exl3_wave_check` - including its full STREAM LOAN battery (lend, borrower overwrite,
  restore, run/run_multi inside loans) and the OVERLAP/PIPE arms - run with STRATA_EXL3_SMALL_M=1
  leaked into the environment: PASS, all arms byte-identical.
- Focused suite on the fixed HIP build: 34/35 (`ctest -R "exl3|ple_parity|ple_exl3|spec|draft|
  verify_batch|suffix|route_resident|resident_plan"`, primary visible; `exl3_peer_test` skips
  primary-only and PASSES with both GPUs; `ple_parity` remains fixture-blocked by the absent
  Q2_0 GGUF - pre-existing, no PLE code touched, not counted as a pass).  CUDA/SYCL toolchains
  absent on this machine: only HIP/gfx1100 was built.
- End-to-end: all four uninstrumented OFF/ON pairs TOKEN-IDENTICAL with identical acceptance
  and stop fields; the no-spec M1 control token-identical; a mid-generation SIGTERM (563
  tokens committed, exit 124, zero stall lines) followed by a fresh valid 128-token session in
  a NEW PROCESS - this is a kill-and-restart check, NOT a same-process STOP/cancel protocol
  test, and is labeled as such; the borrow smoke ran with STRATA_EXL3_PREFILL_BORROW=1 and the
  flag on, but the loan itself did not engage (the engine logs "too few cache slots to borrow"
  at cache 126, an eligibility condition, not a skip we introduced) - the flag-on loan
  interplay is instead covered by exl3_wave_check's loan battery run with the flag leaked on.

**End-to-end throughput (matched fast config, primary 7900 XTX only, 3bpw + packed MTP,
512 new tokens, temperature=0, turns 2, order-varied pairs):**

| workload | OFF tok/s | ON tok/s | deltas (both pairs) | verdict |
|---|---|---|---|---|
| prose, MTP spec4 (512-token prompt) | 31.89 / 31.94 | 32.16 / 31.81 | +0.8% / -0.4% | NEUTRAL within spread |
| code, MTP spec4 (134-token prompt) | 33.80 / 33.85 | 33.76 / 33.81 | -0.1% / -0.1% | NEUTRAL, very consistent |
| prose, no-spec (M1 windows) | 25.54 | 25.56 | +0.08% | NEUTRAL (never dispatches, as designed) |

**Measured dispatch coverage (STRATA_EXL3_PROF, prose MTP, flag ON, INSTRUMENTED run):** the
session's verify windows were 70 M=1, 88 M=2, 74 M=3, 220 M=4 and 2 M=6 (M=6 exists because
spec 4 plus the suffix headroom admits it; it is rare), plus one 505-token prompt window.
M=6 windows carried 97.4% of their routes grouped (2806/2880, 90 calls) and M=4 windows ~70%
(1356/1920 and 1352/1915, 64+43 calls); M=1, M=2 and the prompt window carried 0%.  The prof
lines are per-window aggregates over both turns and include the prompt window - they are not
turn- or segment-labeled, and the run carried the full instrumentation (TIME + PROF +
DECODE_TIMING), so its throughput is not compared with the uninstrumented pairs.

**Why the -9..-12% per-call FFN-span win does not surface - a HYPOTHESIS, not a measurement:**
the neutral net TG is measured; the explanation is not.  The instrumented run's join counters
range from 0.4 ms (all-hit windows) to 3.2 s (the prompt window) with a 25 ms median across all
455 window lines - the 107-139 ms figures first quoted here were the coldest turn-0 windows of
that instrumented run, not the warm steady state, and warm M=4 joins sit at 17-40 ms against
6-8 ms of admission/preparation.  Staging-dominated window service is therefore a plausible
reading (the grouped path speeds up a slice the OVERLAP arm already partially hides), but it is
NOT established by this record: the profiler perturbs, the lines are not segment-labeled, and
no unperturbed attribution was collected.  No end-to-end gain is claimed and no cause is
asserted.  The opt-in stays exactly that: OFF by default, no promotion, no config change; its
value on this workload is the validated dispatch machinery, the parity evidence and the seam
tests, ready for the peer milestone's decision (where the PADDED arm's reuse is higher and the
windows are device-resident - a different regime this milestone deliberately did not touch).

Artifacts: `bench/results/2026-10-18-hip-exl3-3bpw-small-m-primary-integration/` in the audit
tree (per-run configs/env/build hashes/commands, pair-comparison.txt, prof logs,
wave-check-with-smallm.log, ctests, failed-attempt-1-stall/ with the full stall forensics).
The review-correction round (flag-off baseline fix, one matched prose/code pair each, docs
corrections) lives in
`bench/results/2026-10-19-hip-exl3-3bpw-small-m-primary-integration-correction/`.
