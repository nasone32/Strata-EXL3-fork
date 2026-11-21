# exl3-2026.11.21 — Qwen EXL3 resident-peer release

Tag: **`exl3-2026.11.21`**. This is the fork's EXL3 release line; it
does not bump or replace an upstream Strata `v0.1.x` version.

Scope: Qwen3.8-Flash-Next EXL3 on Linux/HIP, measured on gfx1100 (two RX 7900
XTX, Ryzen 9 7950X, 96 GB RAM, ROCm core-7.14). Everything below was measured
on that machine; nothing in this release was validated on CUDA, SYCL, Windows,
Metal or RDNA4.

## What changed

### Native serial prefill is now a gfx1100 engine default

The optimized native serial prefill kernels (`STRATA_EXL3_PF_SMALL_WMMA_SILU`,
`STRATA_EXL3_PEER_SMALL_BATCH`, `STRATA_EXL3_PEER_DOWN_ACCUMULATE`,
`STRATA_EXL3_PREFILL_INDEXED_BATCH`, `STRATA_GDN_CONVL2`, `STRATA_GDN_HEAD`,
`STRATA_GDN_PP`, `STRATA_HC_NO_FLOAT_ATTN`, `STRATA_PF_QSA_SPLIT_NORM`) are
selected by the engine itself on gfx1100, at the standard 8K chunk as well. No
shell recipe is needed. An explicit environment value, including `0`, still
overrides a default. Other backends keep their existing dispatch.

Validated with the standard 8K smoke profile (`host-8k-Gk6JYQ`, host
collection, P2P and peer loan off): both the `all` and `peer` residency arms
exited cleanly and emitted identical token IDs. Fresh 10,000-token prefill
1,310.7 / 1,291.0 tok/s; decode 65.08 / 54.42 tok/s (one 96-token generation
per arm). No throughput parity claim between the arms.

### Qwen `--resident peer` (peer-complement residency)

Peer-owned experts now live only on the peer GPU, not in pinned RAM. The mode
requires the serial server, peer prefill and native unrepacked experts, and
refuses batching, repacking and CPU execution of peer-owned experts. Host spans
for peer-owned experts are inaccessible, so the engine refuses a prompt
path that would try to read those weights from the host.

In this mode the peer side of the prefill loan is **off**: an explicit peer
loan request is ignored, and the launcher forces `STRATA_EXL3_PEER_PREFILL_BORROW=0`.

Registered expert RAM with 7,200 peer experts at 4.05 bpw: 40.10 GiB instead
of 56.72 GiB; at 3.05 bpw: 30.16 instead of 42.66 GiB. These are arena sizes,
not whole-process requirements, and this mode was **not validated on a 64 GB
machine**. Lowering the peer cap increases pinned RAM use. With 11,000 peer
experts at 3.05 bpw (`3bpw-250k-peer11000-tqhgLk`, 250K capacity, 8K chunks,
static placement, MTP): READY at 250K, fresh PP32K 1,556.88 tok/s, 96 generated
tokens at 65.24 tok/s, no peer fallbacks; automatic primary sizing chose 96
slots/layer; minimum sampled free VRAM 4,198.7 MiB primary / 112.8 MiB on the
display peer.

### Main borrow: loan-aware main sizing and default peer loan

With full residency (`--resident all`), eligible prefill loans from both the
primary and peer caches are now automatic by default; `STRATA_EXL3_PEER_PREFILL_BORROW=0`
opts out. Main automatic sizing accounts for the ranks the prompt will borrow:
in `default-peer-loan-m9Iov8` (3.05 bpw, reserves 512/2048 MiB) it chose 143
instead of 105 slots/layer, the prompt used 39 borrowed ranks, and all 268
lent experts were restored before decode with zero fallbacks (PP16K
1,507.78 tok/s, decode 74.76 tok/s). The same run's 4.05 bpw `peer` arm used a
30-rank primary loan. An explicit fixed primary cache count is not resized.

Reserves are sizing margins, not runtime floors: with 512 MiB the primary
reached 85 MiB minimum sampled free VRAM during the request in that run.

### Improved automatic sizing with retries

The peer tier checks VRAM during expert upload and after prompt-workspace
setup, and retries a smaller ranked subset before publishing residency when
the allocations do not fit. `--peer-slots` left unset means "fill the byte
budget". In `auto-reserve-ab-QfW35p` (3.05 bpw, `resident all`, 250K capacity)
both reserve arms retried twice and then completed without fallbacks; in
`display-main-autosizing-OXXf5J` the 4.05 bpw `peer` arm reached READY after
four retries with 7,278 peer experts (39.92 GiB registered). Check the engine
log for the final counts and any retries.

### Optional: 16K prefill chunks

`--qwen-16k-peer` (native 4.05 bpw on two RX 7900 XTX with enough free VRAM)
selects 16,384-token chunks, defaults to a 7,200-expert peer cap and sets a
400-expert peer loan target where borrowing is enabled. Off by default.
Validation is incomplete: in `host-validation-yTTdV7` the `all` arm completed
a 20,000-token prompt at 16K chunks (52.672 s prefill, 96 tokens generated,
400 experts restored, zero fallbacks), while the `peer` arm stalled at layer 6
of the first chunk and the session was interrupted; the logs do not establish
the cause, and the 16K path needs a clean repetition. Keep 8K chunks unless
you specifically need 16K, and prefer `--resident all` with it.

### Optional: fused P2P decode output

`STRATA_EXL3_PEER_DEVICE_OUT=2` with `STRATA_EXL3_PEER_OUTPUT_FUSED=1` (and
`--cpu off`) sends peer decode results directly to the primary with a fused
wait/scatter. It changes decode output transport only (activations to the peer
still travel through the host) and needs working GPU peer access. Fused P2P
did not consistently beat host collection on the tested pair, so it is not a
default; the demo launchers export `STRATA_EXL3_PEER_DEVICE_OUT=0`.

### Portable demo launchers

New `examples/` directory (nothing machine-specific, no binaries):

- `examples/qwen-exl3-3bpw-resident-all.sh` — 3.05 bpw, `--resident all`, automatic sizing on both cards.
- `examples/qwen-exl3-4bpw-resident-peer.sh` — 4.05 bpw, `--resident peer`, automatic sizing on both cards.
- `examples/README.md` — inputs, GPU/reserve choices, optional flags, troubleshooting.

Model, tokenizer and MTP runtime are passed through environment variables or
flags; the engine binary defaults to the local build (`STRATA_EXL3_ENGINE`
overrides). `--dry-run` resolves and validates the configuration without a
built engine. The launcher also gained `--engine-log FILE` for engine
diagnostics.

### Docs

- README: `-DCMAKE_PREFIX_PATH` for the ROCm prefix, an `ldd` check for missing
  ROCm libraries, the published two-GPU benchmark profile, the low-RAM
  `--resident peer` option, the optional 16K and fused P2P sections, capacity
  and reserve warnings, and reporting guidance for the display-move slowdown
  and the ROCclr teardown abort below.
- `docs/AI_SETUP_EXL3_AMD.md`: GPU ordering by PCIe link and display,
  integrated-GPU ordinals, sizing expert caches for the actual desktop load,
  the reserve recommendation (768 MiB headless primary / 2,048 MiB display
  peer), how automatic sizing retries, and that published Qwen decode numbers
  need the MTP drafter.

## Known issues

- **ROCclr teardown SIGABRT after DONE.** After a completed, fallback-free
  request (`display-main-post-reboot-4bpw-GQ9SU5`: PP32K 1,666.76 tok/s,
  decode 56.87 tok/s, then DONE), the engine aborted during shutdown with
  `Memobj map does not have ptr`. It is the same error signature seen
  previously, not proof of the same cause. Throughput is confirmed; clean
  shutdown is not. A successful generation does not imply a clean exit — keep
  the engine log when it happens.
- **Hot display move between GPUs.** Moving the monitor between GPUs while
  Linux was running was followed by a severe slowdown on the tested machine.
  Rebooting with the monitor unchanged restored both quants (3.05 bpw:
  419.62 → 1,602.39 tok/s prefill, 13.48 → 76.43 tok/s decode; 4.05 bpw peer:
  538.46 → 1,666.76 / 11.13 → 56.87) with no code, configuration or rebuild
  change. The exact trigger is unknown; more than one system variable changed
  with the reboot.

## Not validated

- 64 GB RAM machines (the arenas above were measured on a 96 GB machine).
- CUDA and SYCL builds (compilers unavailable on the test machine).
- Windows, Metal and RDNA4.
- 16K-chunk `--resident peer` prefill (one interrupted arm; no clean
  repetition) and no throughput parity claim between residency modes.
- GPU busy/GTT counters were not a desktop-responsiveness measurement.

## Trying it

Build `strata` for your backend, convert the tokenizer once, then dry-run a
demo launcher before starting it (details in `examples/README.md`):

```sh
export STRATA_EXL3_MODEL=/path/to/Qwen3.8-NEXT-exl3-3bpw
./examples/qwen-exl3-3bpw-resident-all.sh --dry-run
./examples/qwen-exl3-3bpw-resident-all.sh
```

Further reading: `README.md` (two-GPU setup, published profile, options),
`docs/AI_SETUP_EXL3_AMD.md`,
`docs/QWEN_MTP.md`. Run-level evidence for the measurements above is kept in
the maintainer's validation record (`qwen-resident-peer-20261120`); the run IDs
quoted here identify individual arms.
