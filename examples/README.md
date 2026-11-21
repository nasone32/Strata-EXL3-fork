# Portable EXL3 demo launchers

Two example launchers for Qwen3.8-Flash-Next EXL3 on two GPUs, added with the
`exl3-2026.11.21` release. They contain no machine-specific paths and no
binaries: you build the engine and point them at your model, tokenizer and
optional MTP runtime. Both start from automatic expert-cache sizing on both
cards, static peer placement and host-collected peer output (no P2P), matching
the release baseline. Validated on Linux/HIP gfx1100 (two RX 7900 XTX,
Ryzen 9 7950X, 96 GB RAM); other hardware was not validated.

| Launcher | Quant | Residency | Expert caches |
| --- | --- | --- | --- |
| `qwen-exl3-3bpw-resident-all.sh` | 3.05 bpw | `--resident all`: full pinned-RAM expert arena (42.66 GiB registered at 3.05 bpw on the test machine, before other engine and OS memory) | automatic on primary and peer |
| `qwen-exl3-4bpw-resident-peer.sh` | 4.05 bpw | `--resident peer`: peer-owned experts live only in peer VRAM (registered arena 40.10 instead of 56.72 GiB with 7,200 peer experts; not validated on a 64 GB machine) | automatic on primary and peer |

With `--resident peer`, the peer side of the prefill loan is disabled even if
requested; the primary loan still runs. With `--resident all`, eligible
prefill loans from both caches are automatic. Adaptation stays off in both
launchers (`STRATA_EXL3_PEER_ADAPT=0`); with resident-peer experts on the
card, adaptive swaps measured slower.

## Prerequisites

- Build the `strata` engine in this checkout (README build commands,
  `STRATA_ENABLE_HIP=ON`). The launchers use `build-hip-exl3/strata` or
  whatever `STRATA_EXL3_ENGINE` points at. No binaries ship with the release.
- Convert the model tokenizer once with `tools/exl3/tokenizer_to_pack.py`
  (or point `STRATA_EXL3_TOKENIZER` at an existing pack tokenizer).
- Optional, needed to match published decode numbers: the Qwen MTP runtime
  ([Qwen MTP setup](../docs/QWEN_MTP.md)), passed via `STRATA_EXL3_MTP_DIR`.

## Quick start

```sh
export STRATA_EXL3_MODEL=/path/to/Qwen3.8-NEXT-exl3-3bpw
export STRATA_EXL3_MTP_DIR=/path/to/mtp/rt        # optional
./examples/qwen-exl3-3bpw-resident-all.sh --dry-run   # prints the config, starts nothing
./examples/qwen-exl3-3bpw-resident-all.sh             # serve on http://127.0.0.1:8085
```

The 4.05 bpw low-RAM variant works the same way with
`qwen-exl3-4bpw-resident-peer.sh`. A `--dry-run` needs no built engine and
touches nothing.

## Inputs the launchers pass through

| Input | Meaning | Default |
| --- | --- | --- |
| `STRATA_EXL3_MODEL` (or `--model-dir DIR`) | EXL3 model directory | none; the launcher refuses to start without it |
| `STRATA_EXL3_TOKENIZER` (or `--tokenizer DIR`) | converted pack tokenizer | detected in the model directory (`tokenizer/` subdir or model root) |
| `STRATA_EXL3_MTP_DIR` (or `--mtp-dir DIR`) | MTP drafter runtime; the launchers pass `--spec 4` | off |
| `STRATA_EXL3_ENGINE` | engine binary | `build-hip-exl3/strata` in this checkout |
| `STRATA_EXL3_PYTHON` | Python interpreter | `.venv/bin/python`, then `python3` |
| `STRATA_ROCM_LIB` | ROCm library directory for `LD_LIBRARY_PATH` | `/opt/rocm/lib`, prepended only when it exists |

Every trailing option goes to the underlying launcher and overrides the
defaults in the script: `--max-context`, `--prefill`, `--reserve`,
`--peer-reserve`, `--peer-slots`, `--port`, `--engine-log FILE`, `--dry-run`
and more (`tools/run_exl3.py --help`).

## GPU and reserve choices

- The first `--gpu` entry is the primary (dense weights, KV cache, primary
  expert cache); `--peer-device 1` is the peer tier, an index inside that
  visible list. This is expert offload, not a layer split.
- Give the primary the fastest effective PCIe link and, when possible, no
  display. If an integrated GPU occupies ordinal 0, pass the discrete
  ordinals explicitly, e.g. `--gpu 1,2 --peer-device 1`.
- Reserves default to 768 MiB on a headless primary and 2048 MiB on a peer
  that drives the display. Swap them (`--reserve 2048 --peer-reserve 768`)
  when the display sits on the primary. They bound automatic sizing; they are
  not runtime guarantees. Raise the display card's reserve if the desktop
  slows down, and check free VRAM and GTT during prefill and decode
  (`rocm-smi --showmeminfo vram gtt`).
- The 250,000-token capacity is the tested capacity on two 24 GiB cards.
  On smaller cards, pass a lower `--max-context`.
- Automatic sizing checks VRAM during expert upload and after prompt-workspace
  setup and retries a smaller ranked subset when allocations do not fit. Check
  the engine log for the final counts and retries; set explicit
  `--expert-cache` / `--peer-slots` values only to reproduce a benchmark or
  cap an unstable setup.

## Optional flags kept out of the defaults

- **16K prefill chunks** (`--qwen-16k-peer`, native 4.05 bpw, requires
  `--prefill 16384`): improves long-prompt prefill when both cards have free
  VRAM. Validation is incomplete: the `--resident all` arm completed a
  20,000-token prompt at 16K chunks, the `--resident peer` arm stalled
  mid-prompt and was not cleanly repeated. Keep the standard 8K chunks with
  the resident-peer launcher.
- **Fused P2P decode output**
  (`STRATA_EXL3_PEER_DEVICE_OUT=2 STRATA_EXL3_PEER_OUTPUT_FUSED=1`, with
  `--cpu off`): sends peer decode results directly to the primary. It changes
  decode output transport only, needs working GPU peer access, and did not
  consistently beat host collection on the tested pair, so it is off unless
  you export the variables yourself.

## Troubleshooting

- `--dry-run` prints the resolved configuration and validates paths without
  starting anything. `--engine-log FILE` keeps engine diagnostics.
- If the engine fails to find ROCm libraries, run `ldd` on it and export
  `STRATA_ROCM_LIB` (see the README build section).
- A completed generation does not imply a clean shutdown: an intermittent
  ROCclr teardown abort (`Memobj map does not have ptr`) is a known issue in
  this release. Keep the engine log if it happens.
- Moving the monitor between GPUs while Linux is running was followed by a
  severe slowdown on the tested machine; rebooting restored performance
  without any change. See the README troubleshooting section.

Measured numbers and their hardware: [release notes](../docs/RELEASE_EXL3_2026-11-21.md).
