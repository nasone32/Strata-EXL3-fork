# Strata-EXL3-fork

**EXL3 support for Strata, developed and measured on Linux/HIP (RDNA3).** A working fork of
[Strata](https://github.com/Niko1221/Strata), forked through
[Ryutaku/Strata](https://github.com/Ryutaku/Strata). The Git history preserves that base for
future contributions and rebasing. This AMD community-test branch builds on Strata `0.1.44.1`
and Ryutaku's EXL3 integration. The upstream README is preserved as
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## What this fork is

Strata runs **Qwen3.8-Flash-Next** on a normal PC. This fork adds an **EXL3** model backend
(`--exl3`): weights load from turboderp's `safetensors` EXL3 format instead of a GGUF pack, with
an optional n-gram embedding table (`--ple-exl3`) and a second GPU as a peer expert tier
(`--peer-device`). It exists to run the model on AMD Radeon cards under Linux/HIP and to collect
community measurements. It is **not** a general release: the setup wizard remains GGUF-oriented,
and nothing here automates downloading models.

**Setting up with an AI agent?** Point it to [docs/AI_SETUP_EXL3_AMD.md](docs/AI_SETUP_EXL3_AMD.md). This is the EXL3-specific AMD Linux setup guide; the upstream `docs/AI_SETUP.md` and setup wizard remain GGUF-oriented.

**Status of the backends in this fork:**

| Backend | Status in this fork |
| --- | --- |
| Linux + HIP (gfx1100, RDNA3, e.g. RX 7900 XTX) | developed and measured here |
| Linux + HIP (gfx1200/gfx1201, RDNA4) | the switches exist; **needs testing — no support claim** |
| CUDA, Windows HIP, Metal, SYCL | carried from upstream; **not validated by this fork** |

The build checks, default-versus-explicit benchmarks, CPU-affinity results and remaining limits are recorded in [docs/EXL3-RELEASE-VALIDATION.md](docs/EXL3-RELEASE-VALIDATION.md).

## Launchers

Two Bash launchers start the local web chat and the OpenAI/Anthropic-compatible API with the
measured EXL3 settings; both accept `--dry-run` to print the configuration without starting the
server or touching the GPU:

```sh
./run-peer.sh --dry-run --model-dir /path/to/Qwen3.8-Flash-Next-exl3-3bpw        # print the config, start nothing
./run-peer-4bpw.sh --dry-run --model-dir /path/to/Qwen3.8-Flash-Next-exl3-4bpw   # same, 250K-context preset
./run-peer.sh --model-dir /path/to/Qwen3.8-Flash-Next-exl3-3bpw --peer-device 1  # real run (needs a built engine)
```

`--model-dir` (or `STRATA_EXL3_MODEL`) is required even for `--dry-run`; the other model inputs
are detected in that directory: the PLE table (`ngram_embedding.safetensors` or
`gram_embedding.safetensors`) and a pack-layout tokenizer (`<model-dir>/tokenizer/`, or the model
directory itself). `--dry-run` prints the config and exits: no server, no engine, no GPU.

Every knob is a flag or an environment variable; nothing is hard-coded to one machine:

| Flag | Environment variable | Meaning |
| --- | --- | --- |
| `--model-dir` | `STRATA_EXL3_MODEL` | required: the EXL3 model directory |
| `--ple-file` | `STRATA_EXL3_PLE_FILE` | the model's n-gram PLE table — required: detected in `<model-dir>` (`ngram_embedding.safetensors` / `gram_embedding.safetensors`) or named |
| `--tokenizer` | `STRATA_EXL3_TOKENIZER` | pack-layout tokenizer directory; detected in `<model-dir>/tokenizer` (or `<model-dir>` itself) when omitted, else required |
| `--engine-bin` | `STRATA_EXL3_ENGINE` | engine binary (default `build-hip-exl3/strata` under this folder) |
| `--expert-profile` | `STRATA_EXL3_EXPERT_PROFILE` | pre-load the VRAM tier from a profile (default: this checkout's `data/expert-profile.bin`; `--no-expert-profile` omits it) |
| `--peer-device` | `STRATA_EXL3_PEER_DEVICE` | second GPU as peer expert tier (default: none) |
| `--peer-slots` | `STRATA_EXL3_PEER_SLOTS` | expert slots on the peer card |
| `--expert-cache` | `STRATA_EXL3_EXPERT_CACHE` | expert slots per layer on the primary |
| `--max-context` | `STRATA_EXL3_MAX_CONTEXT` | KV/state capacity in tokens |
| `--prefill` | `STRATA_EXL3_PREFILL` | batched prompt chunk size |
| `--reserve` | `STRATA_EXL3_RESERVE_MIB` | MiB left free on the primary card |
| `--peer-reserve` | `STRATA_EXL3_PEER_RESERVE_MIB` | MiB left free on the peer card |
| `--aux-cpus` | `STRATA_AUX_CPUS` | CPU list for non-pool threads; see the caveat below |
| `--mtp-dir` | `STRATA_EXL3_MTP_DIR` | optional MTP drafter directory |
| `--spec` / `--spec-min-p` | `STRATA_EXL3_SPEC` / `STRATA_EXL3_SPEC_MIN_P` | MTP verify window (with `--mtp-dir`; default 4 / 0.5) |
| `--gpu` | `STRATA_EXL3_GPUS` | cards, e.g. `0` or `0,1` |

Explicit values always win, including `--aux-cpus 0` / `STRATA_AUX_CPUS=0`: the launcher uses
`:-`-style defaults, so an exported variable (or flag) overrides the suggestion and an explicit
`0` is forwarded, not swallowed.

### The suggested RDNA settings

The launchers suggest a profile of engine switches measured on the developer's **gfx1100**
(RDNA3) card, applied only when neither the flag nor the environment names them; export any of
them to override, or to turn one off explicitly:

```sh
STRATA_EXL3_RDNA=1 STRATA_EXL3_FUSE=3 STRATA_EXL3_WMMA=1 STRATA_WMMA_GEMM=1 \
STRATA_HIP_WMMA=1 STRATA_EXL3_PRIMARY_MULTIROW=2 STRATA_EXL3_PEER_MULTIROW=4 \
STRATA_EXL3_EXPERT_BATCH=1 STRATA_EXL3_PEER_BATCH=1 STRATA_EXL3_PEER_PREFILL=1 \
./run-peer.sh --model-dir /path/to/Qwen3.8-Flash-Next-exl3-3bpw
```

(`PEER_PREFILL` is default-on with `--exl3` anyway; the launchers state it explicitly so an
exported value of yours still wins.)

With `--exl3` the engine itself turns on the **arch-neutral** EXL3 switches by default
(`STRATA_EXL3_WAVESYNC`, `STRATA_EXL3_PIPE`, `STRATA_EXL3_OVERLAP`, `STRATA_EXL3_NO_PREAD`,
`STRATA_EXL3_PACK_STAGE`, `STRATA_EXL3_PREFILL_BORROW`, the peer tier's `PEER`/`PEER_PADDED`/
`PEER_SPARSE`/`PEER_CONVERT`/`PEER_PACK_PLAN`, `PRIMARY_PACK_PLAN` and
`STRATA_EXL3_STAGE_INLINE=2`) and defaults `STRATA_QFUSE`, `STRATA_MMVQ_BUNCH` and
`STRATA_PF_GEMM` to 1 — an explicit value always wins, and the **EXL3-scoped** names
(`STRATA_EXL3_QFUSE`, `STRATA_EXL3_MMVQ_BUNCH`, `STRATA_EXL3_PF_GEMM`) beat the plain names,
which otherwise keep their meaning for GGUF models unchanged. The launchers state all of these
explicitly (and `STRATA_EXL3_PEER=1` with `--peer-device`), so a printed `--dry-run` config is a
complete, reproducible recipe; every one of them yields to an exported value, `0` included.

The launcher profile above adds only what stays **opt-in** in the engine: the RDNA-specific
switches (`RDNA`, `FUSE`, `EXL3_WMMA`, `WMMA_GEMM`, `HIP_WMMA`) and the batching/multirow modes
(`PRIMARY_MULTIROW=2`, `PEER_MULTIROW=4`, `EXPERT_BATCH`, `PEER_BATCH`), measured together on
RDNA3. **On RDNA4 (gfx12) the same profile is untested** — it is a set of switches the engine
accepts, not a promise; there is no gfx12-specific tuning in this fork.

### The CPU list caveat

`--aux-cpus` / `STRATA_AUX_CPUS` decides where the engine's non-pool threads run
(`include/strata/platform/aux_cpus.hpp`). The engine pins nothing until you say so: an unset
`STRATA_AUX_CPUS` leaves the threads unpinned, and a value or flag (`LIST`, `auto` or `off`) is
forwarded into the engine config verbatim.

Measured on this fork's development machine (Ryzen 9 7950X, 16 cores), warm, a short **CODE** and a
short **PROSE** prompt through the 3 bpw peer preset, all token ids identical between these arms
of the same engine (two runs each for `20-23` and true `off`; the earlier `unset` arm ran as the
engine's then-`auto` default, kept for reference):

| aux CPUs | CODE tok/s | PROSE tok/s |
| --- | --- | --- |
| `20-23` | 75.66 | 62.97 |
| off (no pinning) | 75.43 | 62.48 |
| `auto` (engine default at the time) | 72.19 | 60.94 |

**Opt-in candidate, this machine only:** `--aux-cpus 20-23` — logical CPUs 20-23 are the SMT
siblings of physical cores 4-7. Against true `off` it measures **+0.3% CODE / +0.8% PROSE** —
within noise, so it is a *tested candidate*, not a proven optimum. It is **not** a universal
16-core rule: another CPU's best list depends on its layout, so measure your own before setting
one.

The launchers therefore leave `STRATA_AUX_CPUS` **unset by default** (no machine-specific value is
baked in, and the engine's unpinned default is sensible): export it or pass `--aux-cpus` to
forward your own list, including an explicit `off` or `0`.

## Building

A normal build fetches the pinned llama.cpp commit (`third_party/ggml/VERSION.txt`) over the
network during configure — reading a public source repository, not publishing anything. To build
without that fetch, point CMake at a local llama.cpp checkout with `-DSTRATA_GGML_DIR=`:

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt -r tools/exl3/requirements.txt
cmake -B build-hip-exl3 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip-exl3 --target strata exl3_flags_test exl3_route_plan_test \
      peer_adapt_plan_test exl3_moe_multi_parity exl3_peer_test -j2
ctest --test-dir build-hip-exl3 \
      -R '^(exl3_flags_test|exl3_route_plan_test|peer_adapt_plan_test|exl3_moe_multi_parity|exl3_peer_test)$' \
      --output-on-failure
```

`-DSTRATA_GGML_DIR=/path/to/llama.cpp` replaces the fetch with a local checkout of the pinned
commit. Building needs a ROCm 7 hipcc with hipBLAS (`docs/AMD_HIP.md`); the engine ends up at
`build-hip-exl3/strata`, which is the launchers' default `--engine-bin`. `docs/AMD_HIP.md` has the
Windows HIP path and `docs/EXL3-HIP-BENCH.md` the build this fork was measured with.

## Getting a model

**The EXL3 launchers do not download or select a model automatically.** They require a local
model directory supplied with `--model-dir` or `STRATA_EXL3_MODEL`. The upstream `setup.py`
model downloader remains in the source tree for GGUF, but it is **not the EXL3 setup flow**:
do not use it to obtain an EXL3 model or this fork's engine.

The two model branches tested during this fork's development are:

- [3.05bpw_h5_ng5](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5) — use `run-peer.sh`.
- [4.05bpw_h6_ng6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) — use `run-peer-4bpw.sh`.

Download one complete branch into its own folder using your preferred HuggingFace client,
then point the appropriate launcher at that folder. Keep the branch's index, all referenced
weight shards, configuration, tokenizer assets and matching n-gram table together; do not mix
files from the two branches. `docs/EXL3.md` describes the expected format. New users should
follow [docs/AI_SETUP_EXL3_AMD.md](docs/AI_SETUP_EXL3_AMD.md) for hardware-appropriate settings.

Prepare the server's pack-layout tokenizer from the downloaded model with
`tools/exl3/tokenizer_to_pack.py`:

```sh
python3 tools/exl3/tokenizer_to_pack.py \
  --hf /absolute/path/to/model --out /absolute/path/to/tokenizer-pack --check
```

Do not republish quantized model data you do not have the rights to.

## Contributing a community test

This fork targets a **small circle of AMD Radeon (RDNA3/RDNA4) owners on Linux/HIP** — it is not a
general release, and reports are how the RDNA4 path and other cards get measured at all. Strata's
contribution rules apply (see [AGENTS.md](AGENTS.md) and
[docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md)): report from real hardware with
measured numbers, compare against a same-day build, and say what you did not test. For this fork
a report needs, at minimum:

1. **Output correctness against your baseline.** Same model, same prompts: compare the fork's
   answers against the *same quantization* on your reference engine — e.g. the model's EXL3 files
   in ExLlamaV3. EXL3 and GGUF are different quantizations of the same weights, so do not expect
   bit-identical text across them; compare like with like. `ctest -R 'exl3|ple_exl3'` runs the
   EXL3 parity tests (`exl3_parity_k`, `exl3_loader_validation`, `exl3_gpu_parity`,
   `exl3_gemv_parity`, `exl3_moe_multi_parity`, `ple_exl3_parity`, ...), and
   `tools/exl3/check_real.py` prints a checksum of one real weight as the C++ kernel sees it.
2. **The architecture, model and flags.** `gfx_target_version`, the card, the ROCm version, the
   model and its bitrate, and exactly which `STRATA_*` switches were on — a run with the
   defaults and one with the RDNA profile above, both labeled. AMD-only paths belong here;
3. **Warm-versus-cold, stated.** The first EXL3 prompt reads tens of GiB from disk; a cold OS
   page cache is not the steady state. If you drop the page cache between runs
   (`/proc/sys/vm/drop_caches`, needs root, affects the whole machine), say so.
   `tools/exl3/warm_session.py --engine <engine> --config <config> --output <dir>` runs two GENs
   in one engine session so the second measures warm decode; `tools/exl3/bench.py` keeps the raw
   engine logs and exact prompt/output ids. Label every number cold or warm.

### Roadmap

- **More AMD cards on Linux/HIP** — RDNA3 reports (gfx1100 beyond the development machine) and
  RDNA4 (gfx1200/gfx1201) are the biggest gap: the switches exist and nothing about them is
  measured on that generation yet.
- **NVIDIA** — CUDA EXL3 work was validated on one vGPU and merged from a separate branch (see
  `docs/EXL3-CUDA-VALIDATION.md`); PRs from NVIDIA owners are welcome, especially parity runs
  against the same files (`ctest -R 'exl3|ple_exl3'`) on consumer cards.
- **Upstream** — the plan is to rebase onto the latest upstream Strata and contribute the EXL3
  work there. The `amd-exl3-community` branch preserves Ryutaku's base history; keep upstream
  code updates separate from AMD optimization and packaging changes to simplify review.

## License

MIT, as upstream ([LICENSE](LICENSE)); third-party notices in `third_party/` and `SECURITY.md`
carry over.

## Provenance

Source snapshot of [Strata](https://github.com/Niko1221/Strata) `0.1.44.1` (branch
`exl3-hip-prefill-pipe`, commit `a27eefef`) with the EXL3/HIP work on top. The upstream README is
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). No claims here are upstream claims: measured
numbers in the docs come from the machines stated beside them, and every untested path is labeled
above.