# Strata EXL3

Run **Qwen3.8-Flash-Next in EXL3 format** on a PC, using GPU memory and system RAM.
There is a browser chat and an OpenAI- and Anthropic-compatible API.

This is [nasone32's fork](https://github.com/nasone32/Strata-EXL3-fork) of
[Strata](https://github.com/Niko1221/Strata), through
[Ryutaku's EXL3 integration](https://github.com/Ryutaku/Strata). Qwen is the main model
here. MiMo support is a separate experiment, described near the end of this page.
The original Strata introduction is in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

**Tested here:** Linux, ROCm/HIP, Ryzen 9 7950X, 96 GB installed RAM and two
RX 7900 XTX cards (gfx1100). One-GPU Qwen with CPU assistance has also been tested.
CUDA, Windows, Metal, SYCL and other Radeon generations are not validated by this
fork. Upstream support for them is not a test of these EXL3 changes.

## Models

Download a complete quant into its own directory. Do not mix shards, n-gram tables
or configuration files from different downloads.

| Model | Quant tested here |
| --- | --- |
| Qwen3.8-Flash-Next | [EXL3 3.05 bpw](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5) |
| Qwen3.8-Flash-Next | [EXL3 4.05 bpw](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) |
| MiMo-V2.6-Flash-MOPD | [EXL3 2.20 bpw only](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-MOPD-EXL3/tree/main/2.20bpw) |

**Mixed Qwen quants are untested.** The loader accepting a file is not a correctness
or performance result. MiMo results do not apply to other MiMo quants.

Allow room for the original model, the OS, and your applications. Full RAM residency
locks the expert pool in memory; the engine checks whether it fits and refuses if
it cannot leave its safety headroom. A disk repack needs another expert-pool-sized
allocation on disk. It does not replace the original model directory.

## Build once

An AI assistant can follow [docs/AI_SETUP_EXL3_AMD.md](docs/AI_SETUP_EXL3_AMD.md).
For EXL3, **do not use the upstream GGUF setup wizard or its downloaded binaries**.
Build this checkout with a working ROCm 7 installation, hipBLAS, CMake, Ninja and a
C++ compiler. See [the AMD build notes](docs/AMD_HIP.md) for driver/toolkit details.

```sh
git clone --branch amd-exl3-community https://github.com/nasone32/Strata-EXL3-fork.git
cd Strata-EXL3-fork
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r tools/exl3/requirements.txt
cmake -S . -B build-hip-exl3 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip-exl3 --target strata exl3_band8_build -j2
```

CMake fetches the pinned llama.cpp source dependency. To use an existing checkout,
add `-DSTRATA_GGML_DIR=/path/to/llama.cpp`. Do not use `-j$(nproc)` on a low-RAM PC;
HIP compilation uses quite a bit of memory.

Adjust `/opt/rocm` if ROCm is installed elsewhere. Run `ldd build-hip-exl3/strata`
before starting the server or benchmarks. If ROCm libraries such as
`libhipblaslt.so.1` show `not found`, add the directory that contains them to
the current shell and rerun `ldd`. For example, for an install using `/opt/rocm/lib`:

```sh
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Convert the downloaded model's tokenizer once:

```sh
python tools/exl3/tokenizer_to_pack.py --hf /path/to/qwen-exl3 --out /path/to/qwen-exl3/tokenizer
```

## Start Qwen: smoke test, not the benchmark profile

See the [portable two-GPU examples](examples/README.md) for 3.05 bpw full
residency and 4.05 bpw peer residency, or read the
[release notes](docs/RELEASE_EXL3_2026-11-21.md).

These commands do not enable MTP. To compare decode rates, use the
[published two-GPU settings](#match-the-published-two-gpu-qwen-setup) below.

```sh
# One GPU. CPU help is enabled by default.
./run-peer.sh --model-dir /path/to/qwen-3.05bpw

# The 4.05 bpw model uses the same launcher settings.
./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw

# Two GPUs: example with GPU 0 on the faster PCIe link and GPU 1 driving the display.
./run-peer.sh --model-dir /path/to/qwen-3.05bpw --gpu 0,1 --peer-device 1 \
  --expert-cache auto --reserve 768 --peer-reserve 2048
```

Open **http://127.0.0.1:8085**. For an app, use **http://127.0.0.1:8085/v1** as the
OpenAI-compatible base URL. Add `--open` to open the browser automatically, or
`--port 8086` to run a second server. Ctrl+C stops it.

The launcher detects `tokenizer/` and the n-gram table inside the model directory.
Use `--tokenizer DIR` or `--ple-file FILE` if you keep them elsewhere.
`--engine-bin FILE` selects a different build. `--dry-run` prints the configuration
without starting the model.

There is no long speed-flag recipe to paste. The engine selects the tested fast
kernel defaults on gfx1100, including the optimized native serial prefill
kernels. Those kernels are active with the standard 8K chunk as well.
Other architectures keep their existing dispatch.
An explicit environment value, including `0`, still overrides a kernel default.
Repacking is **off** unless you select it yourself.

## Settings worth changing

These depend on your PC or workload. Start with the defaults, not someone else's
CPU affinity list or a cache sized for a different GPU.

For two-GPU Qwen, start with **automatic sizing on both cards**. Leave
`--peer-slots` unset and use `--expert-cache auto`. Choose the reserves for
the workload and the desktop: start with **768 MiB on the headless GPU**
and **2,048 MiB on the GPU that drives the display**. Use the GPU with the
fastest effective PCIe link as the primary (for example, x16 over x8 at
the same generation). Ideally connect the display to the other GPU; if
the links are equivalent, prefer the headless GPU as primary.

These are recommendations, not changes to the launcher defaults. With a
display on the primary and a headless peer, use `--reserve 2048 --peer-reserve
768` with automatic sizing. Check the actual expert counts in the engine log and
free VRAM during prefill and decode. If the desktop slows down, increase
the reserve on the display GPU first. Use fixed counts to reproduce a
benchmark or to cap an otherwise unstable setup.

| Setting | Default / when to change it |
| --- | --- |
| `--resident all` | Full pinned expert pool in RAM. Use `--resident 0` for native Qwen if RAM is too tight; this uses file-backed expert staging and can be slower. |
| `--resident peer` | Native Qwen or MiMo with a peer GPU. Stores peer-owned experts only on that GPU, not in pinned RAM. Qwen requires the serial server and peer prefill. Repacking needs `all`. |
| `--gpu 0,1 --peer-device 1` | A peer is optional; default is GPU 0 alone. Peer index 1 means the second card in the visible list. This is expert offload, not a layer split. |
| `--cpu auto\|on\|off` | Auto enables CPU help for one-GPU Qwen and for MiMo; it leaves it off for Qwen with a peer. Try `--cpu on` with one Qwen GPU. MiMo is normally run with it on. |
| `--cpu-threads N` | At most 8 workers by default, based on available CPU count. More is not always faster. |
| `--aux-cpus LIST` | Unpinned by default. `off`, `auto`, or a list such as `4-7`; numbering is machine-specific. This is separate from the expert worker count. |
| `--expert-cache N` | Qwen: `auto` sizes the primary GPU cache. Smaller caches leave room for context or the desktop. |
| `--peer-slots N` | Qwen: optional upper limit on peer experts. Leave unset for automatic sizing from available VRAM. Automatic sizing can retry a smaller subset if the device allocations do not fit. |
| `--reserve MiB`, `--peer-reserve MiB` | Qwen: primary/peer sizing margins, default 2048/4096 MiB. These are not runtime guarantees; desktop activity and inference can change free VRAM. An explicit primary cache count is not resized by `--reserve`. MiMo uses `--expert-vram GiB` for primary experts. |
| `--max-context N` | Qwen 65536, MiMo 16384. This is capacity, not a guarantee that a long prompt fits your VRAM. |
| `--prefill N` | Qwen 8192, MiMo 512. Smaller prompt chunks can reduce peak memory. |
| `--expert-profile FILE` | Optional initial expert placement; the bundled Qwen profile is used when available. `--no-expert-profile` starts without it. |
| `--mtp-dir DIR --spec N` | Optional Qwen draft model; default window 4 when a drafter is supplied. No drafter is downloaded or silently selected. |

For Qwen adaptation, the engine arguments `--adapt-every`, `--adapt-swaps`,
`--adapt-decay`, `--peer-adapt-swaps`, `--peer-adapt-mib` and `--peer-adapt-min-gain`
control how quickly the GPU caches change. Use a server JSON's `args` list for
these advanced engine options. `STRATA_EXL3_PEER_ADAPT=1` enables adaptive peer
selection; it is not forced on because its value depends on the workload and link.
Auxiliary affinity is `STRATA_AUX_CPUS`; Qwen CPU workers are
`STRATA_EXL3_CPU_THREADS`. CPU hybrid quota is `STRATA_EXL3_CPU_HYBRID` (default
launcher suggestion 2), with `STRATA_EXL3_CPU_HYBRID_MIN_MISSES` (1).

For a saved server JSON, the portable interface is:
`python tools/strata_experts.py launch --config my-qwen.json -- --port 8085`.
Its `config` subcommand can create a starting file; see `--help`.

## Match the published two-GPU Qwen setup

The fixed counts in this section are for benchmark reproduction, not the
recommended starting point for a different GPU pair or desktop load.

**All published Qwen decode rates below use MTP.** The matching
[Qwen MTP runtime](docs/QWEN_MTP.md) is separate from the EXL3 model directory.
These are the candidate settings for two 24 GiB RX 7900 XTX cards. Use the
[release benchmark runner](bench/release-2026-11-12/README.md) to also match the
prompts, generation lengths and cache handling. That runner keeps the later
serial prefill switches off to reproduce the November 12 profiles.

```sh
STRATA_EXL3_PEER_ADAPT=0 ./run-peer.sh --model-dir /path/to/qwen-3.05bpw \
  --gpu 0,1 --peer-device 1 --cpu off --resident all \
  --expert-cache 165 --peer-slots 10200 --prefill 8192 --max-context 65536 \
  --mtp-dir /path/to/mtp/rt --spec 4 --spec-min-p 0.5

STRATA_EXL3_PEER_ADAPT=0 ./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw \
  --gpu 0,1 --peer-device 1 --cpu off --resident all \
  --expert-cache 104 --peer-slots 7200 --prefill 8192 --max-context 65536 \
  --mtp-dir /path/to/mtp/rt --spec 4 --spec-min-p 0.5
```

Run one command at a time. Add `--dry-run` to check the resolved config. If
the fixed benchmark profile does not fit, use automatic sizing or lower
the appropriate cache cap. Report any change when comparing results.

On the tested build, `--resident all` locked **42.66 GiB** (3.05 bpw) or
**56.72 GiB** (4.05 bpw) for the expert arena alone, before other engine and OS
memory. Installed RAM is not available RAM. For native Qwen, `--resident 0`
selects file-backed expert staging. This is a different benchmark mode; the
OS can still cache file pages, so file-backed does not imply physical SSD
reads on every miss.

### Low-RAM option for two-GPU Qwen

```sh
STRATA_EXL3_PEER_ADAPT=0 ./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw \
  --gpu 0,1 --peer-device 1 --resident peer \
  --expert-cache auto --reserve 768 --peer-reserve 2048 --mtp-dir /path/to/mtp/rt
```

`--resident all` remains the default. With `--resident peer` and 7,200
experts on the peer, the registered expert arena measured **40.10 GiB instead of
56.72 GiB** at 4.05 bpw, and **30.16 instead of 42.66 GiB** at 3.05 bpw.
These are arena sizes, not whole-process requirements or validation on a 64 GB
machine. With static peer placement, 4.05 bpw decode was essentially unchanged
in the controlled A/B; adaptive swaps are slower in this mode because outgoing
experts must be copied back to RAM. Start with adaptation off.

Lowering `--peer-slots` frees peer VRAM but **increases** the pinned RAM
complement. The primary can still lend cache space during prefill. With
`--resident peer`, only the peer loan is disabled, so peer-owned weights
stay on that card.

### Optional: 16K prefill on two RX 7900 XTX

If both GPUs have enough free VRAM, add `--qwen-16k-peer` to the native
4.05 bpw command to improve long-prompt prefill. It selects 16K chunks,
defaults to 7,200 peer experts. Native serial Qwen normally borrows space
from the primary cache and, except with `--resident peer`, from the peer
cache during prefill; the experts are restored before decode. This profile
sets the peer loan target to 400 where borrowing is enabled. If VRAM is tight, omit this option and keep the
standard 8K chunks. This profile is separate from the historical benchmark below.

### Optional: direct peer-to-primary output for Qwen

The second GPU still computes its experts with P2P off. The default sends
their decode results through the host; keep that path for baseline tests.
To try direct peer-to-primary output with fused wait/scatter, use the serial
two-GPU launcher with `--cpu off` and prefix your usual command with:

```sh
STRATA_EXL3_PEER_DEVICE_OUT=2 STRATA_EXL3_PEER_OUTPUT_FUSED=1 \
  ./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw \
  --gpu 0,1 --peer-device 1 --peer-slots 7200 --cpu off \
  --mtp-dir /path/to/mtp/rt
```

This requires working GPU peer access. It enables the external verifier's
GPU-hit path and changes decode output transport only; activations going to
the peer still travel through the host. To turn it off, unset both variables
or set `STRATA_EXL3_PEER_DEVICE_OUT=0`. Fused P2P did not consistently beat
host collection on the tested pair, so it is not a default.

For a slow run, record the commit and resolved config. If saving engine logs
is needed, use a saved server JSON with a `log` path. Check for
`resident EXL3 full arena`, `EXL3 peer tier on device 1`, the shutdown
`EXL3 peer summary` (computed routes and fallbacks), and `drafts accepted` in
the request stats. GPU detection or VRAM usage alone does not confirm peer compute.

## Optional: repack the experts

A repack rearranges compressed expert data for shared CPU/GPU use. It does not
re-quantize the model. Dense weights, the n-gram table and draft model are not
repacked, so **keep the original download**.

### Build once, load from disk later

```sh
# Check space first. Then repeat without --dry-run to build it.
python tools/strata_experts.py repack --model /path/to/qwen-4.05bpw \
  --output /path/to/cache/qwen-4.05.strata-experts.json --dry-run
python tools/strata_experts.py repack --model /path/to/qwen-4.05bpw \
  --output /path/to/cache/qwen-4.05.strata-experts.json

./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw --cpu on \
  --experts /path/to/cache/qwen-4.05.strata-experts.json
```

Pass the **descriptor `.strata-experts.json`**, not its data file or generation
directory. Launch validates it against the source model and opens the existing
generation. Building is a separate action. Moving or changing the original model
can invalidate a descriptor; use `validate ... --model DIR` before launching.

### Or repack directly into RAM at each start

```sh
./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw --cpu on --repack-in-memory
```

No packed disk copy is created. The model is read and rearranged into the pinned
RAM arena on **every launch**, so startup takes longer. Both modes need RAM for
the full expert pool plus the rest of the engine and safety headroom. The measured
4 bpw Qwen arena was about **56.7 GiB** on the development machine.

**Limits:** packed CPU experts need AVX-512BW. The validated Qwen packed setup is
**one GPU plus CPU assistance** on Linux/HIP gfx11. Packed Qwen with a peer GPU has
an unresolved output-divergence issue and is deliberately refused unless explicitly
marked experimental. Stay with native experts for normal two-GPU Qwen use.
`--cpu off` keeps a GPU-only packed run. Native and CPU-assisted arithmetic can
produce different token choices; a storage repack is not a promise of identical
answers between CPU and GPU policies.

## MiMo: experimental

Only [vcruz305's MOPD 2.20bpw quant](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-MOPD-EXL3/tree/main/2.20bpw)
was tested for this release. Prepare its tokenizer with the same conversion tool,
then use:

```sh
./run-mimo-mopd.sh --model-dir /path/to/mimo-2.20bpw --cpu on
# Add --gpu 0,1 --peer-device 1 for a peer, --drafter DIR for DFlash,
# or --experts DESCRIPTOR / --repack-in-memory for an explicitly selected repack.
```

CPU assistance is recommended, including with two GPUs. The tested MOPD repack
held **77.44 GiB** of pinned expert data, before other memory use. Leave plenty of
headroom. MiMo's device-result paths are experimental: the launcher enables device
verification misses and, when a peer is selected, peer device results. Set
`STRATA_MIMO_WINDOW_DEVICE_MISSES=0` or `STRATA_MIMO_PEER_DEVICE_RESULTS=0` to compare
with the host transport. A silent GPU fault can still hang a request; this is not
an automatic-recovery guarantee. These flags do not enable anything in Qwen.

## Benchmarks

Short release check on **Ryzen 9 7950X, 96 GB RAM, two RX 7900 XTX, Linux,
ROCm core-7.14**. These are individual runs, not averages or medians.
The published baseline is `3be44426`; both builds use the same compiler and
dependency checkout. Native experts, CPU help off, MTP window 4, 64K context,
8192-token prompt chunks. Primary/peer cache sizes are matched between builds:
165/10200 slots at 3.05 bpw, 104/7200 at 4.05 bpw.

### Two GPUs: published build vs this update

Decode, tokens/s; **512 generated tokens** per CODE/PROSE run:

| Quant / prompt | Published, run 1 | Published, run 2 | Update, run 1 | Update, run 2 |
| --- | ---: | ---: | ---: | ---: |
| 3.05 bpw CODE | 49.7 | 78.3 | 85.3 | 87.3 |
| 3.05 bpw PROSE | 59.0 | 66.0 | 71.5 | 70.4 |
| 4.05 bpw CODE | 25.5 | 61.9 | 76.1 | 76.7 |
| 4.05 bpw PROSE | 43.0 | 57.8 | 68.1 | 67.6 |

**32,768-token prefill**, tokens/s; each run reads the full prompt, no prefix reuse:

| Quant | Published, run 1 | Published, run 2 | Update, run 1 | Update, run 2 |
| --- | ---: | ---: | ---: | ---: |
| 3.05 bpw | 1528 | 1538 | 1507 | 1514 |
| 4.05 bpw | 1372 | 1512 | 1516 | 1523 |

CODE asks for a bubble sort in C; PROSE asks for a PID controller explained simply.
Greedy, seed 1234; generation is capped at 512, not a complete-answer quality test.
The long synthetic prompt has a 16-token answer limit because that test measures
prefill. The second run of each prompt follows the first in the same process;
expert caches stay warm, but conversation/checkpoint reuse is disabled. No OS
cache flushing was done, so "run 1" does **not** mean a controlled cold-disk test.

This compares the **release profiles**, including the update's full pinned-RAM
residency and fast kernel defaults, against the old file-backed profile. Startup
is excluded; full residency pays its expert reads up front. Some continuations
differ between builds (all PROSE pairs and the second 4.05 bpw CODE pair), so
these are not byte-identical speedup measurements. The full counts, token IDs,
settings and reproduction command accompany the release benchmark record.
There is no clear 32K prefill improvement at 3.05 bpw in this small test.

### One GPU: 4.05 bpw

Same candidate binary, 104 primary slots, full RAM residency and MTP settings.
The CPU-assisted arm uses **8 workers plus runtime RAM repacking**. CPU work was
confirmed in the log (60,466 routed rows). This tests the combination, not
repacking alone; conversion time at startup is excluded.

| Test | GPU only, run 1 | GPU only, run 2 | CPU + repack, run 1 | CPU + repack, run 2 |
| --- | ---: | ---: | ---: | ---: |
| CODE decode, tok/s | 54.3 | 54.8 | 58.4 | 59.3 |
| PROSE decode, tok/s | 48.7 | 44.8 | 48.0 | 51.6 |
| 32K prefill, tok/s | 1369 | 1365 | 1360 | 1357 |

All decode rows generated 512 tokens. The CPU-assisted run was not faster in
every individual case; prefill was essentially unchanged. See the
[benchmark record and reproduction command](bench/release-2026-11-12/README.md)
for both comparisons, exact counts and output IDs.

## Problems

If a run fails, keep the config and engine log. Report the exact quant, CPU, RAM,
GPU(s), ROCm version, context size and launch command in
[this fork's issues](https://github.com/nasone32/Strata-EXL3-fork/issues).
Do not include API keys or private prompts in a public log.

On the tested dual RX 7900 XTX system, moving the monitor between GPUs
while Linux was running was followed by a severe slowdown. Rebooting
with the monitor still on the primary restored normal performance on
both Qwen quants without a code or configuration change. The exact
driver or desktop trigger is unknown.

An intermittent ROCclr shutdown abort (`Memobj map does not have ptr`)
has also been observed after a completed request. A successful generation
does not imply a clean shutdown; keep the engine log if this happens.

The server is local-only by default. Listening on another interface requires
`--host ADDRESS --api-key SECRET`; do not expose it without a key.

## Credits

Strata and its original backends: [Niko1221/Strata](https://github.com/Niko1221/Strata).
EXL3 integration: [Ryutaku/Strata](https://github.com/Ryutaku/Strata).
EXL3 format and Qwen quants: [turboderp](https://huggingface.co/turboderp).
Models: the Qwen and Xiaomi teams; MiMo quant: vcruz305.
See [LICENSE](LICENSE) and the models' own licenses.
