# EXL3 setup for an AI assistant

Use this guide for **nasone32/Strata-EXL3-fork**, not the upstream GGUF installer.
The [README](../README.md) is the human guide and lists the supported commands.
Qwen3.8-Flash-Next is the primary model. MiMo is experimental.

## Before changing anything

1. Check Linux, CPU ISA, available RAM (not just installed RAM), disk space, GPU
   names/VRAM and the installed ROCm version. `free -h`, `df -h`, `lscpu`,
   `rocm-smi` and `rocminfo` are useful. Ask before installing drivers or packages
   requiring administrator privileges.
2. Explain the download size and ask before downloading a model. Models and caches
   stay outside Git. Never delete or replace an existing model to make space.
3. Check for other GPU users and occupied ports. Do not kill an unknown process.
4. Keep the server on `127.0.0.1`. Network access requires both the user's consent
   and `--api-key`. Do not publish credentials, private chats or machine logs.

The tested machine is Linux/HIP gfx1100 (two RX 7900 XTX, Ryzen 9 7950X, 96 GB RAM).
Do not describe this as CUDA, Windows, RDNA4, Metal or SYCL validation.

On our dual-7900 XTX test machine, the **primary is the card without a display**
(PCI `0000:03:00.0`); the peer drives the display (PCI `0000:09:00.0`).
The primary therefore starts with more VRAM available for KV cache and experts.
On another machine, check the display connections, free VRAM and negotiated
PCIe width and speed under load before choosing the order. Put the GPU with
the fastest effective PCIe link first in `--gpu` (for example, x16 over x8 at
the same generation). Ideally connect the display to the other GPU. If the
links are equivalent, choose the headless GPU as primary. `--peer-device 1`
refers to the second visible card, not its physical GPU index.

If HIP also lists an integrated GPU, leave its ordinal out of the inference
device list. Check HIP ordinals and PCI bus IDs rather than assuming the
discrete cards are `0,1`. With the launcher, pass only the discrete ordinals
in `--gpu`; it sets `HIP_VISIBLE_DEVICES` for the engine. For example, if
HIP lists the iGPU as `0` and the discrete cards as `1` (primary) and `2`
(peer), use `--gpu 1,2 --peer-device 1`. For a direct engine invocation,
set `HIP_VISIBLE_DEVICES=1,2` and use `--peer-device 1`. Do not stack a
separate visibility mask with launcher ordinals: masking can renumber
devices. This only hides the iGPU from inference; it does not disable it
for the desktop.

### Size the expert caches for the actual desktop load

The peer expert counts in our profiles were sized on **Ubuntu 24.04 with
GNOME**, with the peer driving the display. They are not universal targets.
Leave room for the compositor, other GPU apps and inference workspaces.
Check both cards during prefill and decode, not just at startup. Compare
VRAM and GTT (GPU-accessible system RAM) usage with the same desktop load.
On AMD, `rocm-smi --showmeminfo vram gtt` can help.

If VRAM is nearly full and GTT use grows along with a throughput drop,
check for GPU allocations spilling into system RAM. Nonzero GTT alone
does not prove spill: Strata deliberately uses pinned host memory for
expert residency and staging. If the display peer is under pressure, lower
`--peer-slots`; if the primary is under pressure, lower `--expert-cache`.
Retest at the intended context capacity and prompt chunk size. With
`--resident peer`, lowering `--peer-slots` also increases the pinned RAM
needed for experts, so check available system RAM again.

On a headless machine or when more VRAM is genuinely available, try higher
slot counts in small steps, one GPU at a time. Measure throughput and
minimum free VRAM before keeping a new placement; filling more VRAM does
not by itself make inference faster.

With no explicit peer cap, Qwen sizes the tier from available VRAM,
including plane alignment and a prompt-workspace estimate in addition to
the desktop reserve. It checks VRAM during upload and after prompt
workspace setup. If the allocations do not fit, it retries a smaller
ranked subset before publishing residency. Check the log for the final
count and any retries. If the desktop becomes sluggish, stop the test
and increase the peer reserve or set a lower explicit cap.
This is not a guarantee against later desktop spikes or the extra workspace
of a deferred 16K prompt loan.

## Model choice

- Qwen: tested [3.05bpw_h5_ng5](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)
  and [4.05bpw_h6_ng6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6).
  **Mixed Qwen quants remain untested.**
- MiMo: only [MiMo-V2.6-Flash-MOPD-EXL3 / 2.20bpw](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-MOPD-EXL3/tree/main/2.20bpw)
  is the tested release quant. Do not silently substitute a different quant or a
  previous MiMo checkpoint.
- Download a complete branch/subdirectory. Keep its configuration, index, all
  referenced shards, tokenizer assets and matching n-gram table together.
- Full residency needs the entire expert pool in available RAM plus engine/OS
  headroom. The engine's check is authoritative; never override it with invented
  free-memory values. Qwen 4 bpw's measured arena was about 56.7 GiB; MiMo MOPD's
  was 77.44 GiB. Those are not whole-process RAM requirements.

## Build this fork

Clone `https://github.com/nasone32/Strata-EXL3-fork.git`, branch `amd-exl3-community`.
Follow the README build commands with `STRATA_ENABLE_HIP=ON`, CUDA off and the
actual architecture. The measured target is `gfx1100`. Use a small compile job
count; do not assume all CPU threads fit in RAM.
If CMake cannot find HIP, check the ROCm prefix. Before launching, run `ldd` on
the built `strata` and resolve missing ROCm libraries as shown in the README.

Build **`strata`**, the executable used by the web server, and **`exl3_band8_build`**
if the user wants offline repacking. Building the `strata_mimo` library alone does
not update the server executable. Do not reuse an upstream release binary and
claim it contains this fork's changes.

The Python requirements are `requirements.txt` and `tools/exl3/requirements.txt`.
Create/activate `.venv`. Convert a tokenizer once:

```sh
python tools/exl3/tokenizer_to_pack.py --hf /path/to/model --out /path/to/model/tokenizer
```

No GPU or model is needed for these launcher checks:

```sh
python tools/test_run_exl3.py
python tools/test_strata_experts.py
```

## Configure and start

Start with a dry run, then remove `--dry-run`:

```sh
./run-peer.sh --model-dir /path/to/qwen-3.05bpw --dry-run
./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw --dry-run
```

- Default: one GPU, CPU help on, full RAM residency, native/unrepacked experts,
  port **8085**. The engine selects its gfx1100 fast defaults itself; do not add a
  page of kernel environment variables to a user's command. The native serial
  prefill improvements are on at the standard 8K chunk. Suggest `--qwen-16k-peer`
  separately, only for native 4.05 bpw on two RX 7900 XTX cards with enough VRAM.
- Two GPUs: add `--gpu 0,1 --peer-device 1`. The first visible GPU is primary,
  `--peer-device` is an index in that visible list. This is peer expert offload,
  not layer splitting. CPU help defaults off for Qwen with a peer; `--cpu on`
  explicitly requests it.
- Qwen P2P decode output is optional and off in baseline tests. The second GPU
  still computes its experts with host collection. Only enable P2P when requested:
  use `STRATA_EXL3_PEER_DEVICE_OUT=2` and `STRATA_EXL3_PEER_OUTPUT_FUSED=1` with
  the serial two-GPU launcher and `--cpu off`. See the [README example](../README.md#optional-direct-peer-to-primary-output-for-qwen).
- Low RAM: native serial Qwen with peer prefill, or MiMo with a peer, can use
  `--resident peer`. This excludes peer-owned experts from pinned RAM.
  For Qwen the primary prefill loan is still eligible, but the peer loan is
  disabled even if requested. With full residency, eligible peer loans are
  automatic by default; `STRATA_EXL3_PEER_PREFILL_BORROW=0` opts out.
  Qwen cannot combine this with batching, repacking, or CPU execution of
  peer-owned experts. Start with static peer placement. `--resident 0` is
  Qwen's separate file-backed mode; RAM-only repacking requires `all`.
- Start with `--expert-cache auto` and leave `--peer-slots` unset. Choose
  `--max-context` and `--prefill` for the workload before tuning reserves.
  Recommend **768 MiB on the headless GPU and 2,048 MiB on the display GPU**.
  For the usual headless-primary topology, pass `--reserve 768 --peer-reserve
  2048`. These are recommendations, not launcher defaults or guarantees.
  If the fastest PCIe GPU drives the display, use `--reserve 2048
  --peer-reserve 768` with automatic sizing. Add headroom if the desktop slows down or the request approaches
  the VRAM limit. Do not copy fixed slot counts or 250K capacity from
  benchmark profiles to other machines.
- CPU worker count: `--cpu-threads`; auxiliary affinity: `--aux-cpus` or
  `STRATA_AUX_CPUS`. Leave affinity unset initially. CPU numbering varies.
- MTP is optional for a smoke test but **required to match published Qwen decode
  numbers**: `--mtp-dir DIR --spec 4`. Do not enable it without the matching
  drafter. For the 2×7900 XTX comparison use the [explicit two-GPU profile in
  the README](../README.md#match-the-published-two-gpu-qwen-setup), not the
  quick-start settings. If automatic peer sizing hits a plane-allocation OOM,
  use a smaller `--peer-slots` cap and record it. MiMo uses `--drafter DIR` for
  DFlash instead.
- Adaptation is a tuning choice, not a compulsory kernel optimization. The
  README lists the main/peer cadence, swap budget and decay controls. Use a saved
  server config's `args` for engine-level options; options after `--` in the
  launcher go to the Python server, **not** the engine.
- `--engine-bin` selects a build, `--tokenizer` a converted tokenizer directory.
  Preserve explicit environment overrides, including `0`. Do not print secrets
  in a dry-run record that will be published.

## Repacking is optional and off by default

Two choices, never combine them:

```sh
# Offline, CPU-only build. Preview space before writing.
python tools/strata_experts.py repack --model /path/to/qwen-4.05bpw \
  --output /path/to/cache/qwen.strata-experts.json --dry-run
python tools/strata_experts.py repack --model /path/to/qwen-4.05bpw \
  --output /path/to/cache/qwen.strata-experts.json
./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw --cpu on \
  --experts /path/to/cache/qwen.strata-experts.json

# Or rebuild into RAM at every start. No packed disk files.
./run-peer-4bpw.sh --model-dir /path/to/qwen-4.05bpw --cpu on --repack-in-memory
```

Select the descriptor file, not the payload or fingerprint directory. The source
model is still required. `validate DESCRIPTOR --model MODEL` checks the selection.
Disk repacking needs extra disk space; RAM repacking pays the conversion cost at
every startup, so warn the user about longer launches.

Packed CPU inference needs AVX-512BW; packed GPU support is validated on Linux/HIP
gfx11. The validated Qwen composition is **one GPU + CPU**. Packed Qwen + peer has
an unresolved numerical divergence and is refused by the launch layer unless the
user explicitly requests the experiment. Do not bypass the guard for convenience.
Native two-GPU Qwen is the recommendation. A selected Qwen pack requires resident
`all`; an explicit conflicting residency must be reported rather than overwritten.

## MiMo

Use `run-mimo-mopd.sh --model-dir DIR --cpu on`. Repacking still needs an explicit
descriptor or `--repack-in-memory`. CPU help is recommended even with a peer.
The two device-result transports are experimental launcher defaults:
`STRATA_MIMO_WINDOW_DEVICE_MISSES=1`, and
`STRATA_MIMO_PEER_DEVICE_RESULTS=1` when a peer is present. Their `=0` overrides
work. They do not activate Qwen paths. No universal speedup or fault-recovery
guarantee is claimed.

## Verify and hand over

Wait for the actual loaded/ready state. A large pinned arena can take minutes;
do not restart repeatedly just because startup is quiet. Verify:

```sh
curl http://127.0.0.1:8085/v1/models
curl http://127.0.0.1:8085/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"Say hello."}],"max_tokens":64,"reasoning_effort":"none"}'
```

With an API key, add the bearer header. Give the user the browser URL, API base
URL, exact launch command, log location and how to stop the process. Report what
you did not test. For benchmarks, keep both first and second runs, the actual
prompt/decode counts, cache reuse, output lengths, exact model/build and hardware.
Do not call prefix-cache reuse a fresh 32K prefill, or a changed continuation a
byte-identical speed test.
