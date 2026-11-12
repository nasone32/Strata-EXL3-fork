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
  page of kernel environment variables to a user's command.
- Two GPUs: add `--gpu 0,1 --peer-device 1`. The first visible GPU is primary,
  `--peer-device` is an index in that visible list. This is peer expert offload,
  not layer splitting. CPU help defaults off for Qwen with a peer; `--cpu on`
  explicitly requests it.
- Low RAM: native Qwen can use `--resident 0`. **Qwen does not implement
  `--resident peer`**; it must refuse, not be described as a fallback. MiMo with a
  peer can use that RAM-complement mode. RAM-only repacking requires `all`.
- Tune `--expert-cache`, `--peer-slots`, VRAM reserves, `--max-context`,
  `--prefill` and CPU threads for the machine. Do not copy the developer's
  165/104-slot, 10200/7200-peer-slot or 250K-context settings onto smaller cards.
- CPU worker count: `--cpu-threads`; auxiliary affinity: `--aux-cpus` or
  `STRATA_AUX_CPUS`. Leave affinity unset initially. CPU numbering varies.
- MTP is optional: `--mtp-dir DIR --spec 4`. Do not enable it without the
  matching drafter. MiMo uses `--drafter DIR` for DFlash instead.
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
