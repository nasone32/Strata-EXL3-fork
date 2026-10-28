# Set up Strata EXL3 on AMD Linux — instructions for an AI agent

This is the EXL3 entry point for the **Strata-EXL3-fork AMD community-test release**. Follow this file when a user asks you to install or run EXL3. Do not substitute `setup.py` or `docs/AI_SETUP.md`: those still describe the upstream GGUF installation flow.

## Scope and safety

- Tested fork configuration: Linux, ROCm core-7.14, Ryzen 9 7950X, two RX 7900 XTX cards (`gfx1100`), Qwen3.8-Flash-Next EXL3. Other RDNA3 cards need measurements. RDNA4 (`gfx1200`, `gfx1201`) needs EXL3 testing; accepting a flag does not mean its optimized kernel ran.
- Windows EXL3, NVIDIA/CUDA, Metal and SYCL are not validated by this fork. Do not present these instructions as support for them.
- Inspect before installing. Do not replace a working driver, install system packages, reboot, stop another server, or download large weights without the user's permission.
- Never delete an existing environment, checkout, model or cache to make space. Create a separate environment/build directory if necessary.
- Bind the server to `127.0.0.1`. Do not add a non-local `--host` without explicit permission and an API key. Do not publish logs containing API keys or user conversations.
- Preserve VRAM for the desktop GPU. Do not stop its compositor or spend its whole VRAM budget.
- This file gives a starting configuration, not a guarantee that a particular model fits every Radeon card. If the hardware is insufficient, report the limiting resource rather than forcing allocations.

Deliver a working, verified configuration when possible. If blocked, state the exact error, what succeeded, and what requires the user's decision.

## 1. Read the relevant files

From the repository root, read:

1. `README.md` — fork support scope and portable launchers.
2. `AGENTS.md` — project rules.
3. `docs/EXL3.md` — model format and kernel constraints.
4. `docs/AMD_HIP.md` — ROCm build and device access.
5. `docs/EXL3-RELEASE-VALIDATION.md` — measured coverage and known limits.

Resolve paths from this checkout. Never copy the developer's home directory, disk paths or CPU numbers into a new user's configuration.

## 2. Inspect the machine

Run read-only checks such as:

```sh
uname -a
lscpu
free -h
lsblk -o NAME,SIZE,ROTA,TRAN,MODEL,MOUNTPOINTS
command -v rocminfo
command -v rocm-smi
command -v cmake
command -v ninja
command -v python3
rocminfo
rocm-smi --showproductname --showmeminfo vram --showpids
```

Inspect the outputs; commands may be absent on a fresh installation. Determine:

- GPU architecture, number of cards and free VRAM, not just installed VRAM.
- Which GPU drives the desktop and whether other processes own GPU memory.
- CPU physical/logical cores, allowed CPU set and RAM available.
- Model disk type and space. Use a local SSD/NVMe; a network mount or rotating disk can be unsuitable for expert and PLE reads.
- Installed ROCm version, development compiler and library locations.
- `/dev/kfd` and render-node permissions. If permissions are missing, explain the appropriate `render`/`video` group changes from `docs/AMD_HIP.md`; obtain permission before applying them, and explain that a new login may be needed.

Model files can be much larger than application memory usage suggests. Experts are served through reclaimable RAM file cache; dense weights are uploaded to VRAM; the recommended n-gram IO bypasses file cache. A cold cache still requires SSD reads. Do not describe the whole model file size as required committed RAM, or promise that cache pressure cannot hurt inference speed.

## 3. Obtain the source and dependencies

If the checkout is already present, inspect it rather than cloning over it. Otherwise clone the URL supplied by the user into a new directory. Do not guess the fork URL.

Use Python 3.10 or newer. In the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r tools/exl3/requirements.txt
```

These install packages from external registries; obtain permission if dependency installation was not already authorized. If `.venv` exists, inspect and reuse it when compatible. Do not overwrite a symlink or an unfamiliar environment.

Build requirements: CMake 3.24+, a C++20 compiler, Git, and ROCm development packages including the HIP compiler and hipBLAS. Ninja is used in the example below. Package names vary by distribution; consult the installed environment and `docs/AMD_HIP.md` instead of blindly pasting a distribution-specific command.

A normal configure fetches a pinned llama.cpp dependency. For offline builds, supply `-DSTRATA_GGML_DIR=/absolute/path/to/compatible/llama.cpp`; check its revision against the project pin. Do not point at an arbitrary newer checkout and claim equivalent validation.

## 4. Build for the actual GPU

Choose the architecture from `rocminfo`, not from the developer's example:

- RX 7900 XT/XTX: `gfx1100`.
- Other RDNA3: use the actual reported architecture, subject to the supported list in `cmake/hip_backend.cmake`.
- RDNA4: `gfx1200` or `gfx1201`, as reported. This is a community-test build, not a validated EXL3 performance profile.

Example for gfx1100, from the repository root:

```sh
cmake -S . -B build-hip-exl3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip-exl3 --target strata -j2
```

Replace `gfx1100` appropriately. A multi-target build uses a quoted semicolon list, for example `-DCMAKE_HIP_ARCHITECTURES='gfx1100;gfx1201'`; it still needs correctness testing on each device. Limit build parallelism according to available RAM.

If ROCm discovery fails, inspect the installation before retrying. Valid overrides include:

```text
-DCMAKE_HIP_COMPILER=/absolute/path/to/rocm/llvm/bin/clang++
-DCMAKE_PREFIX_PATH=/absolute/path/to/rocm
```

Check that the engine's runtime libraries come from the installation used to build it. The launchers do not populate `lib_dirs`. If needed, export `LD_LIBRARY_PATH` with the actual ROCm library directories. Do not hardcode `core-7.14` on a machine that uses another layout. `ldd build-hip-exl3/strata` must not report missing libraries.

The upstream setup wizard may download an upstream binary without this fork's EXL3 changes. Do not use that binary as a substitute for the fork build.

## 5. Obtain and verify an EXL3 model

**Model acquisition is separate from setup.** The EXL3 launchers only open a local model folder; they do not download models. Do not run the upstream `setup.py` model downloader for this flow. If the user already downloaded a compatible model, inspect and reuse that folder without downloading another copy.

The two Qwen3.8-Flash-Next EXL3 branches tested during this fork's development are:

- [3.05bpw_h5_ng5](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5) — select `run-peer.sh`.
- [4.05bpw_h6_ng6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6) — select `run-peer-4bpw.sh`.

Advise the user to download one complete branch into a folder on their SSD, then supply that folder to the launcher with `--model-dir /absolute/path/to/downloaded-model` (or `STRATA_EXL3_MODEL`). Keep the bitrates in separate folders. Confirm the selected revision and its current contents; agree on storage location and download size. Do not start a download unless the user requests or authorizes it, and do not download both bitrates by default. A user-authorized download may use their preferred HuggingFace client; it must not silently choose another model.

Keep the complete compatible export, including:

- `config.json` and quantization metadata;
- `model.safetensors.index.json`;
- every weight shard referenced by that index;
- the matching n-gram table, normally `ngram_embedding.safetensors` (the launchers also recognize `gram_embedding.safetensors`);
- original tokenizer files, including `tokenizer.json`, `merges.txt`, and any chat template;
- other export files required by the selected revision.

Do not mix the index, weight shards or PLE table from different revisions. Do not pass a GGUF file to `--exl3`. A directory called “4bpw” is not proof of its contents. The loader validates per-tensor integer bitrates; average bitrate labels have a different meaning.

MTP is optional. Begin without it. An export's `mtp_hyper_connection_mixer_patch.safetensors` alone is not a complete Strata MTP drafter. Add `--mtp-dir` only when the user has a separate compatible drafter in the engine's expected format.

## 6. Prepare the tokenizer

The server expects Strata's pack-layout tokenizer. If a complete compatible pack tokenizer already exists, inspect and reuse it. Otherwise, with the virtual environment active:

```sh
python tools/exl3/tokenizer_to_pack.py \
  --hf /absolute/path/to/exl3-model \
  --out /absolute/path/to/new-tokenizer-pack --check
```

The generated tokenizer directory is **`/absolute/path/to/new-tokenizer-pack/tokenizer`**, not its parent. Pass that child directory as `--tokenizer`. Conversion must complete its round-trip checks. Keep the model's chat template when provided.

Do not install Torch, Transformers or ExLlamaV3 merely to perform this conversion; the converter uses the repository tokenizer code and `regex`.

## 7. Start conservatively; do not blindly use the tuned defaults

The launchers contain developer-sized context and cache defaults. These are not automatic hardware profiles. **Always specify `--gpu` and a deliberate peer choice**: multiple visible GPUs without `--peer-device` can make the server request automatic layer splitting, which the EXL3 path rejects.

Start with a small context, a small prompt chunk, no MTP and CPU affinity unset. Prefer `--expert-cache auto` for initial primary-cache sizing. It means EXL3 slots per layer, not GGUF's global expert-slot count. Automatic sizing is still bounded by later allocations; inspect startup errors and free VRAM.

Activate `.venv` so both launcher configuration generation (`python3`) and the server use the intended environment.

### Single GPU

Use `run-peer.sh` for a 3 bpw export or `run-peer-4bpw.sh` for a 4 bpw export; the latter uses a different API model label. Example initial 3 bpw configuration:

```sh
STRATA_EXL3_PEER=0 STRATA_EXL3_PEER_DEVICE= \
./run-peer.sh \
  --model-dir /absolute/path/to/exl3-model \
  --tokenizer /absolute/path/to/tokenizer-pack/tokenizer \
  --gpu 0 --expert-cache auto \
  --max-context 4096 --prefill 512 --reserve 3072 \
  --port 8095 --dry-run
```

Inspect the generated JSON. It must use this checkout's engine, the intended model/tokenizer, PLE `direct`, the bundled expert profile, one visible GPU and no peer-device argument. Then rerun the same command without `--dry-run`. This is a starting point, not a guarantee for small-VRAM cards.

### Two GPUs: peer expert tier

```sh
STRATA_EXL3_PEER=1 \
./run-peer.sh \
  --model-dir /absolute/path/to/exl3-model \
  --tokenizer /absolute/path/to/tokenizer-pack/tokenizer \
  --gpu 0,1 --peer-device 1 --peer-slots 1024 \
  --expert-cache auto --max-context 4096 --prefill 512 \
  --reserve 3072 --peer-reserve 4096 \
  --port 8095 --dry-run
```

Here `1` is the peer's ordinal **within the visible GPU list**. Verify actual device mapping. Start with the small peer cap shown above, confirm correctness, then increase it against the reported free-VRAM budget. Preserve the desktop reserve; do not copy 10,200 or 7,200 peer slots onto an arbitrary smaller GPU.

Use `--expert-profile /path/to/profile.bin` only for a compatible profile. The default is the bundled `data/expert-profile.bin`; `--no-expert-profile` is an explicit diagnostic arm, not required for normal startup.

### RDNA settings and opt-outs

The engine's measured architecture-independent optimizations are default-on. The portable launchers additionally suggest RDNA-specific settings. Explicit environment values, including `0`, override the suggestions.

For an initial portable-kernel diagnostic, prepend:

```sh
STRATA_EXL3_RDNA=0 STRATA_EXL3_FUSE=0 STRATA_EXL3_WMMA=0 \
STRATA_WMMA_GEMM=0 STRATA_HIP_WMMA=0 \
STRATA_EXL3_PRIMARY_MULTIROW=0 STRATA_EXL3_PEER_MULTIROW=0 \
STRATA_EXL3_EXPERT_BATCH=0 STRATA_EXL3_PEER_BATCH=0 \
STRATA_EXL3_SMALL_M=0
```

These assignments must precede the actual launcher command in the same shell command. On RDNA4, establish this baseline first, then test the suggested profile and record which paths actually activated. Do not remove architecture guards just to make a flag appear active. A mixed-target RDNA build still needs device-specific validation.

Leave `STRATA_AUX_CPUS` unset. The engine no longer silently selects `auto`. For later CPU tuning compare `off`, `auto` and topology-aware lists on that machine. The developer's `20-23` is not a universal 16-core setting.

## 8. Verify the running service, not just the process

Watch startup logs until the engine is ready. Check for missing libraries, unsupported device targets, upload/slot verification failures, refused features, or memory exhaustion.

On the explicitly selected port:

```sh
curl --fail --silent http://127.0.0.1:8095/health
curl --fail --silent http://127.0.0.1:8095/v1/models
```

Health returning HTTP 200 with `status: ok` is not sufficient: require **`loaded: true`** and a real completion. Use a model ID returned by `/v1/models`, rather than assuming the label. Example for the 3 bpw launcher:

```sh
curl --fail --silent http://127.0.0.1:8095/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-exl3-3bpw","messages":[{"role":"user","content":"Write a short bubblesort function in C."}],"max_tokens":512,"temperature":0,"stream":false}'
```

Check that the response is meaningful, not empty, repetitive or corrupted. Open `http://127.0.0.1:8095/` if a browser is appropriate. Repeat a request to distinguish cold loading from a warmed session. Keep the startup and completion logs locally.

If the user wants client integration, give the localhost OpenAI base URL `http://127.0.0.1:8095/v1`, the discovered model ID, and the API key if they configured one. Do not silently change ports or bind addresses to make another application connect.

## 9. Test and tune only after correctness

A full CTest run requires building test executables, not merely `--target strata`. For a focused check:

```sh
cmake --build build-hip-exl3 --target \
  exl3_flags_test exl3_route_plan_test peer_adapt_plan_test \
  exl3_moe_multi_parity exl3_peer_test -j2
ctest --test-dir build-hip-exl3 \
  -R '^(exl3_flags_test|exl3_route_plan_test|peer_adapt_plan_test|exl3_moe_multi_parity|exl3_peer_test)$' \
  --output-on-failure
```

Run tests only when doing so will not contend with the user's server or another GPU workload. Some tests require appropriate model fixtures; report skips and failures honestly. Exclude an unsupported integrated GPU deliberately through the visible-device list, not by hiding failures on a supported test card.

Increase context, prompt chunk and cache budgets incrementally, rechecking free VRAM and output after each change. Add a compatible MTP drafter only after the non-MTP path works. Save the final configuration outside temporary launcher files.

For community performance comparisons:

- Same model revision, GPU configuration, prompt, sampling settings and output limit.
- Measure cold and warm requests separately; never attribute warmed-cache gains to a flag.
- Compare a same-day baseline with a same-day candidate, preferably alternating order.
- Record exact flags and whether optimized paths activated. Compare output IDs for changes claiming bit identity. Different quantizations are not an exact-output oracle for each other.
- `tools/exl3/bench.py` starts its own engine; it is not an HTTP health check. Do not run it alongside the user's server. It has large defaults: pass explicit small prompt/output limits for an initial test.
- Never clear the system-wide file cache as an unannounced setup step.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing `libhipblas*` / ROCm library | Check `ldd` and library directories from the build's ROCm installation. Do not install a second ROCm blindly. |
| `/dev/kfd` permission denied | Check render/video membership and session permissions; explain the required permission change and new login. |
| Unsupported architecture / invalid device function | Confirm actual GPU target and rebuild for it. Do not force gfx1100 on RDNA4. |
| Where does `--layer-split auto` put the split with EXL3? | By layer count: stage i starts at layer n_layers*i/stages (two GPUs: half each); the startup log names K. Pass explicit K to place it by hand. |
| VRAM allocation failure | Confirm competing processes; reduce context/prefill/explicit cache or peer cap. Preserve desktop headroom. Do not stop someone else's process. |
| Very slow first prompt | Inspect model IO, cold expert pages and PLE mode. Keep PLE direct; compare warmed requests before concluding a kernel regression. |
| Empty/incoherent output | Verify model revision, indexed shards, matching PLE and tokenizer. Stop tuning speed until correctness is established. |
| An RDNA flag appears ineffective | Inspect activation/fallback logs and architecture gates. Treat RDNA4 support as a test request, not an established feature. |

## Final handoff to the user

Provide:

1. Checkout, engine, model and tokenizer paths; build architecture and ROCm version.
2. A complete reusable launch command and the chosen reserves, cache sizes, context and peer mapping.
3. Health/model-list results and a coherent completion check.
4. Cold versus warm timing, if measured, with the exact settings.
5. What was not tested, errors/skips, and any remaining user decisions.

For a Reddit/community report, collect hardware, model revision, build revision, exact launch settings, redacted startup logs, correctness observations and measured timings. Label the result RDNA3 or RDNA4. See `docs/COMMUNITY_BENCHMARKS.md` and `docs/EXL3-RELEASE-VALIDATION.md`.
