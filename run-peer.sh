#!/usr/bin/env bash
# Strata-EXL3-fork: the measured 3 bpw EXL3 peer preset (local web chat + OpenAI/Anthropic API).
#
# Usage: ./run-peer.sh --model-dir /path/to/Qwen3.8-Flash-Next-exl3 [other serve/server.py options]
#
# Everything is configurable by flag or environment variable; nothing is hard-coded to one
# machine (see the README). Explicit values always win, including an explicit 0.
#   --model-dir      DIR    (or STRATA_EXL3_MODEL)        the EXL3 model directory, required
#   --ple-file       FILE   (or STRATA_EXL3_PLE_FILE)     the n-gram PLE table (default: detected
#                                                        in <model-dir>, ngram_embedding.safetensors
#                                                        or gram_embedding.safetensors; required:
#                                                        the model needs its PLE table)
#   --tokenizer      DIR    (or STRATA_EXL3_TOKENIZER)    pack-layout tokenizer; detected in
#                                                        <model-dir>/tokenizer or <model-dir> when
#                                                        omitted, else required
#   --engine-bin     PATH   (or STRATA_EXL3_ENGINE)       default: build-hip-exl3/strata here
#   --expert-profile FILE   (or STRATA_EXL3_EXPERT_PROFILE)  default: this checkout's
#                                                        data/expert-profile.bin; --no-expert-profile omits
#   --peer-device    N      (or STRATA_EXL3_PEER_DEVICE)  second GPU as the peer expert tier
#   --peer-slots     N      (or STRATA_EXL3_PEER_SLOTS)   expert slots on the peer
#   --expert-cache   N      (or STRATA_EXL3_EXPERT_CACHE) expert slots per layer on the primary
#   --max-context    N      (or STRATA_EXL3_MAX_CONTEXT)  KV capacity in tokens
#   --prefill        N      (or STRATA_EXL3_PREFILL)      batched prompt chunk size
#   --reserve        MiB    (or STRATA_EXL3_RESERVE_MIB)  VRAM left free on the primary
#   --peer-reserve   MiB    (or STRATA_EXL3_PEER_RESERVE_MIB)  VRAM left free on the peer
#   --aux-cpus       LIST   (or STRATA_AUX_CPUS)          non-pool thread CPUs (LIST|auto|off);
#                                                        default: unset, the engine's own choice
#   --mtp-dir        DIR    (or STRATA_EXL3_MTP_DIR)      optional MTP drafter directory
#   --spec           N      (or STRATA_EXL3_SPEC)         MTP verify window (with --mtp-dir; default 4)
#   --spec-min-p     P      (or STRATA_EXL3_SPEC_MIN_P)   MTP window extend threshold (default 0.5)
#   --gpu            LIST   (or STRATA_EXL3_GPUS)         cards, e.g. 0 or 0,1 (default 0,1)
#   --dry-run                                        print the config JSON and exit: no server,
#                                                    no GPU, nothing started
#
# The RDNA3 (gfx1100) profile below is a suggestion: every STRATA_* switch stays opt-in in the
# engine, and an exported value (including 0) replaces the suggestion. RDNA4 (gfx12) is untested.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"

# ---- preset defaults (measured 3 bpw peer tier; override by flag or environment) ----
ENGINE_DEFAULT="$ROOT/build-hip-exl3/strata"
EXPERT_PROFILE_DEFAULT="$ROOT/data/expert-profile.bin"
EXPERT_CACHE_DEFAULT=165          # expert slots per layer on the primary (measured preset)
PEER_SLOTS_DEFAULT=10200         # expert slots on the peer card
MAX_CONTEXT_DEFAULT=65536
PREFILL_DEFAULT=8192
RESERVE_DEFAULT=2048            # MiB left free on the primary
PEER_RESERVE_DEFAULT=4096       # MiB left free on the peer
PEER_DEVICE_DEFAULT=            # empty: no peer tier unless asked for (engine default 0)
GPUS_DEFAULT=0
SPEC_DEFAULT=4
SPEC_MIN_P_DEFAULT=0.5

# ---- the suggested RDNA3 (gfx1100) profile: engine switches, all opt-in in the engine ----
# Measured on the developer's RX 7900 XTX (gfx1100), ROCm 7.14; RDNA4 (gfx12) needs testing.
export STRATA_EXL3_RDNA="${STRATA_EXL3_RDNA:-1}"
export STRATA_EXL3_FUSE="${STRATA_EXL3_FUSE:-3}"
export STRATA_EXL3_WMMA="${STRATA_EXL3_WMMA:-1}"
export STRATA_WMMA_GEMM="${STRATA_WMMA_GEMM:-1}"
export STRATA_HIP_WMMA="${STRATA_HIP_WMMA:-1}"
export STRATA_EXL3_PRIMARY_MULTIROW="${STRATA_EXL3_PRIMARY_MULTIROW:-2}"
export STRATA_EXL3_PEER_MULTIROW="${STRATA_EXL3_PEER_MULTIROW:-4}"
export STRATA_EXL3_EXPERT_BATCH="${STRATA_EXL3_EXPERT_BATCH:-1}"
export STRATA_EXL3_PEER_BATCH="${STRATA_EXL3_PEER_BATCH:-1}"
export STRATA_EXL3_PEER_PREFILL="${STRATA_EXL3_PEER_PREFILL:-1}"
# The arch-neutral EXL3 switches the engine turns on by default with --exl3 are stated
# explicitly too, so the printed config is reproducible; an exported value (0 included) wins.
export STRATA_EXL3_OVERLAP="${STRATA_EXL3_OVERLAP:-1}"
export STRATA_EXL3_NO_PREAD="${STRATA_EXL3_NO_PREAD:-1}"
export STRATA_EXL3_PACK_STAGE="${STRATA_EXL3_PACK_STAGE:-1}"
export STRATA_EXL3_WAVESYNC="${STRATA_EXL3_WAVESYNC:-1}"
export STRATA_EXL3_PIPE="${STRATA_EXL3_PIPE:-1}"
export STRATA_EXL3_PREFILL_BORROW="${STRATA_EXL3_PREFILL_BORROW:-1}"
export STRATA_EXL3_PEER_PADDED="${STRATA_EXL3_PEER_PADDED:-1}"
export STRATA_EXL3_PEER_SPARSE="${STRATA_EXL3_PEER_SPARSE:-1}"
export STRATA_EXL3_PEER_CONVERT="${STRATA_EXL3_PEER_CONVERT:-1}"
export STRATA_EXL3_PEER_PACK_PLAN="${STRATA_EXL3_PEER_PACK_PLAN:-1}"
export STRATA_EXL3_PRIMARY_PACK_PLAN="${STRATA_EXL3_PRIMARY_PACK_PLAN:-1}"
export STRATA_EXL3_STAGE_INLINE="${STRATA_EXL3_STAGE_INLINE:-2}"
# The shared GGUF-era switches (STRATA_QFUSE, STRATA_MMVQ_BUNCH, STRATA_PF_GEMM) are NOT set
# here: with --exl3 the engine defaults them to 1 itself (STRATA_EXL3_QFUSE / _MMVQ_BUNCH /
# _PF_GEMM or the plain names override, 0 included); an exported value of yours is
# forwarded as given (see the README).
# STRATA_AUX_CPUS is NOT set here: the engine's own default applies, and an exported value
# (LIST, auto, off or 0) is forwarded as given, without a machine-specific list baked in.

# ---- flags (explicit beats environment beats preset) ----
MODEL_DIR="${STRATA_EXL3_MODEL:-}"
PLE_FILE="${STRATA_EXL3_PLE_FILE:-}"
TOKENIZER_DIR="${STRATA_EXL3_TOKENIZER:-}"
ENGINE_BIN="${STRATA_EXL3_ENGINE:-$ENGINE_DEFAULT}"
EXPERT_PROFILE="${STRATA_EXL3_EXPERT_PROFILE:-$EXPERT_PROFILE_DEFAULT}"
PEER_DEVICE="${STRATA_EXL3_PEER_DEVICE:-$PEER_DEVICE_DEFAULT}"
PEER_SLOTS="${STRATA_EXL3_PEER_SLOTS:-$PEER_SLOTS_DEFAULT}"
EXPERT_CACHE="${STRATA_EXL3_EXPERT_CACHE:-$EXPERT_CACHE_DEFAULT}"
MAX_CONTEXT="${STRATA_EXL3_MAX_CONTEXT:-$MAX_CONTEXT_DEFAULT}"
PREFILL="${STRATA_EXL3_PREFILL:-$PREFILL_DEFAULT}"
RESERVE_MIB="${STRATA_EXL3_RESERVE_MIB:-$RESERVE_DEFAULT}"
PEER_RESERVE_MIB="${STRATA_EXL3_PEER_RESERVE_MIB:-$PEER_RESERVE_DEFAULT}"
AUX_CPUS="${STRATA_AUX_CPUS:-}"       # empty = unset: the engine's own choice
MTP_DIR="${STRATA_EXL3_MTP_DIR:-}"
GPUS="${STRATA_EXL3_GPUS:-$GPUS_DEFAULT}"
SPEC="${STRATA_EXL3_SPEC:-$SPEC_DEFAULT}"
SPEC_MIN_P="${STRATA_EXL3_SPEC_MIN_P:-$SPEC_MIN_P_DEFAULT}"
NO_EXPERT_PROFILE=0
DRY_RUN=0
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --ple-file) PLE_FILE="$2"; shift 2 ;;
        --tokenizer) TOKENIZER_DIR="$2"; shift 2 ;;
        --engine-bin) ENGINE_BIN="$2"; shift 2 ;;
        --expert-profile) EXPERT_PROFILE="$2"; NO_EXPERT_PROFILE=0; shift 2 ;;
        --no-expert-profile) NO_EXPERT_PROFILE=1; shift ;;
        --peer-device) PEER_DEVICE="$2"; shift 2 ;;
        --peer-slots) PEER_SLOTS="$2"; shift 2 ;;
        --expert-cache) EXPERT_CACHE="$2"; shift 2 ;;
        --max-context) MAX_CONTEXT="$2"; shift 2 ;;
        --prefill) PREFILL="$2"; shift 2 ;;
        --reserve) RESERVE_MIB="$2"; shift 2 ;;
        --peer-reserve) PEER_RESERVE_MIB="$2"; shift 2 ;;
        --aux-cpus) AUX_CPUS="$2"; shift 2 ;;       # an explicit flag value (0 included) is forwarded
        --mtp-dir) MTP_DIR="$2"; shift 2 ;;
        --gpu) GPUS="$2"; shift 2 ;;
        --spec) SPEC="$2"; shift 2 ;;
        --spec-min-p) SPEC_MIN_P="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) SERVER_ARGS+=("$1"); shift ;;            # anything else: serve/server.py's own options
    esac
done

usage_error() { printf 'run-peer.sh: %s\nSee the header of run-peer.sh or the README.\n' "$1" >&2; exit 2; }

[[ -n "$MODEL_DIR" ]] || usage_error "--model-dir (or STRATA_EXL3_MODEL) is required: the EXL3 model directory"
[[ -d "$MODEL_DIR" ]] || usage_error "--model-dir $MODEL_DIR is not a directory"

# --tokenizer is optional: a pack layout tokenizer inside the model directory is detected.
if [[ -z "$TOKENIZER_DIR" ]]; then
    if [[ -f "$MODEL_DIR/tokenizer/vocab.json" ]]; then
        TOKENIZER_DIR="$MODEL_DIR/tokenizer"
    elif [[ -f "$MODEL_DIR/vocab.json" ]]; then
        TOKENIZER_DIR="$MODEL_DIR"                 # a model dir that already carries the pack layout
    else
        usage_error "no pack-layout tokenizer in $MODEL_DIR (tokenizer/vocab.json): pass --tokenizer, or make one with python3 tools/exl3/tokenizer_to_pack.py --hf <model-dir> --out <pack-dir>"
    fi
fi
[[ -f "${TOKENIZER_DIR%/}/vocab.json" ]] || usage_error "no vocab.json under --tokenizer $TOKENIZER_DIR (pack layout: vocab.json, merges.txt, token_type.json; convert an HF tokenizer with tools/exl3/tokenizer_to_pack.py)"

# The n-gram PLE table is required: detect it beside the weights, or take --ple-file.
if [[ -z "$PLE_FILE" ]]; then
    for cand in ngram_embedding.safetensors gram_embedding.safetensors; do
        if [[ -f "$MODEL_DIR/$cand" ]]; then PLE_FILE="$MODEL_DIR/$cand"; break; fi
    done
fi
[[ -n "$PLE_FILE" ]] || usage_error "no ngram_embedding.safetensors in $MODEL_DIR: pass --ple-file (the model's n-gram PLE table is required; see docs/EXL3.md)"
[[ -f "$PLE_FILE" ]] || usage_error "the n-gram PLE table is not a file: $PLE_FILE (see docs/EXL3.md)"

# The bundled expert profile (measured preset) by default; an explicit path wins.
if [[ "$NO_EXPERT_PROFILE" == 1 ]]; then
    EXPERT_PROFILE=""
elif [[ -z "$EXPERT_PROFILE" ]]; then
    :
elif [[ ! -f "$EXPERT_PROFILE" ]]; then
    usage_error "expert profile not found: $EXPERT_PROFILE (pass --expert-profile PATH, or --no-expert-profile)"
fi

# The peer tier is opt-in in the engine (STRATA_EXL3_PEER): state it explicitly with
# --peer-device, so the config is reproducible; an exported value (0 included) wins.
if [[ -n "$PEER_DEVICE" ]]; then
    export STRATA_EXL3_PEER="${STRATA_EXL3_PEER:-1}"
fi

# ---- the engine's argument vector ----
ENGINE_ARGS=("--exl3" "$MODEL_DIR" "--ple-exl3" "$PLE_FILE"
             "--expert-cache" "$EXPERT_CACHE"
             "--prefill" "$PREFILL"
             "--max-context" "$MAX_CONTEXT"
             "--kv" "int8"
             "--ple-io" "direct"
             "--vram-reserve-mib" "$RESERVE_MIB")
if [[ -n "$EXPERT_PROFILE" ]]; then
    ENGINE_ARGS+=("--expert-profile" "$EXPERT_PROFILE")
fi
if [[ -n "$PEER_DEVICE" ]]; then
    ENGINE_ARGS+=("--peer-device" "$PEER_DEVICE"
                  "--peer-slots" "$PEER_SLOTS"
                  "--peer-reserve-mib" "$PEER_RESERVE_MIB")
fi
# --spec is the MTP drafter's window: only with a drafter, or the engine turns the window off.
if [[ -n "$MTP_DIR" ]]; then
    [[ -d "$MTP_DIR" ]] || usage_error "--mtp-dir $MTP_DIR is not a directory (the MTP drafter; see docs/DETAILS.md)"
    ENGINE_ARGS+=("--mtp" "$MTP_DIR" "--spec" "$SPEC" "--spec-min-p" "$SPEC_MIN_P")
fi

# ---- the config (one python call; the engine's env carries the RDNA profile and the cards) ----
CONFIG_JSON="$(python3 - "$ROOT" "$MODEL_DIR" "$TOKENIZER_DIR" "$ENGINE_BIN" "$PEER_DEVICE" \
                        "$PEER_SLOTS" "$EXPERT_CACHE" "$MAX_CONTEXT" "$PREFILL" "$RESERVE_MIB" \
                        "$PEER_RESERVE_MIB" "$AUX_CPUS" "$MTP_DIR" "$GPUS" "$PLE_FILE" \
                        "$EXPERT_PROFILE" "${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}" <<'PY'
import json, os, sys
(root, model_dir, tokenizer_dir, engine_bin, peer_device, peer_slots, expert_cache,
 max_context, prefill, reserve_mib, peer_reserve_mib, aux_cpus, mtp_dir, gpus,
 ple_file, expert_profile) = sys.argv[1:17]
engine_args = sys.argv[17:]
env = {}
# The RDNA profile (exported by the launcher, so present) and the shared GGUF-era switches
# (present only when YOU exported them): an explicit value, 0 included, is forwarded as given.
for k in ("STRATA_EXL3_RDNA", "STRATA_EXL3_FUSE", "STRATA_EXL3_WMMA", "STRATA_WMMA_GEMM",
          "STRATA_HIP_WMMA", "STRATA_EXL3_PRIMARY_MULTIROW", "STRATA_EXL3_PEER_MULTIROW",
          "STRATA_EXL3_EXPERT_BATCH", "STRATA_EXL3_PEER_BATCH", "STRATA_EXL3_PEER_PREFILL",
          "STRATA_EXL3_OVERLAP", "STRATA_EXL3_NO_PREAD", "STRATA_EXL3_PACK_STAGE",
          "STRATA_EXL3_WAVESYNC", "STRATA_EXL3_PIPE", "STRATA_EXL3_PREFILL_BORROW",
          "STRATA_EXL3_PEER", "STRATA_EXL3_PEER_PADDED", "STRATA_EXL3_PEER_SPARSE",
          "STRATA_EXL3_PEER_CONVERT", "STRATA_EXL3_PEER_PACK_PLAN", "STRATA_EXL3_PRIMARY_PACK_PLAN",
          "STRATA_EXL3_STAGE_INLINE", "STRATA_EXL3_PEER_ADAPT",
          "STRATA_EXL3_QFUSE", "STRATA_EXL3_MMVQ_BUNCH", "STRATA_EXL3_PF_GEMM",
          "STRATA_QFUSE", "STRATA_MMVQ_BUNCH", "STRATA_PF_GEMM"):
    if k in os.environ:
        env[k] = os.environ[k]
if aux_cpus:                      # empty = unset: the engine's own choice; explicit values forwarded
    env["STRATA_AUX_CPUS"] = aux_cpus
if gpus:
    env["HIP_VISIBLE_DEVICES"] = gpus
cfg = {
    "backend": "hip",
    "cwd": root,
    "exe": engine_bin,
    "args": engine_args,
    "env": env,
    "model_name": "qwen3.8-flash-next-exl3-3bpw",
    "tokenizer": tokenizer_dir,
}
if gpus:
    cfg["gpu"] = [int(x) for x in gpus.split(",")]
print(json.dumps(cfg, indent=2))
PY
)"

if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$CONFIG_JSON"
    printf '\n[dry-run] no server started, no GPU touched. Engine: %s\n' "$ENGINE_BIN"
    exit 0
fi

[[ -f "$ENGINE_BIN" ]] || { printf 'engine binary missing: %s\nBuild it first:\n  cmake -B build-hip-exl3 -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100\n  cmake --build build-hip-exl3 --target strata -j"$(nproc)"\nOr point --engine-bin / STRATA_EXL3_ENGINE at one.\n' "$ENGINE_BIN" >&2; exit 1; }
PYTHON="${STRATA_EXL3_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || { printf 'python not found (%s)\n' "$PYTHON" >&2; exit 1; }

CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/strata-peer.XXXXXX")"
trap 'rm -f -- "$CONFIG_FILE"' EXIT
printf '%s\n' "$CONFIG_JSON" > "$CONFIG_FILE"

"$PYTHON" "$ROOT/serve/server.py" --engine strata --config "$CONFIG_FILE" --host 127.0.0.1 "${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"}"