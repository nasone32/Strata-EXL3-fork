#!/usr/bin/env bash
# Portable demo launcher: Qwen3.8-Flash-Next EXL3 at 3.05 bpw on two GPUs,
# full pinned-RAM expert residency (--resident all) with automatic expert-cache
# sizing on both cards (no fixed slot counts). Part of the exl3-2026.11.21
# release; validated on Linux/HIP gfx1100 (two RX 7900 XTX, Ryzen 9 7950X,
# 96 GB RAM). See examples/README.md and docs/RELEASE_EXL3_2026-11-21.md.
#
# Nothing in this script is machine-specific. Bring your own paths:
#   export STRATA_EXL3_MODEL=/path/to/Qwen3.8-NEXT-exl3-3bpw   # required (or pass --model-dir DIR)
#   export STRATA_EXL3_TOKENIZER=/path/to/pack/tokenizer       # optional; auto-detected in the model dir
#   export STRATA_EXL3_MTP_DIR=/path/to/mtp/rt                 # optional; enables the MTP drafter (spec 4)
#   export STRATA_EXL3_ENGINE=/path/to/strata                  # optional; default: build-hip-exl3/strata in this checkout
#   export STRATA_EXL3_PYTHON=/path/to/python                  # optional; default: .venv/bin/python, then python3
#   export STRATA_ROCM_LIB=/path/to/rocm/lib                   # optional; default: /opt/rocm/lib when it exists
#
# GPU choice: the first --gpu entry is the primary (dense weights, KV cache,
# primary expert cache); --peer-device 1 is the second expert tier. Give the
# primary the fastest effective PCIe link and, when possible, no display.
# If HIP lists an integrated GPU first, pass the discrete ordinals explicitly,
# e.g. --gpu 1,2 --peer-device 1.
#
# Reserve choice: 768 MiB on a headless primary, 2048 MiB on a peer that
# drives the display (the values below). Swap them (--reserve 2048
# --peer-reserve 768) when the display sits on the primary. They are sizing
# margins for automatic sizing, not runtime guarantees; raise the display
# card's reserve if the desktop slows down.
#
# P2P output is OFF by default: peer decode results return through the host,
# as in the release baseline. To opt in (only on request, working GPU peer
# access required):
#   STRATA_EXL3_PEER_DEVICE_OUT=2 STRATA_EXL3_PEER_OUTPUT_FUSED=1 \
#     ./examples/qwen-exl3-3bpw-resident-all.sh ...
#
# Trailing options pass through to the launcher and override the defaults
# below, e.g. --max-context 65536 (lower the 250K tested capacity on smaller
# cards), --prefill 16384, --port 8086, --engine-log FILE. Append --dry-run
# first: it prints the resolved configuration and starts nothing.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${STRATA_EXL3_MODEL:-}" ] && ! printf '%s\n' "$@" | grep -q -- '--model-dir'; then
    echo "qwen-exl3-3bpw-resident-all.sh: set STRATA_EXL3_MODEL to the 3.05 bpw EXL3 model" \
         "directory, or pass --model-dir DIR" >&2
    exit 2
fi

# Baseline transport and placement, overridable from the shell: host-collected
# peer output (no P2P) and static peer placement (no adaptive swaps). With
# full residency, eligible prefill loans from both caches are automatic.
export STRATA_EXL3_PEER_DEVICE_OUT="${STRATA_EXL3_PEER_DEVICE_OUT:-0}"
export STRATA_EXL3_PEER_OUTPUT_FUSED="${STRATA_EXL3_PEER_OUTPUT_FUSED:-0}"
export STRATA_EXL3_PEER_ADAPT="${STRATA_EXL3_PEER_ADAPT:-0}"

# Prepend the standard ROCm library directory only when it exists, so the
# script also runs on machines with ROCm elsewhere or a properly linked build.
ROCM_LIB="${STRATA_ROCM_LIB:-/opt/rocm/lib}"
if [ -d "$ROCM_LIB" ]; then
    export LD_LIBRARY_PATH="$ROCM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$ROOT/run-peer.sh" \
    --expert-cache auto \
    --gpu 0,1 --peer-device 1 --cpu off --resident all \
    --max-context 250000 --prefill 8192 \
    --reserve 768 --peer-reserve 2048 \
    --spec 4 "$@"
