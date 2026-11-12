#!/usr/bin/env bash
# Same hardware-aware entry point; select the downloaded 4.05 bpw model directory.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${STRATA_EXL3_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
exec "$PYTHON" "$ROOT/tools/run_exl3.py" --family qwen "$@"
