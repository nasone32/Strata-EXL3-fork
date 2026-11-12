#!/usr/bin/env bash
# Qwen3.8-Flash-Next EXL3 (tested with 3.05 bpw).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${STRATA_EXL3_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
exec "$PYTHON" "$ROOT/tools/run_exl3.py" --family qwen "$@"
