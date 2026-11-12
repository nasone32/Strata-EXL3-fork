#!/usr/bin/env bash
# Experimental MiMo-V2.6-Flash-MOPD EXL3 entry point (tested quant: 2.20bpw).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${STRATA_EXL3_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3
exec "$PYTHON" "$ROOT/tools/run_exl3.py" --family mimo "$@"
