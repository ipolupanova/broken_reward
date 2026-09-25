#!/usr/bin/env bash
# One-command entry point. No dependencies beyond Python 3.9+.
#
#   ./run.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "python3 not found. Install Python 3.9+ and re-run." >&2
    exit 1
fi

"$PY" run_all.py
