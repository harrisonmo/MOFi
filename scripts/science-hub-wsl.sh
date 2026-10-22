#!/usr/bin/env bash
# Run the science hub in a separately configured Linux environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCIENCE_HUB_PYTHON:?Set SCIENCE_HUB_PYTHON to the Linux science Python executable}"
if [[ -n "${SCIENCE_HUB_CP2K_ENV:-}" ]]; then
    export PATH="$PATH:$SCIENCE_HUB_CP2K_ENV/bin"
    export CP2K_DATA_DIR="${CP2K_DATA_DIR:-$SCIENCE_HUB_CP2K_ENV/share/cp2k/data}"
fi
export PYTHONPATH="$ROOT:$ROOT/Crystal-Builder${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
cd "$ROOT"
exec "$SCIENCE_HUB_PYTHON" "$@"
