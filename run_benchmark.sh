#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MATLAB_BIN="${MATLAB_BIN:-/Applications/MATLAB_R2025b.app/bin/matlab}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" evaluate.py --device cpu

if [[ "${RUN_SPECTRAL_TIMING:-0}" == "1" ]]; then
  "$MATLAB_BIN" -batch "addpath('matlab'); benchmark_spectral_3d"
fi

"$PYTHON_BIN" benchmark_inference.py --config config.json
echo "Benchmark results are in results/ and figures/."
