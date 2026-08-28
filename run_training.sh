#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" train.py --model both --device cpu
"$PYTHON_BIN" evaluate.py --device cpu
