#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f run_test.py ]]; then
  echo "Missing run_test.py. Copy the Kaggle bundle into the repo root first."
  exit 1
fi

if [[ ! -f test_config.txt ]]; then
  echo "Missing test_config.txt. Copy the Kaggle bundle into the repo root first."
  exit 1
fi

mkdir -p results
python3 run_test.py --config test_config.txt --out results/ --seed 42
