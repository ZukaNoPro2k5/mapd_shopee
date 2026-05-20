#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHOD="${1:-GreedyBFS}"

if [[ ! -f run_test.py ]]; then
  echo "Missing run_test.py. Copy the Kaggle bundle into the repo root first."
  exit 1
fi

if [[ ! -f smoke_config.txt ]]; then
  echo "Missing smoke_config.txt."
  exit 1
fi

mkdir -p results/smoke
python3 run_test.py \
  --method "$METHOD" \
  --config smoke_config.txt \
  --out results/smoke \
  --seed 42
