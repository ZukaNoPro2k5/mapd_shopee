#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

METHOD="${1:-GreedyBFS}"

if [[ ! -f smoke_suite_config.txt ]]; then
  echo "Missing smoke_suite_config.txt."
  exit 1
fi

mkdir -p results/smoke_suite
python3 run_test.py \
  --method "$METHOD" \
  --config smoke_suite_config.txt \
  --out results/smoke_suite
