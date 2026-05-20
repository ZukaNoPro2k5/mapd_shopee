#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Local default = smoke test only. Full Phase 1/full-map runs belong on Kaggle."
exec bash scripts/run_smoke_test.sh "${1:-GreedyBFS}"
