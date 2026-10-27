#!/usr/bin/env bash
# Run AnTBench api inference followed by evaluation.
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/run_experiments.sh" api "${EXPERIMENT:-main}"
