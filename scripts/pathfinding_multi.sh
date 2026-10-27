#!/usr/bin/env bash
# Run AnTBench local inference followed by evaluation.
set -euo pipefail
bash "$(dirname "${BASH_SOURCE[0]}")/run_experiments.sh" local "method"
