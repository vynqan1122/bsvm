#!/usr/bin/env bash
# Table C.8: author BSVM plus three priorities, per-class F1 across four kernels.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table c8 "$@"
