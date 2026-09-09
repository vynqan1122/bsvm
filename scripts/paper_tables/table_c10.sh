#!/usr/bin/env bash
# Table C.10: author BSVM plus three priorities, Exp2 and steel, four kernels.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table c10 "$@"
