#!/usr/bin/env bash
# Table 4: RBF, minority/majority F1, seven models including three new priorities.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table 4 "$@"
