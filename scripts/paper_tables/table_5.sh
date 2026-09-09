#!/usr/bin/env bash
# Table 5: RBF, accuracy and macro metrics, six models.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table 5 "$@"
