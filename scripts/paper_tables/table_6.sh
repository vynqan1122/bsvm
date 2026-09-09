#!/usr/bin/env bash
# Table 6: selected-model training/prediction timing and SV counts, both protocols.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table 6 "$@"
