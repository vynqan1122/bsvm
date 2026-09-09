#!/usr/bin/env bash
# Table C.12: OVR multiclass, author/soft-margin SVM plus three priorities.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_paper_table c12 "$@"
