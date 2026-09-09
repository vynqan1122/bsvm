#!/usr/bin/env bash
# Shared launcher. Arguments after the table-specific script override defaults.
set -euo pipefail

run_paper_table() {
    local table_id="$1"
    shift
    local project_root
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    cd -- "$project_root"

    local python_bin="${PYTHON:-}"
    if [[ -z "$python_bin" ]]; then
        if [[ -x "$project_root/.venv-linux/bin/python" ]]; then
            python_bin="$project_root/.venv-linux/bin/python"
        elif [[ -x "$project_root/.venv/bin/python" ]]; then
            python_bin="$project_root/.venv/bin/python"
        elif [[ -x "$project_root/.venv/Scripts/python.exe" ]]; then
            python_bin="$project_root/.venv/Scripts/python.exe"
        elif command -v python3 >/dev/null 2>&1; then
            python_bin="python3"
        else
            python_bin="python"
        fi
    fi

    # Limit numerical-library threads for comparable timing; callers may override.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    exec "$python_bin" "$project_root/examples/run_paper_table.py" \
        --table "$table_id" \
        --profile-json "$project_root/config/ci_profiles.json" \
        --output-dir "$project_root/outputs/paper_tables/table_$table_id" \
        "$@"
}
