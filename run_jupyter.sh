#!/usr/bin/env bash
# Start Jupyter without writing configuration into the account home directory.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export JUPYTER_CONFIG_DIR="$project_dir/.jupyter/config"
export JUPYTER_DATA_DIR="$project_dir/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$project_dir/.jupyter/runtime"
export IPYTHONDIR="$project_dir/.ipython"
export MPLCONFIGDIR="$project_dir/.mplconfig"

exec "$project_dir/.venv/bin/jupyter" lab --notebook-dir="$project_dir"
