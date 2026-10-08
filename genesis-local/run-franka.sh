#!/usr/bin/env bash
set -euo pipefail
genesis_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GS_CACHE_FILE_PATH="$genesis_dir/.cache/genesis"
export XDG_CACHE_HOME="$genesis_dir/.cache"
cd "$genesis_dir"
exec "$genesis_dir/.venv/bin/python" "$genesis_dir/franka_demo.py" "$@"
