#!/usr/bin/env bash
# Reuses the existing environment, installed packages, and repository.
set -Eeuo pipefail
genesis_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$genesis_dir"
exec > >(tee -a "$genesis_dir/install.log") 2>&1
trap 'printf "Installation failed at line %s. See %s/install.log\n" "$LINENO" "$genesis_dir" >&2' ERR
uv_bin="$genesis_dir/.tools/uv/uv"
python_bin="$genesis_dir/.venv/bin/python"
test -x "$uv_bin"
test -x "$python_bin"
test -f "$genesis_dir/genesis-world/pyproject.toml"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
# Share existing wheel files through hardlinks to avoid duplicating CUDA libraries.
export UV_CACHE_DIR="$HOME/.cache/uv"
export UV_NO_CONFIG=1
export UV_PYTHON_INSTALL_DIR="$genesis_dir/.tools/python"
export UV_PYTHON_BIN_DIR="$genesis_dir/.tools/bin"
export UV_PYTHON_INSTALL_BIN=0
unset UV_TORCH_BACKEND
# Triton compiles a CUDA driver extension and needs the Python C headers.
if ! "$python_bin" -c 'import sysconfig; from pathlib import Path; assert (Path(sysconfig.get_path("include"))/"Python.h").is_file()'; then
    "$uv_bin" venv --allow-existing --managed-python --python 3.12 "$genesis_dir/.venv"
fi
"$uv_bin" pip install --python "$python_bin" --link-mode hardlink \
    --index-url https://pypi.org/simple 'torch==2.14.0' -e './genesis-world[render]'
"$uv_bin" pip check --python "$python_bin"
export GS_CACHE_FILE_PATH="$genesis_dir/.cache/genesis"
export XDG_CACHE_HOME="$genesis_dir/.cache"
"$python_bin" verify_genesis.py --backend cpu 2>&1 | tee "$genesis_dir/verify-cpu.log"
"$python_bin" verify_genesis.py --backend cuda --franka --render 2>&1 | tee "$genesis_dir/verify-cuda.log"
"$uv_bin" pip freeze --python "$python_bin" > "$genesis_dir/installed-packages.txt"
git -C genesis-world rev-parse HEAD > "$genesis_dir/installed-commit.txt"
printf '\nInstallation and verification passed. Run: bash "%s/run-franka.sh"\n' "$genesis_dir"
