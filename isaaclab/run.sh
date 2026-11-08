#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ENV=/home/ubuntu/Dev/Envs/isaaclab
export XDG_CACHE_HOME="$ISAACLAB_ENV/cache/xdg"
export XDG_CONFIG_HOME="$ISAACLAB_ENV/cache/config"
export XDG_DATA_HOME="$ISAACLAB_ENV/cache/data"
export OMNI_CONFIG_PATH="$ISAACLAB_ENV/cache/omni/config"
export CUDA_CACHE_PATH="$ISAACLAB_ENV/cache/cuda"
export TORCH_HOME="$ISAACLAB_ENV/cache/torch"
export HF_HOME="$ISAACLAB_ENV/cache/huggingface"
export MPLCONFIGDIR="$ISAACLAB_ENV/cache/matplotlib"
export NUMBA_CACHE_DIR="$ISAACLAB_ENV/cache/numba"
export TMPDIR="$ISAACLAB_ENV/cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$OMNI_CONFIG_PATH" \
    "$CUDA_CACHE_PATH" "$TORCH_HOME" "$HF_HOME" \
    "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TMPDIR"
exec "$ISAACLAB_ENV/bin/python" -B "$@"
