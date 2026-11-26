#!/usr/bin/env bash
set -euo pipefail
cd -- "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/experience:$PWD"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WARP_CACHE_DIR="${WARP_CACHE_DIR:-/tmp/warp_cache_gpu_0}"
export WARP_CACHE_PATH="$WARP_CACHE_DIR"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
exec flock /tmp/claude-gpu.lock "$PE_PYTHON" -B "$@"
