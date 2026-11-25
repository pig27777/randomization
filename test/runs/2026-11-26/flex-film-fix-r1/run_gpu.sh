#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
export PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/experience:$PWD/test/bag_model:$PWD"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WARP_CACHE_DIR="${WARP_CACHE_DIR:-/tmp/warp_cache_gpu_0}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
if [[ "${1:-}" == --step1 ]]; then
  shift
  exec flock /tmp/claude-gpu.lock timeout 600 "$PE_PYTHON" -B "$@"
fi
exec flock /tmp/claude-gpu.lock "$PE_PYTHON" -B "$@"
