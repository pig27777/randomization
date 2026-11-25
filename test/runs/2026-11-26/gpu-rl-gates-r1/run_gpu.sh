#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
export PYTHONPATH="$PWD/src:$PWD/experience:$PWD/test/bag_model:$PWD${PE_EXTRA_PYTHONPATH:+:$PE_EXTRA_PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WARP_CACHE_DIR="${WARP_CACHE_DIR:-/tmp/warp_cache_gpu_0}"
exec flock /tmp/claude-gpu.lock /home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python -B "$@"
