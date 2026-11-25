#!/usr/bin/env bash
set -euo pipefail
lab=/home/ubuntu/Dev/Envs/isaaclab/src/IsaacLab/source
export PYTHONPATH="$lab/isaaclab:$lab/isaaclab_physx:$lab/isaaclab_newton:$lab/isaaclab_visualizers:$lab/isaaclab_ov:$lab/isaaclab_contrib:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec flock /tmp/claude-gpu.lock ./isaaclab/run.sh \
  test/isaaclab/benchmark_newton_yam_bag.py --headless "$@"
