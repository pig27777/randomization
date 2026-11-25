#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run_dir/../../../.."
native=/home/ubuntu/workspace/czx/system0/sim
lab="$native/IsaacLab/source"
export PYTHONPATH="$lab/isaaclab:$lab/isaaclab_physx:$lab/isaaclab_newton:$lab/isaaclab_visualizers:$lab/isaaclab_ov:$lab/isaaclab_contrib:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec flock /tmp/claude-gpu.lock "$native/python.sh" -B "$run_dir/retest.py" --headless "$@"
