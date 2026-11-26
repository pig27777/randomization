#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run/../../../.."
PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
driver=test/bag_model/rss_cpu_flex_c2_fix.py
case "${1:-evaluate}" in
    train)
        "$PE_PYTHON" -B "$driver" "$run" train --width 24 --dropout .2 --scorer-lr .0003 --support-weight 1.5
        ;;
    evaluate)
        flock /tmp/claude-gpu.lock "$PE_PYTHON" -B "$driver" "$run" evaluate
        "$PE_PYTHON" -B test/bag_model/rss_cpu_flex_c2_publish.py "$run"
        ;;
    *)
        echo "Usage: $0 [train|evaluate]" >&2
        exit 2
        ;;
esac
