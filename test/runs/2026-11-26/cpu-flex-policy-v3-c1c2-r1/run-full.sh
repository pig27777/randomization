#!/usr/bin/env bash
set -euo pipefail
run="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$run/../../../.."
PE_PYTHON=/home/ubuntu/workspace/gzl/randomization/.venv-physical-parcel/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
worker=test/bag_model/rss_cpu_flex_c1c2.py
policy=test/bag_model/rss_cpu_flex_c1c2_policy.py
publish=test/bag_model/rss_cpu_flex_c1c2_report.py
gate="${1:-$run/pilot-5/summary.json}"
"$PE_PYTHON" -B "$worker" "$run/collect" collect --gate "$gate" --workers 22
"$PE_PYTHON" -B "$policy" "$run" train
"$PE_PYTHON" -B "$worker" "$run/closed" closed --workers 22
"$PE_PYTHON" -B "$worker" "$run/matched" matched --workers 22
"$PE_PYTHON" -B "$policy" "$run" plan
"$PE_PYTHON" -B "$policy" "$run" branches --workers 22
"$PE_PYTHON" -B "$policy" "$run" summarize
"$PE_PYTHON" -B "$publish" "$run"
bash "$run/check-frozen.sh" > "$run/frozen-check.log"
