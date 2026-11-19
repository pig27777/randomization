#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
first="$(dirname "$run_dir")/rss2027-stage1-warp-film-normal-r1"
python=.venv-physical-parcel/bin/python
env PYTHONPATH="$first/isolated-package" "$python" -B test/bag_model/record_physical_parcel.py \
    --engine warp --record-penetration-peaks --config "$run_dir/config.json" --output "$run_dir/recording"
"$python" -B "$first/analyze_trial.py" "$run_dir" --cpu \
    "$(dirname "$run_dir")/rss2027-stage1-pickup-baseline-cpu-r6/recording/trial/result.json"
