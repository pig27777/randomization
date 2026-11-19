#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
"$python" -B test/bag_model/prepare_physical_parcel.py --model-options "$(cat "$run_dir/prepare.json")" \
    --seconds .50 --output "$run_dir/preparation" > "$run_dir/preparation.log" 2>&1
"$python" -B test/bag_model/record_physical_parcel.py --record-penetration-peaks --config "$run_dir/config.json" \
    --output "$run_dir/recording" > "$run_dir/recording.log" 2>&1
