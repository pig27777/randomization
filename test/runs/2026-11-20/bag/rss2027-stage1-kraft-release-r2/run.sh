#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
runner="$(dirname "$run_dir")/rss2027-stage1-whole-bag-release-r1/run.sh"
.venv-physical-parcel/bin/python -B test/bag_model/prepare_physical_parcel.py \
    --model-options "$(cat "$run_dir/prepare.json")" --seconds .50 \
    --output "$run_dir/preparation" > "$run_dir/preparation.log" 2>&1
bash "$runner" "$run_dir"
