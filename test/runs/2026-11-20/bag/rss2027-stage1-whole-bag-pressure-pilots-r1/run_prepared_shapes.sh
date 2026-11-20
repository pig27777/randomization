#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
series=${1:-prepared-series}
for name in bubble-027 paperboard-127 kraft-127; do
    origin="$run_dir/slack-series/$name/config.json"
    case_dir="$run_dir/$series/$name"
    mkdir -p -- "$case_dir"
    if ! "$python" -B test/bag_model/prepare_physical_parcel.py --seconds .30 \
        --model-options "$(jq -c . "$origin")" --output "$case_dir/preparation" > "$case_dir/preparation.log" 2>&1; then
        printf '%s preparation failed, see %s/preparation.log\n' "$name" "$case_dir"
        continue
    fi
    jq --arg state "$case_dir/preparation/initial_state.npz" \
        '.initial_state=$state | .press_start_offset_m=.035' "$origin" > "$case_dir/config.json"
    jq '.smoke_seconds=.02' "$case_dir/config.json" > "$case_dir/smoke.json"
    jq '.smoke_seconds=.30' "$case_dir/config.json" > "$case_dir/settle.json"
    REQUIRE_CLEAR_APPROACH=1 bash "$run_dir/run.sh" "$series/$name"
done
