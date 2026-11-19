#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
reference="$(dirname "$run_dir")/rss2027-stage1-warp-film-normal-r1"
python=.venv-physical-parcel/bin/python
for worlds in 1 16 64; do
    for phase in settle close lift_hold; do
        name="$worlds-$phase"
        if ! env PYTHONPATH="$reference/isolated-package" "$python" -B test/bag_model/rss2027_yam_batch_profile.py \
            "$reference/recording/trial" "$run_dir/$name.json" --worlds "$worlds" --phase "$phase" \
            > "$run_dir/$name.log" 2>&1; then
            printf 'Device timing failed for %s. See %s.log\n' "$name" "$name"
        fi
    done
done
