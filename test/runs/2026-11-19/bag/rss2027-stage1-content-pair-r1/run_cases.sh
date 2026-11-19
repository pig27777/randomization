#!/usr/bin/env bash
set -u
run_dir=$(dirname "$0")
for case in compact-offset broad-offset
do
    .venv-physical-parcel/bin/python -B test/bag_model/record_physical_parcel.py \
        --config "$run_dir/$case.json" --output "$run_dir/$case" > "$run_dir/$case.log" 2>&1
    trial_exit=$?
    printf '%s %d\n' "$case" "$trial_exit"
done
