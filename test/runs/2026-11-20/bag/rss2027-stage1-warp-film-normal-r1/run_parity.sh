#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
python=.venv-physical-parcel/bin/python
cpu=test/runs/2026-11-19/bag/rss2027-stage1-pickup-baseline-cpu-r3/recording/trial
warp=test/runs/2026-11-19/bag/rss2027-stage1-pickup-baseline-warp-r3/recording
audit=test/bag_model/rss2027_payload_contact_parity.py
projection=test/runs/2026-11-20/bag/rss2027-stage1-ccd-parity-r1/check_normal_depth.py
env PYTHONPATH="$run_dir/isolated-package" "$python" -B "$audit" "$warp" "$cpu" \
    "$run_dir/parity.json" --film-vertical-velocity .01
"$python" -B "$projection" "$run_dir/normal-depth.json" "$run_dir/parity.json"
