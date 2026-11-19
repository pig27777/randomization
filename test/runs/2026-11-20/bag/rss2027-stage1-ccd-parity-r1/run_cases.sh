#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
cpu=test/runs/2026-11-19/bag/rss2027-stage1-pickup-baseline-cpu-r3/recording/trial
warp=test/runs/2026-11-19/bag/rss2027-stage1-pickup-baseline-warp-r3/recording
audit=test/bag_model/rss2027_payload_contact_parity.py

.venv-physical-parcel/bin/python -B "$audit" "$warp" "$cpu" "$run_dir/tol-0p1um.json" \
    --film-vertical-velocity .01 --ccd-tolerance-m 1e-7
.venv-physical-parcel/bin/python -B "$audit" "$warp" "$cpu" "$run_dir/tol-1um.json" \
    --film-vertical-velocity .01 --ccd-tolerance-m 1e-6
