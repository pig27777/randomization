#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$0")
run_dir=${1:-$source_dir}
python=.venv-physical-parcel/bin/python
"$python" -B test/bag_model/record_physical_parcel.py --record-penetration-peaks \
    --config "$run_dir/config.json" --output "$run_dir/recording"
"$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"
"$python" -B "$source_dir/analyze_release.py" "$run_dir"
if [ -s "$run_dir/recording/release-contacts.json" ]; then
    "$python" -B test/bag_model/rss2027_release_traction.py "$run_dir"
fi
