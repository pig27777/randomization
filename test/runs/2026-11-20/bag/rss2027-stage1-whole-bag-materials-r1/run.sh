#!/usr/bin/env bash
set -euo pipefail
run_dir=${2:-$(dirname "$0")}
python=.venv-physical-parcel/bin/python
recorder=test/bag_model/record_physical_parcel.py

case "${1:?specify smoke or a material name}" in
    smoke)
        for config in "$run_dir"/*/smoke.json; do
            case_dir=$(dirname "$config")
            "$python" -B "$recorder" --record-penetration-peaks \
                --config "$config" --output "$case_dir/smoke" > "$case_dir/smoke.log" 2>&1
        done
        ;;
    pe|kraft|bubble|paperboard)
        case_dir="$run_dir/$1"
        "$python" -B - "$case_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
if (result["outcome"] != "incomplete" or result["warnings"] or
        result["sim_time_s"] < .020 or result["peaks"]["penetration_mm"] > 1.):
    raise SystemExit(f"Smoke did not pass: {result['outcome']} {result['reason']}")
PY
        "$python" -B test/bag_model/prepare_physical_parcel.py \
            --model-options "$(cat "$case_dir/prepare.json")" --seconds .50 \
            --output "$case_dir/preparation" > "$case_dir/preparation.log" 2>&1
        "$python" -B "$recorder" --record-penetration-peaks \
            --config "$case_dir/config.json" --output "$case_dir/recording" > "$case_dir/recording.log" 2>&1
        ;;
    *)
        exit 2
        ;;
esac
