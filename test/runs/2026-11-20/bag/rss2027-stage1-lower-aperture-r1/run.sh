#!/usr/bin/env bash
set -euo pipefail
run_dir=${1:-$(dirname "$0")}
runner="$(dirname "$run_dir")/rss2027-stage1-whole-bag-materials-r1/run.sh"
python=.venv-physical-parcel/bin/python
bash "$runner" smoke "$run_dir"
for material in kraft bubble; do
    case_dir="$run_dir/$material"
    [[ -f "$case_dir/smoke.json" ]] || continue
    if "$python" -B - "$case_dir/smoke/trial/trace.json" <<'PY'
import json
from pathlib import Path
import sys

trace = json.loads(Path(sys.argv[1]).read_text())
settle = [row for row in trace if row["phase"] == "settle"]
if not settle or settle[-1]["t"] < .28:
    raise SystemExit("Missing observations near the end of the settle phase")
force = max(max(row["contact_magnitude_each_N"]) for row in settle)
if force > .001:
    raise SystemExit(f"Settling exceeded the 1 mN fingertip clearance gate: {force:g} N")
PY
    then
        bash "$runner" "$material" "$run_dir"
        "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$case_dir/recording/trial" "$case_dir/capture"
    fi
done
