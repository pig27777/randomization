#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$0")
run_dir=${1:-$source_dir}
python=.venv-physical-parcel/bin/python
"$python" -B "$source_dir/record_ridge.py" --config "$run_dir/smoke.json" --output "$run_dir/smoke" > "$run_dir/smoke.log" 2>&1
"$python" -B - "$run_dir/smoke/summary.json" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text())
assert result["outcome"] == "smoke_only" and result["sim_time_s"] >= .02
assert not result["warnings"] and result["dimensions"]["film_vertices"] == 318
assert result["peaks"]["penetration_mm"] <= 1.
assert result["normal_press"]["peak_direct_support_N"] <= .05
PY
"$python" -B "$source_dir/record_ridge.py" --config "$run_dir/config.json" --output "$run_dir/recording" \
    --record-penetration-peaks > "$run_dir/run.log" 2>&1
if "$python" -B - "$run_dir/recording/trial/trace.json" <<'PY'
import json
from pathlib import Path
import sys

rows = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(not any(row["phase"] in ("close", "lift") for row in rows))
PY
then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$run_dir/recording/trial" "$run_dir/capture"
fi
