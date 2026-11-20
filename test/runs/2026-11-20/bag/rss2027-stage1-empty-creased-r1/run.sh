#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
case_dir="$run_dir/aligned"
python=.venv-physical-parcel/bin/python
for mode in smoke settle; do
    "$python" -B "$run_dir/record_creased.py" --config "$case_dir/$mode.json" --output "$case_dir/$mode" \
        --record-penetration-peaks > "$case_dir/$mode.log" 2>&1
    summary="$case_dir/$mode/summary.json"
    jq -c '{outcome,sim_time_s,peaks,warnings}' "$summary"
    jq -e '.outcome=="smoke_only" and .sim_time_s>=.config.smoke_seconds and
           .dimensions.film_vertices==142 and .warnings=={} and .peaks.penetration_mm<=1' "$summary" >/dev/null
done
"$python" -B - "$case_dir" <<'PY'
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

case = Path(sys.argv[1])
trial = case / "settle/trial"
model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
data = mujoco.MjData(model)
with np.load(trial / "trajectory.npz") as archive:
    data.qpos[:] = archive["final_qpos"]
mujoco.mj_fwdPosition(model, data)
points = data.flexvert_xpos
y_center = float(np.mean(points[:, 1]))
outside = abs(points[:, 1] - y_center) >= .035
height = float(points[:, 2].max() - points[outside, 2].max())
(case / "settled-height.json").write_text(json.dumps(dict(crest_over_flanks_m=height), indent=2) + "\n")
raise SystemExit(height < .002)
PY
"$python" -B "$run_dir/record_creased.py" --config "$case_dir/config.json" --output "$case_dir/recording" \
    --record-penetration-peaks --record-capture-forces > "$case_dir/run.log" 2>&1
if jq -e 'any(.[]; .phase == "close")' "$case_dir/recording/trial/trace.json" >/dev/null; then
    "$python" -B test/bag_model/rss2027_yam_capture_audit.py "$case_dir/recording/trial" "$case_dir/capture"
fi
