#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
cpu_run=$(dirname "$run_dir")/rss2027-stage1-pickup-baseline-cpu-r3
python=.venv-physical-parcel/bin/python

if "$python" -B - "$cpu_run" "$run_dir" <<'PY'
import json
from pathlib import Path
import sys

cpu_run, output = map(Path, sys.argv[1:])
result_path = cpu_run / "recording/trial/result.json"
if not result_path.is_file():
    (output / "gate.json").write_text(json.dumps({"ready": False, "reason": "CPU trial did not produce a result"}, indent=2) + "\n")
    raise SystemExit(1)
result = json.loads(result_path.read_text())
trace = json.loads((result_path.parent / "trace.json").read_text())
hold = [row for row in trace if row["phase"] == "lift_hold"]
passed = dict(
    motion=result["outcome"] == "pickup_ok" and result["motion_completed"],
    duration=3. <= result["sim_time_s"] <= 5.,
    numerical=not result["warnings"] and result["peaks"]["penetration_mm"] <= 1.,
    hold=bool(hold) and all(row["support_N"] <= .05 and min(row["grip_each_N"]) > .001 for row in hold),
)
decision = dict(ready=all(passed.values()), checks=passed, cpu_result=str(result_path))
(output / "gate.json").write_text(json.dumps(decision, indent=2) + "\n")
if not decision["ready"]:
    raise SystemExit(1)
cfg = json.loads((cpu_run / "config.json").read_text())
cfg["wall_limit"] = 2400.
(output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
then
    "$python" -B test/bag_model/record_physical_parcel.py --engine warp --record-penetration-peaks \
        --config "$run_dir/config.json" --output "$run_dir/recording"
fi
