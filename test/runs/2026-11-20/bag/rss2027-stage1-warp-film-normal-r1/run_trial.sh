#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
original=test/runs/2026-11-20/bag/rss2027-stage1-pickup-baseline-warp-r4/config.json
python=.venv-physical-parcel/bin/python

if "$python" -B - "$run_dir" <<'PY'
import json
import math
from pathlib import Path
import sys

run = Path(sys.argv[1])
frames = json.loads((run / "normal-depth.json").read_text())[0]["frames"]
checks = {}
for name in ("warp_end_close", "warp_before_failure", "warp_failure"):
    case = frames[name]
    cpu, warp = case["cpu"], case["warp"]
    angle, error = None, None
    if len(cpu) == len(warp) == 1:
        dot = sum(a * b for a, b in zip(cpu[0]["direction"], warp[0]["direction"], strict=True))
        angle = math.degrees(math.acos(max(-1., min(1., dot))))
        error = abs(warp[0]["difference_m"])
    checks[name] = dict(angle_deg=angle, separation_error_m=error,
                        passed=angle is not None and angle < 1. and error < 5e-6)
ready = all(check["passed"] for check in checks.values())
state = dict(ready=ready, checks=checks, cpu_reference_warp_contact_count=len(frames["cpu_near_failure_time"]["warp"]))
(run / "trial-gate.json").write_text(json.dumps(state, indent=2) + "\n")
raise SystemExit(0 if ready else 1)
PY
then
    cp -- "$original" "$run_dir/trial-config.json"
    env PYTHONPATH="$run_dir/isolated-package" "$python" -B test/bag_model/record_physical_parcel.py \
        --engine warp --record-penetration-peaks --config "$run_dir/trial-config.json" --output "$run_dir/recording"
fi
