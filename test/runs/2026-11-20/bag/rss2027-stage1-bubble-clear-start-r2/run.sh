#!/usr/bin/env bash
set -euo pipefail
run_dir=$(dirname "$0")
runner="$(dirname "$run_dir")/rss2027-stage1-whole-bag-materials-r1/run.sh"
bash "$runner" smoke "$run_dir"
.venv-physical-parcel/bin/python -B - "$run_dir/bubble/smoke/trial/trace.json" <<'PY'
import json
from pathlib import Path
import sys

trace = json.loads(Path(sys.argv[1]).read_text())
settle = [row for row in trace if row["phase"] == "settle"]
if not settle or settle[-1]["t"] < .28:
    raise SystemExit("Missing observations near the end of the settle phase")
maximum = max(max(row["contact_magnitude_each_N"]) for row in settle)
if maximum > .001:
    raise SystemExit(f"Settling exceeded the 1 mN fingertip clearance gate: {maximum:g} N")
PY
bash "$runner" bubble "$run_dir"
