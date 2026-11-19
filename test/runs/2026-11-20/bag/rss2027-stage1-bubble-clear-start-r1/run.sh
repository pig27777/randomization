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
if not trace or any(max(row["contact_magnitude_each_N"]) > .001 for row in trace):
    raise SystemExit("Smoke observations include native fingertip contact at the starting distance")
PY
bash "$runner" bubble "$run_dir"
