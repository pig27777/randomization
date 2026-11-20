#!/usr/bin/env bash
set -euo pipefail
run=$(dirname "$0")
reference=$(dirname "$run")/rss2027-stage1-warp-film-normal-r1
prep=$(dirname "$run")/rss2027-stage1-pickup-baseline-cpu-r4
py=.venv-physical-parcel/bin/python
cp "$reference/create_overlay.py" "$run/create_overlay.py"
"$py" -B "$run/create_overlay.py"
cmp "$reference/overlay-info.json" "$run/overlay-info.json"
"$py" -B test/bag_model/prepare_physical_parcel.py --model-options "$(cat "$prep/prepare.json")" --seconds .5 --output "$run/preparation"
"$py" -B - "$reference/trial-config.json" "$run" <<'PY'
import json
from pathlib import Path
import sys

config_path, run = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
config["initial_state"] = str(run / "preparation/initial_state.npz")
(run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
env PYTHONPATH="$run/isolated-package" "$py" -B test/bag_model/record_physical_parcel.py \
    --engine warp --record-penetration-peaks --config "$run/config.json" --output "$run/recording"
