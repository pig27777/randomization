"""One corrected-envelope recording through the unchanged shared YAM actions."""
import json
from pathlib import Path
import time

import physical_parcel_yam as backend
from yam_platform.protocol import run

root = Path(__file__).resolve().parent
started = time.perf_counter()
last = 0.0
previous_phase = None


def observe(robot, row):
    global last, previous_phase
    now = time.perf_counter()
    if now - last >= 2 or row["phase"] != previous_phase:
        progress = dict(row, wall_s=now - started)
        temporary = root / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(root / "progress.json")
        last = now
    if row["phase"] != previous_phase:
        print(json.dumps(dict(time_s=row["t"], phase=row["phase"], wall_s=now-started)), flush=True)
        previous_phase = row["phase"]


cfg = backend.Config(dt=.000025, pinch_coeff=0., contact_time=.001, wall_limit=7200., grasp_face='side', support_height=.15, yaw=90., tilt=30.)
result = run(backend, cfg, root / "trial", on_sample=observe)
(root / "summary.json").write_text(json.dumps(result, indent=2))
