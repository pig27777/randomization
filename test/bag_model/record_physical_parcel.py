"""Record a physical parcel trial through the shared YAM protocol."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import time

import physical_parcel_yam
from yam_platform.protocol import run


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("cpu", "warp"), default="cpu")
    args = parser.parse_args()
    backend = (physical_parcel_yam if args.engine == "cpu" else
               import_module("physical_parcel_yam_warp"))
    cfg = backend.Config(**json.loads(args.config.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recorder_source.py").write_bytes(Path(__file__).read_bytes())
    last = 0.0
    previous_phase = None
    started = time.perf_counter()
    device_samples = []

    def observe(robot, row):
        nonlocal last, previous_phase
        now = time.perf_counter()
        if now - last >= 5 or row["phase"] != previous_phase:
            if args.engine == "warp":
                device_samples.append(dict(t_s=row["t"], **robot.sim.telemetry()))
            temporary = args.output / "progress.tmp"
            progress = dict(row, wall_s=now-started)
            if device_samples:
                progress["gpu"] = device_samples[-1]
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(args.output / "progress.json")
            last = now
        if row["phase"] != previous_phase:
            print(json.dumps(dict(time_s=row["t"], phase=row["phase"], wall_s=now-started)), flush=True)
            previous_phase = row["phase"]

    result = run(backend, cfg, args.output / "trial", on_sample=observe)
    if args.engine == "warp":
        result["gpu_samples"] = device_samples
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
