"""Test the archived YAM phases at 12,000 worlds with explicit buffer sizes."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--ccd-capacity", type=int)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    run = args.output
    overlay_script = run.with_name("rss2027-stage1-warp-film-normal-r1") / "create_overlay.py"
    source = run.with_name("rss2027-yam-heterogeneous-scaling-r1")
    trial = source / "recording/trial"
    temporary_script = run / overlay_script.name
    if (run / "summary.json").exists() or temporary_script.exists():
        raise FileExistsError(run)
    shutil.copyfile(overlay_script, temporary_script)
    try:
        subprocess.run([sys.executable, "-B", str(temporary_script)], check=True)
    finally:
        temporary_script.unlink()
    if json.loads((run / "overlay-info.json").read_text()) != json.loads((source / "overlay-info.json").read_text()):
        raise ValueError("Isolated contact module differs from the prior run")

    start = datetime.now().astimezone().isoformat(timespec="seconds")
    env = {**os.environ, "PYTHONPATH": str(run / "isolated-package")}
    attempts = []
    variants = [("standard", []), ("compact", ["--compact-capacity"])]
    if args.ccd_capacity is not None:
        variants = [("compact", ["--compact-capacity", "--ccd-capacity", str(args.ccd_capacity)])]
    for variant, option in variants:
        outcome = "passed"
        for phase in ("close", "lift_hold"):
            key = f"{variant}-{phase}"
            output, log = run / f"{key}.json", run / f"{key}.log"
            snapshot = subprocess.check_output([
                "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ], text=True).strip()
            launch = ["--eager"] if args.eager else []
            command = [sys.executable, "-B", "test/bag_model/rss2027_yam_batch_profile.py",
                       str(trial), str(output), "--worlds", "12000", "--phase", phase, "--scaling", *option, *launch]
            print(f"Starting {key} | {snapshot}", flush=True)
            with log.open("x") as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            attempt = dict(variant=variant, phase=phase, returncode=result.returncode,
                           preflight_gpu_csv=snapshot, log=log.name)
            attempts.append(attempt)
            if result.returncode:
                message = log.read_text()
                lower = message.lower()
                is_oom = "cuda" in lower and ("out of memory" in lower or "cuda_error_out_of_memory" in lower)
                outcome = "gpu_out_of_memory" if is_oom else "execution_error"
                attempt.update(status=outcome, world_buffers_allocated="World buffers allocated" in message)
                print(f"Finished {key}: {outcome}", flush=True)
                break
            profile = json.loads(output.read_text())
            attempt.update(status="passed" if profile["valid"] else "health_failed", result=profile)
            print(f"Finished {key}: {attempt['status']}, {profile['sim_sec_per_wall_sec']} sim s/wall s", flush=True)
            if not profile["valid"]:
                outcome = "health_failed"
        if outcome != "gpu_out_of_memory":
            break

    summary = dict(host=platform.node(), start=start, end=datetime.now().astimezone().isoformat(timespec="seconds"),
                   worlds=12000, reference_trial=str(trial), outcome=outcome, attempts=attempts,
                   execution="eager" if args.eager else "cuda_graph", ccd_capacity=args.ccd_capacity)
    with (run / "summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if outcome == "execution_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
