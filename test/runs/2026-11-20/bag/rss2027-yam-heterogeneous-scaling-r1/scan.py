"""Run the two-phase YAM scaling sweep in fresh processes under the shared lock."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys


WORLDS = (1, 16, 64, 128, 256, 512, 1024, 2048, 4096)
PHASES = ("close", "lift_hold")


def main():
    run = Path(__file__).parent
    recording = run / "recording"
    baseline = json.loads((recording / "summary.json").read_text())
    if (baseline["outcome"] != "pickup_ok" or not baseline["motion_completed"]
            or baseline["warnings"] or baseline["peaks"]["penetration_mm"] > 1.
            or not 3. <= baseline["sim_time_s"] <= 5.
            or any(sample["overflow_bits"] for sample in baseline["gpu_samples"])):
        raise RuntimeError("Reconstructed online reference failed the original YAM acceptance")
    env = {**os.environ, "PYTHONPATH": str(run / "isolated-package")}
    trial = recording / "trial"
    rows = []
    best = plateau = 0
    stop = "grid_limit"

    for n in WORLDS:
        phases = {}
        for phase in PHASES:
            key = f"{n}-{phase}"
            output, log = run / f"{key}.json", run / f"{key}.log"
            command = [sys.executable, "-B", "test/bag_model/rss2027_yam_batch_profile.py",
                       str(trial), str(output), "--worlds", str(n), "--phase", phase, "--scaling"]
            print(f"Starting {key}", flush=True)
            with log.open("x") as stream:
                execution = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if execution.returncode:
                message = log.read_text().lower()
                stop = "gpu_out_of_memory" if "out of memory" in message and "cuda" in message else "execution_error"
                phases[phase] = dict(valid=False, returncode=execution.returncode, log=log.name, failure=stop)
                break
            result = json.loads(output.read_text())
            phases[phase] = result
            if result["overflow_bits"]:
                stop = "capacity_overflow"
            print(f"Finished {key}: valid={result['valid']}, "
                  f"sim_s/wall_s={result['sim_sec_per_wall_sec']}", flush=True)

        both_valid = len(phases) == len(PHASES) and all(phase["valid"] for phase in phases.values())
        slower = min(phase["sim_sec_per_wall_sec"] for phase in phases.values()) if both_valid else None
        rows.append(dict(worlds=n, phases=phases, slower_phase_sim_sec_per_wall_sec=slower))
        if stop != "grid_limit":
            break
        if slower is not None:
            plateau = plateau + 1 if best and slower <= best * 1.03 else 0
            best = max(best, slower)
            if plateau == 2:
                stop = "plateau"
                break

    with (run / "summary.json").open("x") as stream:
        json.dump(dict(host=platform.node(), stop=stop, rows=rows), stream, indent=2, allow_nan=False)
        stream.write("\n")
    if stop == "execution_error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
