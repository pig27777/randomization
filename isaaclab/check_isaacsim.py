"""Check the separately authorized Isaac Sim runtime without changing global settings."""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error(f"refusing to overwrite {args.output}")

start = time.perf_counter()
from isaacsim import SimulationApp

app = SimulationApp({
    "headless": True,
    "extra_args": ["--portable", "--portable-root", "/home/ubuntu/Dev/Envs/isaaclab/cache/kit"],
})
try:
    startup_s = time.perf_counter() - start
    for _ in range(10):
        app.update()
    import omni.usd
    stage_exists = omni.usd.get_context().get_stage() is not None
    result = {"headless": True, "updates": 10, "stage_exists": stage_exists, "startup_s": startup_s}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not stage_exists:
        raise RuntimeError("Isaac Sim started without a USD stage")
finally:
    app.close()
