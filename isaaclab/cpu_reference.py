"""Run the frozen CPU protocols unchanged; each formal case holds the shared lock."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/reference_v1/contents/segment"
TESTS = ("rest", "r1y", "r1z", "r2edge", "r2center", "r3")
CATEGORIES = ("hard", "bendable", "compressible")


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--test", choices=(*TESTS, "describe"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpu_reference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.batch:
        for category in CATEGORIES:
            for test in TESTS:
                for repeat in range(1, 4):
                    subprocess.run([
                        "flock", "/tmp/claude-gpu.lock", sys.executable, "-B", __file__,
                        "--category", category, "--test", test, "--repeat", str(repeat),
                        "--output", str(args.output),
                    ], check=True)
        return
    if args.category is None or args.test is None:
        parser.error("category/test are required without --batch")
    output = args.output / f"{args.category}-{args.test}-{args.repeat}.json"
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    if not REFERENCE.is_dir():
        parser.error("prepare and checksum the frozen v1 source archive first")
    sys.path.insert(0, str(REFERENCE))
    import mujoco
    from model import ANCHORS, build, theory_checks
    from mj_adapter import OMEGA_MAX
    import protocols

    desc = build(ANCHORS[args.category], omega_max=OMEGA_MAX)
    start = time.perf_counter()
    if args.test == "describe":
        result = theory_checks(desc)
    elif args.test == "rest":
        result = protocols.rest(desc.kappa, T=0.3, desc=desc)
    elif args.test.startswith("r1"):
        result = protocols.r1(desc.kappa, axis=args.test[-1], desc=desc)
    elif args.test.startswith("r2"):
        result = protocols.r2(desc.kappa, grip=args.test[2:], desc=desc)
    else:
        result = protocols.r3(desc.kappa, desc=desc)
    wall_s = time.perf_counter() - start
    payload = {
        "reference": "frozen v1", "mujoco": mujoco.__version__, "category": args.category,
        "test": args.test, "repeat": args.repeat, "wall_s": wall_s,
        "source_sha256": {str(p.relative_to(REFERENCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in REFERENCE.glob("*.py")},
        "description": asdict(desc), "result": result,
    }
    output.write_text(json.dumps(payload, default=serializable, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"file": str(output), "wall_s": wall_s, "result": result},
                     default=serializable, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
