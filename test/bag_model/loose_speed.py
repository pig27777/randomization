"""C3 (test/bag_model/REPORT.md round 1): speed of the MuJoCo loose-bag representations.

Single environment: real-time factor of one evaluate() (sim_s / wall_s, wrap included in wall_s).
Parallel: N evaluate() calls with random grasp parameters (test/policy/README.md ranges) and seeds on a
pool of P processes; evaluate/s = N / total wall time, as the policy line measures it (test/policy/REPORT.md).

Usage: python -B test/bag_model/loose_speed.py --bag plates|flex --content box_loose [--set JSON] [--name label] [--n 56] [--procs 28]
       [--out c3_speed.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

from pathlib import Path
import sys

BAG = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(BAG))

from loose_protocols import KRAFT_SIZE, OUT, content_kappa, evaluate
from mj_loose import LooseBag

RANGES = [(-0.5, 0.5), (-0.5, 0.5), (0.0, math.pi), (2.0, 50.0), (0.05, 0.3), (0.5, 4.0)]


def spec_for(bag_kind, content, fill, overrides):
    k = content_kappa(content)
    bag = LooseBag(kind=bag_kind, size=KRAFT_SIZE) if content.startswith("box") else LooseBag(kind=bag_kind, fill=fill)
    bag = replace(bag, **overrides)
    return {"kappa": k.to_dict(), "cls": "hard" if k.rigid else content, "bag": asdict(bag)}


def _one(args):
    spec, params, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        r = evaluate(spec, params, seed)
    except Exception as err:          # recorded, counted as an error in the summary
        return {"error": repr(err), "params": params, "seed": seed}
    r.update(params=params, seed=seed)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, choices=["plates", "flex"])
    ap.add_argument("--content", default="box_loose")
    ap.add_argument("--fill", type=float, default=0.75)
    ap.add_argument("--n", type=int, default=56)
    ap.add_argument("--procs", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", default="{}", help="JSON overrides of LooseBag fields")
    ap.add_argument("--dt", type=float, default=None, help="scene step (default: the content model's 0.4 ms)")
    ap.add_argument("--name", default=None, help="label in the summary (default: --bag)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = spec_for(a.bag, a.content, a.fill, json.loads(a.set))
    spec["dt"] = a.dt
    rng = np.random.default_rng(a.seed)
    jobs = [(spec, [float(rng.uniform(lo, hi)) for lo, hi in RANGES], int(rng.integers(1 << 30))) for _ in range(a.n)]
    single = _one((spec, [0.0, 0.0, math.pi / 2, 20.0, 0.1, 2.0], 0))
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    with Pool(a.procs) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    wall = time.perf_counter() - t0
    ok = [r for r in res if "error" not in r]
    outcomes = {}
    for r in ok:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    summary = dict(bag=a.name or a.bag, set=json.loads(a.set), content=a.content, dt=a.dt or 4e-4, single_rtf=single.get("sim_s", 0) / single.get("wall_s", 1),
                   single_wall_s=single.get("wall_s"), single_sim_s=single.get("sim_s"), single_outcome=single.get("outcome"),
                   n=a.n, procs=a.procs, wall_s=wall, evals_per_s=a.n / wall, errors=len(res) - len(ok),
                   median_wall_s=float(np.median([r["wall_s"] for r in ok])) if ok else None,
                   median_sim_s=float(np.median([r["sim_s"] for r in ok])) if ok else None,
                   outcomes=outcomes, loadavg_before=load0, loadavg_after=os.getloadavg(), cpus=os.cpu_count())
    print(json.dumps(summary), flush=True)
    if a.out:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / a.out, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
        with open(OUT / a.out.replace(".jsonl", "_runs.jsonl"), "a") as fh:
            for r in res:
                fh.write(json.dumps(dict(r, bag=a.name or a.bag, content=a.content), default=str) + "\n")


if __name__ == "__main__":
    main()
