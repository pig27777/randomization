"""Policy line round 1 (test/policy/REPORT.md): P1 environment check, R1 / R3 training, P4 cross evaluation.

    python run.py p1
    python run.py train --dist R1 --tag main          # same --n1/--n2 for R1 and R3
    python run.py eval --tag main                     # P4: both policies on R1 and R3 test objects
Formal runs: flock /tmp/claude-gpu.lock python run.py ...
Results: results/<tag>/ (jsonl of every evaluation, policy weights, summaries).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import env
from env import evaluate, observe, sample_object
from learn import Model, decode
from model import ANCHORS, Kappa

RES = Path(__file__).resolve().parent / "results"
WORKERS = 28
TRAIN_SEED = {"R1": 1_000_000, "R3": 2_000_000}     # object / placement seeds of the training samples
TEST_SEED = {"R1": 7_000_000, "R3": 8_000_000}      # P4 test objects, disjoint from training


def _eval_job(job):
    spec, params, seed, meta = job
    r = evaluate(spec, params, seed)
    return {**meta, "seed": seed, "params": [float(x) for x in params], "cls": spec["cls"],
            "obs": observe(spec).tolist(), **{k: v for k, v in r.items() if k != "cls"}}


def run_jobs(jobs, out_path, label):
    rows, t0 = [], time.time()
    with Pool(WORKERS, maxtasksperchild=50) as pool, open(out_path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_eval_job, jobs, chunksize=1)):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 100 == 0:
                c = Counter(x["outcome"] for x in rows)
                print(f"[{label}] {i + 1}/{len(jobs)} {time.time() - t0:.0f} s {dict(c)}", flush=True)
    return rows, time.time() - t0


def objects(dist, n, base):
    return [(sample_object(np.random.default_rng(base + i), dist), base + i) for i in range(n)]


# ------------------------------------------------------------------ P1

def p1(args):
    out = RES / "p1"
    out.mkdir(parents=True, exist_ok=True)
    box = Kappa(size=(0.25, 0.18, 0.10), mass=1.0, segments=(1, 1, 1), rigid=True)
    objs = {"hard_box": {"kappa": box.to_dict(), "cls": "hard", "friction": 0.8},
            **{k: {"kappa": ANCHORS[k].to_dict(), "cls": k, "friction": 0.8} for k in ANCHORS}}
    nominal = (0.0, 0.0, math.pi / 2, 20.0, 0.1, 2.0)
    sets = [(name, "nominal_20N", nominal) for name in objs]
    sets += [(name, "low_2N", (0.0, 0.0, math.pi / 2, 2.0, 0.1, 2.0)) for name in ("hard_box", "hard", "bendable")]
    sets += [("compressible", "high_40N", (0.0, 0.0, math.pi / 2, 40.0, 0.1, 2.0)),
             ("bendable", "along_long_10N", (0.0, 0.0, 0.0, 10.0, 0.1, 1.0))]
    jobs = [(objs[name], p, s, {"object": name, "set": tag}) for name, tag, p in sets for s in range(10)]
    path = out / "p1.jsonl"
    path.unlink(missing_ok=True)
    rows, wall = run_jobs(jobs, path, "p1")
    summary = {}
    for name, tag, _ in sets:
        rr = [r for r in rows if r["object"] == name and r["set"] == tag]
        summary[f"{name}/{tag}"] = {"outcomes": dict(Counter(r["outcome"] for r in rr)),
                                    "max_force_median": float(np.median([r["max_grip_force_N"] for r in rr if r.get("max_grip_force_N") is not None])),
                                    "wall_median_s": float(np.median([r["wall_s"] for r in rr])),
                                    "reasons": sorted({r["reason"] for r in rr if r["reason"]})}
    n_fail = sum(r["outcome"] == "sim_fail" for r in rows)
    summary["_total"] = {"n": len(rows), "sim_fail": n_fail, "sim_fail_rate": n_fail / len(rows), "wall_s": wall}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ training

def fit(rows, seed):
    rows = [r for r in rows if r["outcome"] != "sim_fail"]
    obs = np.array([r["obs"] for r in rows])
    from learn import encode
    z = encode(np.array([r["params"] for r in rows]))
    y = np.array([r["outcome"] == "success" for r in rows], float)
    return Model.fit(obs, z, y, seed=seed)


def _act_job(a):
    model_path, obs, seed = a
    return Model.load(model_path).act(np.array(obs), seed=seed)


def act_many(model_path, obs_list, seed=0):
    with Pool(WORKERS) as pool:
        return pool.map(_act_job, [(str(model_path), list(o), seed) for o in obs_list])


def train(args):
    out = RES / args.tag / args.dist
    out.mkdir(parents=True, exist_ok=True)
    data = out / "samples.jsonl"
    data.unlink(missing_ok=True)
    rng = np.random.default_rng(TRAIN_SEED[args.dist])
    base = TRAIN_SEED[args.dist]
    log = {"dist": args.dist, "n1": args.n1, "n2": args.n2, "noise": args.noise, "rounds": []}

    # round 1: uniform exploration
    objs = objects(args.dist, args.n1, base)
    jobs = [(spec, tuple(decode(rng.uniform(-1, 1, 6))), seed, {"round": 1}) for spec, seed in objs]
    rows, wall = run_jobs(jobs, data, f"{args.dist} r1")
    log["rounds"].append({"n": len(rows), "wall_s": wall, "outcomes": dict(Counter(r["outcome"] for r in rows)),
                          "sim_s": float(sum(r.get("sim_s") or 0 for r in rows))})
    t0 = time.time()
    model = fit(rows, seed=1)
    model.save(out / "model_r1.npz")
    log["rounds"][-1]["fit_s"] = time.time() - t0

    # round 2: around the current policy (Gaussian noise in z), 25 % uniform
    objs = objects(args.dist, args.n2, base + args.n1)
    best = act_many(out / "model_r1.npz", [observe(s) for s, _ in objs])
    jobs = []
    for (spec, seed), (z, _) in zip(objs, best):
        zz = rng.uniform(-1, 1, 6) if rng.random() < 0.25 else np.clip(z + rng.normal(0, args.noise, 6), -1, 1)
        jobs.append((spec, tuple(decode(zz)), seed, {"round": 2}))
    rows2, wall = run_jobs(jobs, data, f"{args.dist} r2")
    log["rounds"].append({"n": len(rows2), "wall_s": wall, "outcomes": dict(Counter(r["outcome"] for r in rows2)),
                          "sim_s": float(sum(r.get("sim_s") or 0 for r in rows2))})
    t0 = time.time()
    model = fit(rows + rows2, seed=2)
    model.save(out / "model.npz")
    log["fit_s"] = time.time() - t0
    log["samples_used"] = sum(r["outcome"] != "sim_fail" for r in rows + rows2)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


# ------------------------------------------------------------------ P4

def evaluate_policies(args):
    out = RES / args.tag / "p4"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "p4.jsonl"
    path.unlink(missing_ok=True)
    jobs = []
    for test in ("R1", "R3"):
        objs = objects(test, args.n, TEST_SEED[test])
        for pol in ("R1", "R3"):
            acts = act_many(RES / args.tag / pol / "model.npz", [observe(s) for s, _ in objs])
            for i, ((spec, seed), (z, p)) in enumerate(zip(objs, acts)):
                jobs.append((spec, tuple(decode(z)), seed,
                             {"policy": pol, "test": test, "obj": i, "pred_success": p}))
    run_jobs(jobs, path, "p4")
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["p1", "train", "eval"])
    ap.add_argument("--dist", choices=["R1", "R3"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n1", type=int, default=2000)
    ap.add_argument("--n2", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    globals()["WORKERS"] = args.workers
    {"p1": p1, "train": train, "eval": evaluate_policies}[args.cmd](args)


if __name__ == "__main__":
    main()
