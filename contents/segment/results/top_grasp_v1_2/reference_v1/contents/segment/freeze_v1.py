"""Reproducible v1 acceptance; formal invocations run under the project flock.

Results use a new named directory; existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import mujoco

from calibrate import KNOBS, anchor_for, entries
from model import ANCHORS, TABLE_PATH, T5_SEGMENTS, analytic_internal, build, internal_params, load_table
from mj_adapter import OMEGA_MAX
from protocols import r1, r2, r3, rest

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "freeze_v1"


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


def hashes():
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(clean(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def measure(job):
    cat, seg, knob, internal = job
    k = anchor_for(cat, seg)
    t0 = time.perf_counter()
    r = r2(k, "edge", internal=internal) if knob == "droop" else r1(k, knob[-1], internal=internal)
    val = r.get("droop" if knob == "droop" else "stiffness")
    valid = r["fail"] is None and val is not None and np.isfinite(val) and val > 0
    if knob == "droop":
        valid = valid and not r.get("table_touch", True) and not r.get("dropped", True) and abs(r.get("pad_fz_ratio", 0) - 1) <= .05
    else:
        valid = valid and not r["lower_bound"]
    field = KNOBS[knob][1]
    return dict(category=cat, segments=seg, kappa=k.to_dict(), knob=knob, target=getattr(k, field), internal=internal,
                measured=val, valid=bool(valid), wall_s=time.perf_counter() - t0, result=r)


def run_jobs(jobs, workers):
    rows = []
    with Pool(workers) as pool:
        for row in pool.imap_unordered(measure, jobs):
            rows.append(row)
            print(f"{row['category']} {row['segments']} {row['knob']}: {row['measured']} valid={row['valid']} {row['wall_s']:.1f}s", flush=True)
    return rows


def random_case(job):
    index, k, destination, table = job
    destination = Path(destination)
    destination.mkdir()
    os.chdir(destination)
    started = time.perf_counter()
    result = dict(index=index, kappa=k.to_dict())
    try:
        desc = build(k, table=table, omega_max=OMEGA_MAX)
        result["rest"] = rest(k, desc=desc)
        result["r3"] = r3(k, desc=desc)
        failure = result["r3"]
        t = failure.get("fail_t")
        if t is not None:
            events = failure.get("events", {})
            active = [(when, name) for name, when in events.items() if when <= t]
            failure["failure_phase"] = max(active)[1] if active else "settle" if t < .3 else "close"
        result["confirmed_failure"] = bool(result["rest"]["fail"] or result["r3"]["fail"])
        result["execution_error"] = False
    except Exception:
        # Preserve infrastructure errors as errors, not invented simulation results;
        # other independent samples still run and acceptance remains unverified.
        result.update(execution_error=True, traceback=traceback.format_exc(), confirmed_failure=False)
    result["wall_s"] = time.perf_counter() - started
    save(destination / "result.json", result)
    return result


def random_scan(out, count, seed, workers, table):
    from sampling import sample_kappa
    rng = np.random.default_rng(seed)
    samples = [sample_kappa(rng, "mixed", table) for _ in range(count)]
    save(out / "samples.json", dict(seed=seed, kappas=[k.to_dict() for k in samples]))
    jobs = [(i, k, str(out / f"case_{i:03d}"), table) for i, k in enumerate(samples)]
    rows = []
    with Pool(workers) as pool:
        for row in pool.imap_unordered(random_case, jobs):
            rows.append(row)
            print(f"random {row['index']}: fail={row['confirmed_failure']} error={row['execution_error']} {row['wall_s']:.1f}s", flush=True)
    save(out / "summary.json", dict(total=count, confirmed_failures=sum(r["confirmed_failure"] for r in rows),
                                    execution_errors=sum(r["execution_error"] for r in rows), rows=rows))


def evaluate_cases(out, count, observe=True, table=None):
    import math
    from policy_protocol import load_protocol
    evaluate, protocol_hash = load_protocol(table)
    rows = []
    for cat, k in ANCHORS.items():
        for seed in range(count):
            trace = []
            diagnostics = []
            def observer(sim, phase, t, forces):
                gripped = phase in ("lift", "carry", "lower")
                speed = sim.velocity_diagnostic(sim.d.qvel[sim.pad_v["pA"]], gripped)
                if max(speed.values()) > 1 and not any(d["phase"] == phase for d in diagnostics):
                    diagnostics.append(dict(t=t, phase=phase, **speed, confirmed_failure=sim.failure()))
                if cat != "bendable":
                    return
                outer = [i for i, seg in enumerate(sim.desc.segments)
                         if seg.idx[1] in (0, k.segments[1] - 1) and abs(seg.pos[0]) <= .5 * .04 + seg.half[0]]
                rot = sim.d.xmat[[sim.seg_bodies[i] for i in outer]].reshape(-1, 3, 3)
                roll = np.degrees(np.arctan2(rot[:, 2, 1], rot[:, 2, 2]))
                grip = .5 * (sim.pad_pos("pA") + sim.pad_pos("pB"))
                trace.append(dict(t=t, phase=phase, forces=forces.copy(), com=sim.com().tolist(),
                                  outer_roll_deg=roll.tolist(), max_relative_angle_deg=sim.max_relative_angle_deg(),
                                  outer_centers_from_grip=(sim.d.xpos[[sim.seg_bodies[i] for i in outer]] - grip).tolist(),
                                  grip_contacts=[(p, int(b), sim.point_world(b, local).tolist())
                                                 for p, b, local, _ in sim.grip_anchors()]))
            spec = dict(kappa=k.to_dict(), cls=cat, friction=k.surface_friction)
            grasp = (0, 0, math.pi / 2, 20 if cat != "compressible" else 10, .1, 2.)
            result = evaluate(spec, grasp, seed, observer=observer if observe else None)
            row = dict(category=cat, seed=seed, grasp=grasp, result=result, trace=trace,
                       velocity_diagnostics=diagnostics,
                       policy_source_sha256=protocol_hash)
            save(out / f"evaluate_{cat}_{seed:02d}.json", row)
            rows.append(row)
            print(f"evaluate {cat} seed={seed}: {result}", flush=True)
    save(out / "summary.json", dict(policy_source_sha256=protocol_hash, rows=rows,
                                    median_wall_s={cat: np.median([r["result"]["wall_s"] for r in rows if r["category"] == cat])
                                                   for cat in ANCHORS}))


def compare_case(job):
    """Paired solver responses using identical internal parameters and 10 ms iteration samples."""
    import mj_adapter as adapter
    import protocols as protocols
    category, solver, test, table = job
    adapter.SOLVER = f'solver="{solver}" iterations="100" ls_iterations="20" tolerance="1e-8"'
    iterations = []
    step = protocols.Runner.step
    def observed_step(runner, *args, **kwargs):
        recorded = step(runner, *args, **kwargs)
        if recorded:
            iterations.append(int(runner.sim.d.solver_niter.max()))
        return recorded
    protocols.Runner.step = observed_step
    started = time.perf_counter()
    k = ANCHORS[category]
    desc = build(k, table=table, omega_max=OMEGA_MAX)
    try:
        result = (r1(k, test[-1], desc=desc) if test.startswith("r1") else
                  r2(k, "edge", desc=desc) if test == "r2" else r3(k, desc=desc))
    finally:
        protocols.Runner.step = step
    return dict(category=category, solver=solver, test=test, result=result,
                iteration_samples=len(iterations), max_iterations=max(iterations, default=0),
                median_iterations=float(np.median(iterations)),
                at_iteration_cap=sum(v >= 100 for v in iterations), wall_s=time.perf_counter() - started)


def solver_compare(out, table, workers):
    jobs = [(cat, solver, test, table) for cat in ("bendable", "compressible")
            for solver in ("CG", "Newton") for test in ("r1y", "r1z", "r2", "r3")]
    rows = []
    with Pool(workers) as pool:
        for row in pool.imap_unordered(compare_case, jobs):
            rows.append(row)
            print(f"compare {row['category']} {row['solver']} {row['test']}: "
                  f"iterations<={row['max_iterations']} fail={row['result']['fail']}", flush=True)
    save(out / "summary.json", dict(rows=rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["probe", "calibrate", "m3", "random", "evaluate", "compare"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-root", type=Path, default=BASE)
    ap.add_argument("--solver", choices=("CG", "Newton"), default="Newton")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--mults", type=float, nargs="+", default=[.1, .3, 1., 3., 10.])
    ap.add_argument("--all-segments", action="store_true")
    ap.add_argument("--categories", nargs="+", choices=["bendable", "compressible"],
                    default=["bendable", "compressible"])
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    import mj_adapter as adapter
    adapter.SOLVER = f'solver="{args.solver}" iterations="100" ls_iterations="20" tolerance="1e-8"'
    out = (args.output_root / args.run_id).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    metadata = dict(action=args.action, started_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=hashes(), timestep_s=.0004,
                    mujoco_version=mujoco.__version__, python_version=platform.python_version(),
                    load_average=os.getloadavg(),
                    solver_options=adapter.SOLVER,
                    args=vars(args) | {"table": str(args.table) if args.table else None,
                                      "output_root": str(args.output_root)})
    input_table = args.table or TABLE_PATH
    metadata["input_table"] = dict(path=str(input_table), sha256=hashlib.sha256(input_table.read_bytes()).hexdigest())
    save(out / "manifest.json", metadata)
    table = load_table(input_table)
    if args.action == "random":
        random_scan(out, args.count, args.seed, args.workers, table)
        return
    if args.action == "evaluate":
        evaluate_cases(out, args.count, table=table)
        return
    if args.action == "compare":
        solver_compare(out, table, args.workers)
        return
    configs = [(cat, seg) for cat in args.categories for seg in T5_SEGMENTS[cat]]
    if args.all_segments:
        configs = list(entries(args.categories))
    if args.action == "m3":
        configs += [("hard", seg) for seg in T5_SEGMENTS["hard"]]
    jobs = []
    if args.action == "probe":
        for cat in ("bendable", "compressible"):
            k = ANCHORS[cat]
            for base in (analytic_internal(k), internal_params(k, table, cat)):
                for knob in KNOBS:
                    jobs.append((cat, k.segments, knob, base))
    else:
        for cat, seg in configs:
            k = anchor_for(cat, seg)
            base = internal_params(k, table, cat, strict=False) if args.table else analytic_internal(k)
            for knob, (name, _) in KNOBS.items():
                for mult in (args.mults if args.action == "calibrate" else [1.]):
                    jobs.append((cat, seg, knob, {q: base[q] * (mult if q == name else 1) for q in ("E_y", "E_z", "D")}))
    rows = run_jobs(jobs, args.workers)
    save(out / "measurements.json", rows)
    if args.action == "calibrate":
        active = {f"{cat}:{'x'.join(map(str, seg))}" for cat, seg in configs}
        new = {key: value for key, value in table.items() if key not in active} if args.table else {}
        from dataclasses import replace
        for row in rows:
            if row["valid"]:
                cat, seg, knob = row["category"], row["segments"], row["knob"]
                name, field = KNOBS[knob]
                ref = analytic_internal(replace(anchor_for(cat, seg), **{field: row["measured"]}))[name]
                key = f"{cat}:{'x'.join(map(str, seg))}"
                new.setdefault(key, {}).setdefault(knob, []).append([row["measured"], row["internal"][name] / ref])
        for key in active & new.keys():
            entry = new[key]
            for knob in KNOBS:
                points = entry.get(knob, [])
                points.sort()
            entry["mapping_version"] = args.run_id
        save(out / "table.json", new)
    elif args.action == "m3":
        soft, hard = [], []
        for r in rows:
            k = anchor_for(r["category"], r["segments"])
            if k.rigid:
                r["hit"] = bool(r["result"]["fail"] is None and (
                    abs(r["measured"]) <= .02 and r["result"].get("suspended", False)
                    if r["knob"] == "droop" else r["result"]["lower_bound"] and r["measured"] >= r["target"]))
                hard.append(r)
                continue
            r["in_table"] = internal_params(k, table, r["category"], strict=False)["in_table"][r["knob"]]
            r["relative_error"] = r["measured"] / r["target"] - 1 if r["measured"] is not None else None
            r["hit"] = bool(r["valid"] and r["in_table"] and abs(r["relative_error"]) <= .2)
            soft.append(r)
        save(out / "summary.json", dict(hits=sum(r["hit"] for r in soft), total=18,
                                        hard_pass=all(r["hit"] for r in hard), rows=rows))


if __name__ == "__main__":
    main()
