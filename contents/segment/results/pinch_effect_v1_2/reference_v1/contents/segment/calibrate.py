"""Build the response -> joint-parameter table (segment_model.md, "参数由整体响应定义" steps 2-3).

Round 2: one entry per (category anchor geometry, segment count) for every segment count the sampler can
draw (sampling.BOUNDS). For each knob (E_y -> squeeze_y, E_z -> squeeze_z, D -> droop) the knob is swept over
a wide log range with the other two at the anchor targets; each valid run stores (measured response,
knob / analytic knob for that response). Pass 2 repeats with the other knobs corrected by pass 1.
Validity: R1 secant measured (no lower bound, no failure); R2 without failure, drop or table contact.
internal_params() interpolates, and sampling.check_kappa() rejects targets outside the measured range.

    python calibrate.py --output-dir results/<new-run-id> [--workers N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from model import ANCHORS, analytic_internal, internal_params
from sampling import BOUNDS, lock_cap

KNOBS = {"squeeze_y": ("E_y", "squeeze_stiffness_y"),
         "squeeze_z": ("E_z", "squeeze_stiffness_z"),
         "droop": ("D", "droop_ratio")}
# knob multipliers on the analytic value at the anchor target: wide enough to cover the sampling bounds
MULTS = {"squeeze_y": list(np.geomspace(0.05, 20, 9)), "squeeze_z": list(np.geomspace(0.05, 20, 9)),
         "droop": list(np.geomspace(0.1, 300, 10))}


def anchor_for(category, segs):
    a = ANCHORS[category]
    return replace(a, segments=segs, lock_strain_y=min(a.lock_strain_y, lock_cap(segs[1])),
                   lock_strain_z=min(a.lock_strain_z, lock_cap(segs[2])))


def measure(args):
    category, segs, knob, mult, base = args
    from protocols import r1, r2
    k = anchor_for(category, segs)
    internal = dict(base)
    name, field_ = KNOBS[knob]
    internal[name] = base[name] * mult
    t0 = time.time()
    if knob in ("squeeze_y", "squeeze_z"):
        r = r1(k, knob[-1], strain_stop=0.25, internal=internal)
        val = r["stiffness"]
        ok = r["fail"] is None and np.isfinite(val) and val > 0 and not r["lower_bound"]
    else:
        r = r2(k, "edge", internal=internal)
        val = r.get("droop")
        ok = r["fail"] is None and val is not None and 0 < val < 0.95 and not r["table_touch"] and not r["dropped"]
    corr = None
    if ok:
        corr = internal[name] / analytic_internal(replace(k, **{field_: float(val)}))[name]
    return dict(category=category, segments=list(segs), knob=knob, mult=float(mult), knob_value=internal[name],
                measured=None if val is None or not np.isfinite(val) else float(val), ok=bool(ok), correction=corr,
                fail=r["fail"], wall_s=round(time.time() - t0, 1))


def entries(categories):
    for cat in categories:
        b = BOUNDS[cat]
        for segs in itertools.product(b["nx"], b["ny"], b["nz"]):
            yield cat, tuple(int(v) for v in segs)


def run_pass(table, workers, categories):
    jobs = []
    for cat, segs in entries(categories):
        base = internal_params(anchor_for(cat, segs), table, geometry=cat)
        base = {q: base[q] for q in ("E_y", "E_z", "D")}
        for knob in KNOBS:
            jobs += [(cat, segs, knob, m, base) for m in MULTS[knob]]
    with Pool(workers) as p:
        rows = list(p.imap_unordered(measure, jobs, chunksize=1))
    new = {}
    for r in rows:
        if r["ok"]:
            key = f"{r['category']}:{'x'.join(map(str, r['segments']))}"
            new.setdefault(key, {}).setdefault(r["knob"], []).append([r["measured"], r["correction"]])
    for v in new.values():
        for knob in v:
            v[knob].sort()
    return new, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--categories", nargs="+", default=["bendable", "compressible"])
    ap.add_argument("--output-dir", type=Path, required=True, help="new result directory; never overwrite the active table")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    table1, rows1 = run_pass({}, args.workers, args.categories)
    table2, rows2 = run_pass(table1, args.workers, args.categories)
    table_path = args.output_dir / "table.json"
    table_path.write_text(json.dumps(table2, indent=1))
    log = args.output_dir / "calibration_runs.json"
    log.write_text(json.dumps({"pass1": rows1, "pass2": rows2, "wall_s": time.time() - t0}, indent=1))
    bad = [r for r in rows2 if not r["ok"]]
    print(f"table -> {table_path}  ({len(rows1) + len(rows2)} runs, pass-2 unusable {len(bad)}, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
