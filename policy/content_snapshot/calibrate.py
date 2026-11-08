"""Build the response -> joint-parameter table (segment_model.md, "参数由整体响应定义" steps 2-3).

For every (anchor geometry, segment count) and each knob (E_y -> squeeze_y, E_z -> squeeze_z, D -> droop):
scale the knob around its current estimate, run R1-y / R1-z / R2, and store pairs
(measured response, knob / analytic knob for that response). internal_params() log-interpolates them.
Pass 2 repeats the sweeps with the other two knobs already corrected by pass 1.
A squeeze point is used only if the 5-15 % secant was measured before the lock; droop is measured with a
0.3 m lift so the hanging end never reaches the table (the protocol's 0.1 m lift caps droop near 0.3).

    python calibrate.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from model import ANCHORS, T5_SEGMENTS, TABLE_PATH, analytic_internal, internal_params

KNOBS = {"squeeze_y": ("E_y", "squeeze_stiffness_y"),
         "squeeze_z": ("E_z", "squeeze_stiffness_z"),
         "droop": ("D", "droop_ratio")}
MULTS = {"squeeze_y": [0.3, 0.55, 1.0, 1.8, 3.3], "squeeze_z": [0.3, 0.55, 1.0, 1.8, 3.3],
         "droop": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}
# wider sweep for the T6 base (bendable 6x3x2): squeeze over > 2 decades, droop 0.05-0.45
WIDE = {"squeeze_y": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 25.0],
        "droop": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]}
T6_KEY = ("bendable", (6, 3, 2))


def measure(args):
    anchor, segs, knob, mult, base = args
    from protocols import r1, r2
    k = replace(ANCHORS[anchor], segments=segs)
    internal = dict(base)
    name, field_ = KNOBS[knob]
    internal[name] = base[name] * mult
    t0 = time.time()
    if knob in ("squeeze_y", "squeeze_z"):
        r = r1(k, knob[-1], strain_stop=0.2, internal=internal)
        val = r["stiffness"]
        ok = (r["fail"] is None and np.isfinite(val) and val > 0 and r["secant_range"] == [0.05, 0.15]
              and (r["lock_strain"] is None or r["lock_strain"] >= 0.15))
    else:
        r = r2(k, "edge", internal=internal, lift_h=0.3)
        val = r.get("droop")
        ok = r["fail"] is None and val is not None and val > 0 and not r["table_touch"] and not r["dropped"]
    corr = None
    if ok:
        kk = replace(k, **{field_: float(val)})
        corr = internal[name] / analytic_internal(kk)[name]
    return dict(anchor=anchor, segments=list(segs), knob=knob, mult=mult, knob_value=internal[name],
                measured=None if val is None else float(val), ok=bool(ok), correction=corr, fail=r["fail"],
                table_touch=r.get("table_touch"), wall_s=round(time.time() - t0, 1))


def entries():
    for anchor, seg_list in T5_SEGMENTS.items():
        if ANCHORS[anchor].rigid:
            continue
        for segs in seg_list:
            yield anchor, tuple(segs)


def run_pass(table, workers):
    jobs = []
    for anchor, segs in entries():
        k = replace(ANCHORS[anchor], segments=segs)
        base = internal_params(k, table, geometry=anchor)
        base.pop("table", None)
        for knob in KNOBS:
            mults = WIDE.get(knob, MULTS[knob]) if (anchor, segs) == T6_KEY else MULTS[knob]
            jobs += [(anchor, segs, knob, m, base) for m in mults]
    with Pool(workers) as p:
        rows = list(p.imap_unordered(measure, jobs))
    new = {}
    for r in rows:
        key = f"{r['anchor']}:{'x'.join(map(str, r['segments']))}"
        if r["ok"]:
            new.setdefault(key, {}).setdefault(r["knob"], []).append([r["measured"], r["correction"]])
    for v in new.values():
        for knob in v:
            v[knob].sort()
    return new, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    t0 = time.time()
    table1, rows1 = run_pass({}, args.workers)
    table2, rows2 = run_pass(table1, args.workers)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(table2, indent=1))
    log = TABLE_PATH.parent / "calibration_runs.json"
    log.write_text(json.dumps({"pass1": rows1, "pass2": rows2, "wall_s": time.time() - t0}, indent=1))
    bad = [r for r in rows1 + rows2 if not r["ok"]]
    print(f"table -> {TABLE_PATH}  ({len(rows1) + len(rows2)} runs, {len(bad)} unusable, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
