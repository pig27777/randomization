"""Round-2 runs (test/content_model/REPORT.md, round 2): S1 stability scan, S5 surface friction, S6 = T0-T8.
S3 (MJWarp) is mjwarp_validate.py; S4 (table) is calibrate.py.

    python test/content_model/run_round2.py s1 s5 s6 ...
Results: results/round2/<name>.json. Formal runs: flock /tmp/claude-gpu.lock python test/content_model/run_round2.py ...
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import sys

CONTENT = Path(__file__).resolve().parents[2] / "contents" / "segment"
sys.path.insert(0, str(CONTENT))

import numpy as np

import run_round1 as R1
from model import ANCHORS, T5_SEGMENTS, Kappa, internal_params, load_table
from sampling import check_kappa, sample_kappa

RES = CONTENT / "results" / "round2"
WORKERS = 30


def save(name, data):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(R1._clean(data), indent=1, ensure_ascii=False))
    print(f"-> results/round2/{name}.json")


def run_jobs(fn, jobs):
    with Pool(min(WORKERS, len(jobs))) as p:
        return list(p.imap(fn, jobs, chunksize=1))


# ------------------------------------------------------------------ S1 / T7

def s1(n=200, seed=2026):
    rng = np.random.default_rng(seed)
    kappas = [sample_kappa(rng, "mixed").to_dict() for _ in range(n)]
    old = json.loads((R1.RES / "t7.json").read_text())
    old_k = [f["kappa"] for f in old["failures"]]
    table = load_table()
    old_excluded = [check_kappa(Kappa.from_dict(k), None, table) for k in old_k]
    rows = run_jobs(R1.t7_case, list(enumerate(kappas + old_k)))
    new, olds = rows[:n], rows[n:]

    def fail_of(r):
        return r["exception"] or (r["rest"] or {}).get("fail") or (r["r3"] or {}).get("fail")

    fails = [{"i": r["i"], "failure": fail_of(r), "kappa": r["kappa"],
              "phase": _phase(r)} for r in new if fail_of(r)]
    old_res = [{"i": i, "round1_failure": f["failure"], "now": fail_of(r), "excluded_by": ex}
               for i, (f, r, ex) in enumerate(zip(old["failures"], olds, old_excluded))]
    save("s1_t7", {"n": n, "seed": seed, "n_fail": len(fails), "fail_rate": len(fails) / n, "failures": fails,
                   "old13": old_res, "rows": new})


def _phase(r):
    r3 = r.get("r3") or {}
    if (r.get("rest") or {}).get("fail"):
        return "rest"
    t, ev = r3.get("fail_t"), r3.get("events") or {}
    if t is None:
        return None
    past = [e for e, v in ev.items() if v <= t]
    return max(past, key=lambda e: ev[e]) if past else "closing"


# ------------------------------------------------------------------ S5 surface friction

S5_FRICTION = [0.2, 0.3, 0.45, 0.65, 0.9]


def s5_case(args):
    from protocols import r3
    name, mu = args
    k = replace(ANCHORS[name], surface_friction=mu)
    r = r3(k, desc=R1.desc_of(k, name))
    return {"anchor": name, "surface_friction": mu, **{q: r.get(q) for q in
            ("fail", "slip_mm", "dropped", "grip_force", "pad_fz_ratio", "com_swing_mm")}}


def s5():
    rows = run_jobs(s5_case, [(n, mu) for n in ("hard", "compressible") for mu in S5_FRICTION])
    out = {"rows": rows}
    for n in ("hard", "compressible"):
        s = [r["slip_mm"] for r in rows if r["anchor"] == n]
        out[f"{n}_monotone"] = all(v is not None for v in s) and bool(np.all(np.diff(s) < 0))
    save("s5_friction", out)


# ------------------------------------------------------------------ S6 = T0-T8 with the round-2 protocol

def s6():
    R1.RES = RES                          # run_round1 writers go to results/round2/
    R1.WORKERS = WORKERS
    for name, fn in (("t0", R1.t0), ("t1", R1.t1), ("t2", R1.t2_t4), ("t5", t5), ("t6", R1.t6), ("t8", R1.t8)):
        t = time.time()
        fn()
        print(f"{name}: {time.time() - t:.0f} s", flush=True)


def t5():
    """T5 with out-of-table targets listed separately (no extrapolation)."""
    jobs = [(n, tuple(s)) for n, segs in T5_SEGMENTS.items() for s in segs]
    rows = run_jobs(R1.t5_case, jobs)
    table = load_table()
    for r in rows:
        k = replace(ANCHORS[r["anchor"]], segments=tuple(r["segments"]))
        r["in_table"] = internal_params(k, table, r["anchor"])["in_table"] if not k.rigid else None
        r["error"] = {q: (None if r["measured"][q] is None or k.rigid else r["measured"][q] / r["target"][q] - 1)
                      for q in ("squeeze_y", "squeeze_z", "droop")}
    R1.save("t5", {"rows": rows})


TESTS = {"s1": s1, "s5": s5, "s6": s6}


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="+", help="s1 s5 s6")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    WORKERS = args.workers
    for n in args.tests:
        t = time.time()
        TESTS[n]()
        print(f"{n}: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
