"""Round-1 tests T0-T8 on the MuJoCo adapter (test/content_model/REPORT.md). Newton T0: newton_check.py.

    python run_round1.py t0 t1 t2 ...      # or: all
Results: results/<test>.json. Formal runs go through: flock /tmp/claude-gpu.lock python run_round1.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from model import ANCHORS, T5_SEGMENTS, G, Kappa, build, load_table, theory_checks
from sampling import sample_kappa
from mj_adapter import OMEGA_MAX, scene, Sim

RES = Path(__file__).resolve().parent / "results"
WORKERS = 24


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "curve"}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def save(name, data):
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(_clean(data), indent=1, ensure_ascii=False))
    print(f"-> results/{name}.json")


def desc_of(k: Kappa, geometry=None):
    return build(k, omega_max=OMEGA_MAX, table=load_table(), geometry=geometry)


def run_jobs(fn, jobs):
    with Pool(min(WORKERS, len(jobs))) as p:
        return list(p.imap(fn, jobs))


# ------------------------------------------------------------------ T0 (MuJoCo part)

def t0_case(args):
    name, k = args
    desc = desc_of(k, name.split("/")[0])
    th = theory_checks(desc)
    built = scene(desc, (0, 0, 0.5), table=False, gravity=False)
    sim = Sim(built)
    m, d = sim.m, sim.d
    M = float(sum(m.body_mass[b] for b in sim.obj_bodies))
    com = sim.com() - np.array([0, 0, 0.5])
    # whole-object inertia about the COM, world axes
    I = np.zeros((3, 3))
    for b in sim.obj_bodies:
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - (com + [0, 0, 0.5])
        I += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    L = np.array(k.size)
    I_box = k.mass / 12 * np.array([L[1] ** 2 + L[2] ** 2, L[0] ** 2 + L[2] ** 2, L[0] ** 2 + L[1] ** 2])
    seg_err = 0.0
    if not desc.rigid:
        for i, s in enumerate(desc.segments):
            b = sim.seg_bodies[i]
            seg_err = max(seg_err, float(np.max(np.abs(np.sort(m.body_inertia[b]) - np.sort(s.inertia)) / np.max(s.inertia))))
    out = {"case": name, "segments": k.segments, "rigid": k.rigid, "mobile": k.mobile, "com_offset": k.com_offset,
           "mass_target": k.mass, "mass_desc": th["mass"], "mass_mujoco": M,
           "com_target": th["com_target"], "com_desc": th["com"], "com_mujoco": com.tolist(),
           "com_err_mm": float(np.linalg.norm(com - np.array(th["com_target"])) * 1e3),
           "segment_inertia_max_rel_err": seg_err,
           "n_bodies": m.nbody, "n_tendons": m.ntendon, "n_fixed_joints": built.n_fixed, "n_joints": len(desc.joints)}
    if not k.mobile and not any(k.com_offset):
        out["inertia_mujoco"] = np.diag(I).tolist()
        out["inertia_box"] = I_box.tolist()
        out["inertia_max_rel_err"] = float(np.max(np.abs(np.diag(I) - I_box) / I_box))
    return out


def t0():
    cases = [(n, ANCHORS[n]) for n in ANCHORS]
    cases += [("bendable/6x4x3", replace(ANCHORS["bendable"], segments=(6, 4, 3))),
              ("compressible/com", replace(ANCHORS["compressible"], com_offset=(0.15, -0.1, 0.05))),
              ("compressible/mobile", replace(ANCHORS["compressible"], mobile=True)),
              ("hard/com_mobile", replace(ANCHORS["hard"], com_offset=(0.2, 0.0, -0.1), mobile=True)),
              ("bendable/1x1x1", replace(ANCHORS["bendable"], segments=(1, 1, 1)))]
    rows = run_jobs(t0_case, cases)
    save("t0_mujoco", {"cases": rows})


# ------------------------------------------------------------------ T1 / energy

def t1_case(args):
    from protocols import energy, rest
    name, k = args
    desc = desc_of(k, name)
    return {"anchor": name, "rest": rest(k, T=2.0, desc=desc), "energy": energy(k, T=1.0, desc=desc)}


def t1():
    save("t1", {"rows": run_jobs(t1_case, [(n, ANCHORS[n]) for n in ANCHORS])})


# ------------------------------------------------------------------ T2-T4

def anchor_test(args):
    from protocols import r1, r2, r3
    name, test = args
    k = ANCHORS[name]
    desc = desc_of(k, name)
    fn = {"R1y": lambda: r1(k, "y", desc=desc), "R1z": lambda: r1(k, "z", desc=desc),
          "R2edge": lambda: r2(k, "edge", desc=desc), "R2center": lambda: r2(k, "center", desc=desc),
          "R3": lambda: r3(k, desc=desc)}[test]
    t0_ = time.time()
    r = fn()
    r["wall_s"] = time.time() - t0_
    r["anchor"] = name
    r["internal"] = desc.internal
    return r


def t2_t4():
    jobs = [(n, t) for n in ANCHORS for t in ("R1y", "R1z", "R2edge", "R2center", "R3")]
    rows = run_jobs(anchor_test, jobs)
    for test, tests in (("t2", ("R1y", "R1z")), ("t3", ("R2edge", "R2center")), ("t4", ("R3",))):
        save(test, {"rows": [r for r in rows if r["test"] in tests]})
    curves = {f"{r['anchor']}/{r['test']}": r.get("curve") for r in rows if r["test"].startswith("R1")}
    (RES / "t2_curves.json").write_text(json.dumps(curves))


# ------------------------------------------------------------------ T5 response hit

def t5_case(args):
    from protocols import r1, r2
    name, segs = args
    k = replace(ANCHORS[name], segments=segs)
    desc = desc_of(k, name)
    a, b, c = r1(k, "y", desc=desc), r1(k, "z", desc=desc), r2(k, "edge", desc=desc)
    return {"anchor": name, "segments": segs, "table_used": desc.internal.get("table"),
            "target": {"squeeze_y": k.squeeze_stiffness_y, "squeeze_z": k.squeeze_stiffness_z,
                       "lock_y": k.lock_strain_y, "lock_z": k.lock_strain_z, "droop": k.droop_ratio},
            "measured": {"squeeze_y": a["stiffness"], "squeeze_z": b["stiffness"], "lock_y": a["lock_strain"],
                         "lock_z": b["lock_strain"], "droop": c.get("droop")},
            "fail": [a["fail"], b["fail"], c["fail"]], "r2": c, "r1y_capped": a["capped"],
            "r1z_capped": b["capped"]}


def t5():
    jobs = [(n, tuple(s)) for n, segs in T5_SEGMENTS.items() for s in segs]
    rows = run_jobs(t5_case, jobs)
    for r in rows:
        r["error"] = {q: (None if r["measured"][q] is None or ANCHORS[r["anchor"]].rigid
                          else r["measured"][q] / r["target"][q] - 1) for q in ("squeeze_y", "squeeze_z", "droop")}
    save("t5", {"rows": rows})


# ------------------------------------------------------------------ T6 controllability

T6_SQUEEZE = [0.05, 0.158, 0.5, 1.58, 5.0]
T6_DROOP = [0.06, 0.12, 0.2, 0.28, 0.36]


def t6_case(args):
    from protocols import r1, r2
    knob, v = args
    base = ANCHORS["bendable"]
    if knob == "squeeze_y":
        k = replace(base, squeeze_stiffness_y=v)
        desc = desc_of(k, "bendable")
        a, c = r1(k, "y", desc=desc), r2(k, "edge", desc=desc)
    else:
        k = replace(base, droop_ratio=v)
        desc = desc_of(k, "bendable")
        a, c = r1(k, "y", desc=desc), r2(k, "edge", desc=desc)
    return {"knob": knob, "target": v, "squeeze_y": a["stiffness"], "r1y_capped": a["capped"],
            "droop": c.get("droop"), "table_touch": c.get("table_touch"), "fail": [a["fail"], c["fail"]]}


def t6():
    rows = run_jobs(t6_case, [("squeeze_y", v) for v in T6_SQUEEZE] + [("droop", v) for v in T6_DROOP])
    out = {"rows": rows}
    for knob, q in (("squeeze_y", "squeeze_y"), ("droop", "droop")):
        vals = [r[q] for r in rows if r["knob"] == knob]
        out[f"{knob}_monotone"] = all(v is not None for v in vals) and bool(np.all(np.diff(vals) > 0))
    save("t6", out)


# ------------------------------------------------------------------ T7 random scan

def t7_case(args):
    from protocols import r3, rest
    i, kd = args
    k = Kappa.from_dict(kd)
    t0_ = time.time()
    try:
        desc = desc_of(k)
        a = rest(k, T=2.0, desc=desc)
        b = r3(k, desc=desc) if a["fail"] is None else None
        err = None
    except Exception as e:     # recorded as a failure with the kappa; the scan continues
        a, b, err = None, None, f"{type(e).__name__}: {e}"
    return {"i": i, "kappa": kd, "rest": a, "r3": b, "exception": err, "wall_s": time.time() - t0_,
            "n_fixed": sum(j.fixed for j in desc.joints) if err is None else None}


def t7(n=200, seed=1):
    rng = np.random.default_rng(seed)
    kappas = [sample_kappa(rng).to_dict() for _ in range(n)]
    rows = run_jobs(t7_case, list(enumerate(kappas)))
    fails = []
    for r in rows:
        f = r["exception"] or (r["rest"] or {}).get("fail") or (r["r3"] or {}).get("fail")
        if f:
            fails.append({"i": r["i"], "failure": f, "kappa": r["kappa"]})
    save("t7", {"n": n, "seed": seed, "n_fail": len(fails), "fail_rate": len(fails) / n, "failures": fails,
                "rows": rows})


# ------------------------------------------------------------------ T8 mobile mass

def t8_case(mobile):
    from protocols import r3
    k = replace(ANCHORS["compressible"], mobile=mobile, mobile_frac=0.3, mobile_range=0.03, mobile_freq=2.0)
    r = r3(k, desc=desc_of(k, "compressible"))
    r["mobile"] = mobile
    return r


def t8():
    save("t8", {"rows": run_jobs(t8_case, [False, True])})


TESTS = {"t0": t0, "t1": t1, "t2": t2_t4, "t5": t5, "t6": t6, "t7": t7, "t8": t8}


def main():
    global WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="+", help="t0 t1 t2 (=T2-T4) t5 t6 t7 t8 or all")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--t7-n", type=int, default=200)
    args = ap.parse_args()
    WORKERS = args.workers
    names = list(TESTS) if args.tests == ["all"] else args.tests
    for n in names:
        t = time.time()
        TESTS[n](n=args.t7_n) if n == "t7" else TESTS[n]()
        print(f"{n}: {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
