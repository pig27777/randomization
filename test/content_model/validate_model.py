"""M1/M2 CPU model validation. Run under the project GPU lock even on CPU.

flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python validate_model.py mechanics
flock /tmp/claude-gpu.lock /home/ubuntu/Dev/Envs/mjwarp/bin/python validate_model.py integrity --anchor compressible
Outputs are separate from round2 and the calibration table is read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import unittest
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))

import mj_adapter as A
import protocols as P
from model import ANCHORS, G, PAD, build, load_table

ROOT = CONTENT
OUT = ROOT / "results" / "modeling"
RUN_KIND = "formal"
SOURCES = ("bushing.py", "mj_adapter.py", "model.py", "protocols.py", "../../test/content_model/validate_model.py", "metrics.py",
           "../../test/content_model/stability_estimate.py", "../../test/content_model/tests/test_bushing.py", "results/table.json")
SOURCE_HASHES = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def save(name, result):
    OUT.mkdir(parents=True, exist_ok=True)
    result["run_kind"] = RUN_KIND
    result["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["source_sha256"] = SOURCE_HASHES
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2) + "\n")
    details = ("rows", "cycles", "source_sha256", "stiffness", "gradients", "stops", "damping", "segment_positions", "segment_quaternions",
               "joint_displacements", "joint_rotations")
    print(f"{path}: {clean({k: v for k, v in result.items() if k not in details})}", flush=True)


def mechanics():
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "tests"), pattern="test_bushing.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checks = sys.modules["test_bushing"].BushingTest
    save("m1", {"tests": result.testsRun, "passed": result.wasSuccessful(),
                 "stiffness": checks.measurements, "gradients": checks.gradients,
                 "stops": getattr(checks, "stop_errors", []),
                 "damping": getattr(checks, "damping", []),
                 "releases": checks.releases,
                 "failures": [(str(t), s) for t, s in result.failures + result.errors]})
    return result.wasSuccessful()


def layer_record(sim):
    pos = sim.seg_pos()
    rot = sim.d.xmat[sim.seg_bodies].reshape(-1, 3, 3)
    half = np.array([s.half for s in sim.desc.segments])
    # z extent of a rotated box, including local corner rotation.
    bottom = pos[:, 2] - (np.abs(rot[:, 2, :]) * half).sum(1)
    layers = np.array([s.idx[2] for s in sim.desc.segments])
    return {"layer_centres": [pos[layers == z].mean(0) for z in range(sim.desc.kappa.segments[2])],
            "bottom_z": float(bottom.min()), "max_strain": sim.max_strain(),
            "welds_active": int(sim.d.eq_active[sim.welds].sum()),
            "confinement": sim.external_compaction_load,
            "confinement_raw": sim.external_compaction_load_raw}


def full_height_pads(xml, height):
    root = ET.fromstring(xml)
    hz = height / 2 - A.PAD_T
    for name in ("pA", "pB"):
        body = root.find(f".//body[@name='{name}']")
        core = body.find(f"geom[@name='{name}_core']")
        size = np.fromstring(core.get("size"), sep=" ")
        size[2] = hz
        core.set("size", A._v(size))
        for sign in (1, -1):
            for axis in (0, 2):
                geom = body.find(f"geom[@name='{name}_e{axis}{sign}']")
                ends = np.fromstring(geom.get("fromto"), sep=" ").reshape(2, 3)
                ends[:, 2] = sign * hz if axis == 0 else [-hz, hz]
                geom.set("fromto", A._v(ends.ravel()))
    return ET.tostring(root, encoding="unicode")


def integrity(anchor, mode, variant, grip, hold_force=True):
    A.JOINT_MODE = mode
    k = ANCHORS[anchor]
    desc = build(k, table=load_table(), geometry=anchor, omega_max=A.OMEGA_MAX)
    lx, ly, lz = k.size
    if grip == "center":
        pads = [("pA", 1, (0, -(ly / 2 + P.GAP + A.PAD_T), lz / 2)),
                ("pB", 1, (0, ly / 2 + P.GAP + A.PAD_T, lz / 2))]
        direction = {"pA": np.array([0, 1., 0]), "pB": np.array([0, -1., 0])}
        travel, normal, xmin = .45 * ly, 1, -.6
    else:
        x0 = -lx / 2 + PAD / 2
        pads = [("pA", 2, (x0, 0, lz + P.GAP + A.PAD_T)),
                ("pB", 2, (x0, 0, -P.GAP - A.PAD_T))]
        direction = {"pA": np.array([0, 0, -1.]), "pB": np.array([0, 0, 1.])}
        travel, normal, xmin = .45 * lz, 2, -lx / 2 + .05
    built = A.scene(desc, (0, 0, lz / 2 + 1e-4), pads, table_xmin=xmin)
    if variant == "frictionless_table":
        dt = A.BUSHING_DT if mode == "bushing" else A.DT
        pairs = "\n".join(A.pad_pair("table", f"s{i}", s.mass, dt=dt, friction=0)
                          for i, s in enumerate(desc.segments))
        built.xml = built.xml.replace("<contact>", "<contact>" + pairs)
    if variant == "full_height":
        if grip != "center":
            raise ValueError("full_height is a centre grip diagnostic")
        built.xml = full_height_pads(built.xml, lz)
    sim = A.Sim(built, ["pA", "pB"])
    height = max(.1, lx + .05)
    phases = [("lift", height / .05, lambda t: np.array([0, 0, min(.05 * t, height)])),
              ("hold", 2., lambda t: np.array([0, 0, height]))]
    sim, runner, rows, events, closed = P._grip_run_tracked(
        k, pads, direction, travel, .025, phases, xmin, None, desc, normal,
        track_from="lift", sim=sim, record=layer_record, hold_force=hold_force)
    end = {"fail": runner.fail, "fail_t": runner.fail_t, "closed": closed, "events": events,
           "anchor": anchor, "mode": mode, "variant": variant, "grip": grip,
           "dt": sim.m.opt.timestep, "rows": rows, "hold_force": hold_force, "pass": False}
    end["segment_positions"] = sim.seg_pos()
    end["segment_quaternions"] = sim.d.xquat[sim.seg_bodies].copy()
    end["grip_references"] = getattr(sim, "grip_reference", None)
    if sim.bushings is not None:
        *_, u, phi = sim.bushings.coordinates(sim.d)
        end["joint_displacements"] = u
        end["joint_rotations"] = phi
    if "hold" in events:
        hold = [r for r in rows if r["t"] >= events["hold"]]
        last = [r for r in hold if r["t"] >= events["hold"] + 1.5]
        if last:
            support = np.mean([r["pad_fz"] for r in last]) / (sim.mass * G)
            table = max(r["table"] for r in hold) / (sim.mass * G)
            slip = max(r["slip"] for r in hold) * 1e3 if all(r["slip"] is not None for r in hold) else None
            end.update({"pad_support_ratio": support, "table_force_ratio_max": table,
                        "slip_mm": slip, "relative_angle_deg_max": max(r["relang"] for r in rows),
                        "bottom_z_min_hold": min(r["bottom_z"] for r in hold),
                        "layer_lift_mm": (np.array(last[-1]["layer_centres"])[:, 2]
                                          - np.array(rows[0]["layer_centres"])[:, 2]) * 1e3,
                        "interior_motion_mm": max(r["interior_motion"] for r in hold) * 1e3,
                        "grip_min_force_mean": np.mean([r["F_min"] for r in last]),
                        "pass": runner.fail is None and abs(support - 1) <= .05 and table <= .01
                                and slip is not None and slip <= 5})
    save(f"m2_{mode}_{anchor}_{grip}_{variant}", end)
    return end["pass"]


def rest_energy(anchor):
    A.JOINT_MODE = "bushing"
    k = ANCHORS[anchor]
    desc = build(k, table=load_table(), geometry=anchor, omega_max=A.OMEGA_MAX)
    rest, energy = P.rest(k, desc=desc), P.energy(k, desc=desc)
    ok = (rest["fail"] is None and abs(rest.get("support_ratio", 0) - 1) <= .02
          and energy["pass"])
    save(f"rest_energy_{anchor}", {"rest": rest, "energy": energy, "pass": ok})
    return ok


def release_cycles(anchor):
    """Two real grip/lift/place/open cycles in the same state; never reset the object or its velocity."""
    k = ANCHORS[anchor]
    desc = build(k, table=load_table(), geometry=anchor, omega_max=A.OMEGA_MAX)
    lx, ly, lz = k.size
    pads = [("pA", 1, (0, -(ly / 2 + P.GAP + A.PAD_T), lz / 2)),
            ("pB", 1, (0, ly / 2 + P.GAP + A.PAD_T, lz / 2))]
    direction = {"pA": np.array([0, 1., 0]), "pB": np.array([0, -1., 0])}
    sim = A.Sim(A.scene(desc, (0, 0, lz / 2 + 1e-4), pads), [p[0] for p in pads])
    height = max(.1, lx + .05)
    phases = [("lift", height / .05, lambda t: np.array([0, 0, min(.05 * t, height)])),
              ("hold", 2., lambda t: np.array([0, 0, height])),
              ("place", height / .05, lambda t: np.array([0, 0, max(height - .05 * t, 0)])),
              ("land", .3, lambda t: np.zeros(3))]
    cycles = []
    for cycle in range(2):
        cycle_pads = [(p, normal, sim.pad_pos(p).copy()) for p, normal, _ in pads]
        sim, run, rows, events, closed = P._grip_run_tracked(
            k, cycle_pads, direction, .45 * ly, .025, phases, -.6, None, desc, 1,
            track_from="lift", sim=sim, record=layer_record)
        hold = [r for r in rows if events.get("hold", np.inf) <= r["t"] < events.get("place", np.inf)]
        last = hold[-max(1, round(.5 / P.REC)):]
        support = np.mean([r["pad_fz"] for r in last]) / (sim.mass * G) if last else None
        slip = max(r["slip"] for r in hold) * 1e3 if hold and all(r["slip"] is not None for r in hold) else None
        table = max(r["table"] for r in hold) / (sim.mass * G) if hold else None
        start = {p: sim.pad_pos(p).copy() for p in sim.pad_names}
        opened = {p: np.array(pos) - direction[p] * (.25 * ly + P.GAP) for p, _, pos in pads}
        duration = max(np.linalg.norm(opened[p] - start[p]) for p in start) / .025
        opening_rows = []
        begin = run.t
        while run.fail is None and run.t - begin < duration + 1.:
            f = min(1., (run.t - begin + sim.m.opt.timestep) / duration)
            if run.step({p: start[p] + f * (opened[p] - start[p]) for p in start}):
                pf, tf, _ = sim.contact_forces()
                opening_rows.append(dict(t=run.t - begin, welds=int(sim.d.eq_active[sim.welds].sum()),
                                         confinement=sim.external_compaction_load,
                                         force=max(sim.pad_normal_force(p, 1) for p in start),
                                         table_support=float(tf[2] / (sim.mass * G))))
        released = opening_rows[-max(1, round(.2 / P.REC)):]
        release_ok = (bool(released) and all(r["welds"] == 0 and r["force"] < P.F_TOUCH for r in released)
                      and abs(np.mean([r["table_support"] for r in released]) - 1) <= .05)
        ok = (run.fail is None and support is not None and abs(support - 1) <= .05
              and table <= .01 and slip is not None and slip <= 5 and release_ok)
        cycles.append(dict(cycle=cycle + 1, closed=closed, fail=run.fail, support=support, slip_mm=slip,
                           table_ratio=table, released=release_ok, opening=opening_rows,
                           pass_cycle=ok, rows=rows))
        if not ok:
            break
    ok = len(cycles) == 2 and all(r["pass_cycle"] for r in cycles)
    save(f"release_cycles_{anchor}", dict(anchor=anchor, dt=sim.m.opt.timestep, cycles=cycles, passed=ok))
    return ok


def regressions():
    """Shared grip-entry regressions: hard R2 centre/edge, hard and compressible R3."""
    ok = True
    for name, test, anchor, grip in (("r2_hard_center", P.r2, "hard", "center"), ("r2_hard_edge", P.r2, "hard", "edge"),
                                     ("r3_hard_center", P.r3, "hard", None),
                                     ("r3_compressible_center", P.r3, "compressible", None)):
        k = ANCHORS[anchor]
        desc = build(k, table=load_table(), geometry=anchor, omega_max=A.OMEGA_MAX)
        r = test(k, grip, desc=desc) if grip else test(k, desc=desc)
        r["pass"] = (r["fail"] is None and not r["table_touch"] and abs(r["pad_fz_ratio"] - 1) <= .05
                     and r["slip_mm"] is not None and r["slip_mm"] < 5)
        save(name, r)
        ok &= r["pass"]
    return ok


def run_case(case):
    if case[0] == "integrity":
        return integrity(*case[1:])
    return rest_energy(case[1])


def suite(hold_force=True):
    if not mechanics():
        return False
    cases = [("integrity", "compressible", "truss", v, "center", hold_force)
             for v in ("original", "frictionless_table", "full_height")]
    cases += [("integrity", n, "bushing", "original", g, hold_force)
              for n in ("bendable", "compressible") for g in ("center", "edge")]
    cases += [("rest_energy", n) for n in ("bendable", "compressible")]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_case, cases))
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent / "tests"), pattern="test_sampling.py")
    sampling = unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful()
    # Legacy diagnostics are expected to reveal failures, not gate the new model.
    ok = all(results[3:]) and sampling
    save("suite", {"cases": [{"config": c, "pass": r} for c, r in zip(cases, results)],
                   "sampling_pass": sampling, "pass": ok})
    return ok


def main():
    global OUT, RUN_KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=("mechanics", "integrity", "rest_energy", "suite", "regressions", "release_cycles"))
    ap.add_argument("--anchor", choices=("bendable", "compressible"), default="compressible")
    ap.add_argument("--mode", choices=("bushing", "truss"), default="bushing")
    ap.add_argument("--grip", choices=("center", "edge"), default="center")
    ap.add_argument("--variant", choices=("original", "frictionless_table", "full_height"), default="original")
    ap.add_argument("--exploratory", action="store_true", help="separate development output; not formal evidence")
    ap.add_argument("--run-id", help="named configuration directory for a distinct revision")
    ap.add_argument("--freeze-gap", action="store_true", help="historical grip controller, no force maintenance")
    ap.add_argument("--dt", type=float, help="bushing time step override (step consistency checks)")
    args = ap.parse_args()
    if args.dt:
        A.BUSHING_DT = args.dt
    if args.exploratory:
        OUT = OUT / "exploration"
        RUN_KIND = "exploratory"
    if args.run_id:
        if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
            ap.error("--run-id must be a directory name, not a path")
        OUT = OUT / args.run_id
    start = time.time()
    if args.test == "suite":
        ok = suite(hold_force=not args.freeze_gap)
    elif args.test == "mechanics":
        ok = mechanics()
    elif args.test == "regressions":
        ok = regressions()
    elif args.test == "rest_energy":
        ok = rest_energy(args.anchor)
    elif args.test == "release_cycles":
        ok = release_cycles(args.anchor)
    else:
        ok = integrity(args.anchor, args.mode, args.variant, args.grip, hold_force=not args.freeze_gap)
    print(f"{args.test}: pass={ok}, wall={time.time() - start:.1f}s", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
