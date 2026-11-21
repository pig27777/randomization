"""Paired held-out YAM micro-lift experiment. Run from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "test/low_order_parcel")]
import audit_yam
from experience.yam_platform import protocol
from experience.yam_platform.microlift import MicroLift

TYPES = {"hard": ("rigid_box", .80, 0.), "soft": ("folded_clothes", .62, .35),
         "loose": ("loose_shifting", .35, 0.)}
MASSES = (.16, .32, .68)
RULES = ("continue", "feedback", "retry")
BASE = "d6e519bc"
measure = audit_yam.measure
attachment = audit_yam.attachment
check_state = audit_yam.check_state
dimensions = audit_yam.dimensions
summarize = audit_yam.summarize


class History(list):
    """Keep scalar evaluation channels outside the controller."""
    def append(self, step):
        time, _, result = step
        super().append((time, result.table_clearance.item(), result.support_N.item(),
                        result.state.velocity.norm().item(), *result.state.contacts[0, :, 1].tolist()))


def scene(cfg):
    robot = audit_yam.scene(cfg)
    robot.sim.samples = History()
    return robot


def source_files():
    return {**audit_yam.source_files(), "audit_adapter": Path(audit_yam.__file__),
            "yam_adapter": Path(audit_yam.yam_adapter.__file__),
            "payload_fixture": ROOT / "test/low_order_parcel/test_unified_payload.py"}


def choose(signals):
    return (signals["late_load_N"] < .45 or signals["late_load_N"] < .65 * signals["early_load_N"]
            or abs(signals["gap_change_m"]) > .002)


def trial(job):
    kind, mass, rule, output, smoke = job
    torch.set_num_threads(1)
    preset, fill, indent = TYPES[kind]
    cfg = audit_yam.Config(preset=preset, fill=fill, indent=indent, mass=mass, friction=.40,
                           entry=.033, insertion=.023, fold_entry_contact=True, grasp_face="side",
                           lift_height=.12, lift_speed_m_s=.04, lift_hold_seconds=3.,
                           simulation_seconds=38., smoke_seconds=.04 if smoke else 0.)
    micro = MicroLift(choose if rule == "feedback" else lambda _: rule == "retry")
    current = []
    def record(robot, sample):
        if not current:
            current.append(robot.sim.samples)
    result = protocol.run(sys.modules[__name__], cfg, output, micro_lift=micro, on_sample=record)
    rows = np.asarray(current[0])
    np.savez_compressed(output / "boundary.npz", time_clearance_support_speed_left_right=rows)
    t, clearance, support, velocity, left, right = rows.T
    stable = (clearance > .001) & (support == 0.) & (velocity < .020) & (left > .5) & (right > .5)
    phases = list(result["phases"])
    last_end = phases[-1]["end_s"] if phases else 0.
    if result["sim_time_s"] > last_end:
        phases.append(dict(phase=result["phase"], start_s=last_end, end_s=result["sim_time_s"]))
    last_retry = max((p["end_s"] for p in phases if p["phase"] == "regrasp"), default=0.)
    streak = best = 0
    lost = airborne = False
    for p in phases:
        idx = (t > p["start_s"]) & (t <= p["end_s"])
        if p["phase"] not in ("lift", "lift_hold"):
            airborne = False
            streak = 0
            continue
        for z, normal, quiet in zip(clearance[idx], support[idx], stable[idx], strict=True):
            if z > .001 and normal == 0.:
                airborne = True
            elif airborne and normal > .05:
                lost = True
                airborne = False
            streak = streak + 1 if quiet else 0
            if p["start_s"] >= last_retry:
                best = max(best, streak)
    ok = result["outcome"] == "pickup_release_ok" and best * cfg.dt >= 3.
    decision = micro.decision
    return dict(kind=kind, mass_kg=mass, rule=rule, success=ok, platform=result["outcome"],
                reason=result["reason"], phase=result["phase"], stable_s=best * cfg.dt,
                dropped=bool(lost or result["outcome"] == "drop"), over=result["outcome"] == "over_squeeze",
                retry=bool(decision and decision["retry"]), decision=decision,
                prefix=micro.prefix.hexdigest(), sim_s=result["sim_time_s"], peak_grip_N=result["peaks"]["grip_N"],
                wall_s=result["wall_s"], trial=output.name)


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    tracked = [*source_files().values(), ROOT / "configs/payload_template_X.yaml",
               ROOT / "configs/proxy_material_envelopes.yaml"]
    subprocess.run(["git", "diff", "--exit-code", BASE, "--", *tracked], cwd=ROOT, check=True)
    paths = [*tracked, Path(__file__).resolve(), *protocol.platform_source_files().values(),
             ROOT / "test/policy/PROBE_REPORT.md"]
    manifest = hashes(paths)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(kind, mass, rule, args.output / f"{kind}-{round(mass*1000)}g-{rule}", args.smoke)
            for kind in TYPES for mass in MASSES for rule in RULES]
    if args.smoke:
        jobs = [("soft", .21, "feedback", args.output / "entry", True)]
    meta = dict(frozen_dynamics=BASE, host=platform.node(), python=sys.version, torch=torch.__version__,
                mujoco=mujoco.__version__, source_sha256=manifest,
                jobs=[p.name for _, _, _, p, _ in jobs])
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2))
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as pool:
        results = list(pool.map(trial, jobs))
    if hashes(paths) != manifest:
        raise RuntimeError("Source changed during paired run")
    if not args.smoke:
        for kind in TYPES:
            for mass in MASSES:
                group = [r for r in results if r["kind"] == kind and r["mass_kg"] == mass]
                if len({r["prefix"] for r in group}) != 1:
                    raise RuntimeError(f"Predecision trajectory mismatch for {kind} {mass}")
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
