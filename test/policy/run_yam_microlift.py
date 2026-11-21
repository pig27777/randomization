"""Paired held-out YAM micro-lift experiment. Run from the repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from functools import partial
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
import probe_rgb
from src.low_order_parcel import lift_probe
from experience.yam_platform import protocol
from experience.yam_platform.microlift import MicroLift

TYPES = {"hard": ("rigid_box", .80, 0.), "soft": ("folded_clothes", .62, .35),
         "loose": ("loose_shifting", .35, 0.)}
MASSES = (.16, .32, .68)
RULES = ("continue", "feedback", "retry")
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
                        result.state.velocity.norm().item(), *result.state.contacts[0, :, 1].tolist(),
                        *result.tension[0].tolist(), result.state.torsion.torque.norm().item() if result.state.torsion else 0.))


def scene(cfg):
    robot = audit_yam.scene(cfg)
    robot.sim.samples = History()
    return robot


def source_files():
    return {**audit_yam.source_files(), "probe_rgb": Path(probe_rgb.__file__), "lift_probe": Path(lift_probe.__file__), "audit_adapter": Path(audit_yam.__file__),
            "yam_adapter": Path(audit_yam.yam_adapter.__file__),
            "payload_fixture": ROOT / "test/low_order_parcel/test_unified_payload.py"}


def choose(signals):
    return (signals["late_load_N"] < .45 or signals["late_load_N"] < .65 * signals["early_load_N"]
            or abs(signals["gap_change_m"]) > .002)


def trial(job, *, complete_motion=False):
    kind, mass, rule, output, smoke, fix4, scaled = job
    torch.set_num_threads(1)
    preset, fill, indent = ("rigid_box", 1., 0.) if kind == "taut" else TYPES[kind]
    cfg = audit_yam.Config(complete_motion=complete_motion, pad_radius_m=.012 if fix4 else 0., preset=preset, fill=fill, indent=indent, mass=mass, friction=.40,
                           entry=.033, insertion=.023, fold_entry_contact=True, grasp_face="side",
                           lift_height=.12, lift_speed_m_s=.04, lift_hold_seconds=3.,
                           simulation_seconds=38., smoke_seconds=.04 if smoke else 0.)
    # All nine trials use a nominal, known 100 mm exterior (no fill/material lookup).
    height = lift_probe.probe_height(torch.tensor(.10, dtype=torch.float64)).item() if scaled else .025
    micro = MicroLift(choose if rule == "feedback" else lambda _: rule == "retry", height=height)
    images = {}
    decision_data = {}
    current = []
    def record(robot, sample):
        if not current:
            current.append(robot.sim.samples)
        if fix4 and sample["phase"] == "close" and micro.decision is None:
            images["start"] = probe_rgb.exposure(robot)
        if fix4 and micro.decision and not decision_data:
            sim = robot.sim
            image, mask, ee = probe_rgb.exposure(robot)
            images["end"] = image, mask, ee
            _, initial_mask, initial_ee = images["start"]
            valid = mask is not None and initial_mask is not None and abs(ee - initial_ee) >= 1.
            decision_data.update(t=sample["t"], tension_N=sample["proxy_tension_N"],
                                 support_N=sample["support_N"], clearance_m=sample["proxy_clearance_m"],
                                 pad_moment_Nm=sim.state.torsion.torque.norm().item() if sim.state.torsion else 0.,
                                 gamma_rgb=float(np.clip((mask-initial_mask)/(ee-initial_ee),0.,1.5)) if valid else None,
                                 mask_dy_px=mask-initial_mask if valid else None, fk_dy_px=ee-initial_ee)
    result = protocol.run(sys.modules[__name__], cfg, output, micro_lift=micro, on_sample=record)
    for name, (image, _, _) in images.items():
        image.save(output / f"rgb-{name}.png")
    tape = np.asarray(current[0])
    rows = tape[:, :6]
    if decision_data:
        idx = np.searchsorted(rows[:, 0], micro.decision["time_s"])
        assert abs(rows[idx, 0] - micro.decision["time_s"]) < 1e-8
        decision_data.update(boundary_time_s=rows[idx, 0], exact_support_N=rows[idx, 2],
                             exact_clearance_m=rows[idx, 1], exact_tension_N=tape[idx, 6:8].tolist(),
                             exact_pad_moment_Nm=tape[idx, 8], peak_pad_moment_Nm=tape[:, 8].max())
    (output / "probe.json").write_text(json.dumps(decision_data, indent=2))
    np.savez_compressed(output / "loads.npz", tension_N=tape[:, 6:8], pad_moment_Nm=tape[:, 8])
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
                prefix=micro.prefix.hexdigest(), probe=decision_data, sim_s=result["sim_time_s"], peak_grip_N=result["peaks"]["grip_N"],
                wall_s=result["wall_s"], trial=output.name)


def hashes(paths):
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fix4", action="store_true", help="Enable finite 12 mm pad moments")
    parser.add_argument("--size-probe", action="store_true", help="Scale the probe to the observed outer height")
    parser.add_argument("--projection-controls", action="store_true",
                        help="Run the eight independent fixed-action color-camera controls with Fix 4 and sized probes")
    args = parser.parse_args()
    if args.projection_controls:
        args.fix4 = args.size_probe = True
    tracked = [*source_files().values(), ROOT / "configs/payload_template_X.yaml",
               ROOT / "configs/proxy_material_envelopes.yaml"]
    physics = [ROOT / "src/low_order_parcel" / name for name in (
        "pad_torsion.py","macro_parcel_sim.py","unified_payload.py","lumped_contact_proxy.py","yam_adapter.py")]
    fixtures = [ROOT / "test/low_order_parcel/test_unified_payload.py", *tracked[-2:]]
    subprocess.run(["git","diff","--exit-code","046384ab","--",*physics,*fixtures], cwd=ROOT, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # The exact loaded physics sources are snapshotted per trial. Report candidate
    # patches against HEAD as such when running alongside another experiment.
    delta = subprocess.check_output(["git", "diff", "HEAD", "--", *tracked], cwd=ROOT)
    paths = [*tracked, Path(__file__).resolve(), *protocol.platform_source_files().values(),
             ROOT / "test/policy/PROBE_REPORT.md"]
    manifest = hashes(paths)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(kind, mass, rule, args.output / f"{kind}-{round(mass*1000)}g-{rule}", args.smoke, args.fix4, args.size_probe)
            for kind in TYPES for mass in MASSES for rule in RULES]
    if args.projection_controls:
        jobs = [(kind,mass,"continue",args.output / f"{kind}-{mass:.2f}",False,True,True)
                for mass in (.18,.52) for kind in (*TYPES,"taut")]
    if args.smoke:
        jobs = [("soft", .21, "feedback", args.output / "entry", True, args.fix4, args.size_probe)]
    meta = dict(base_revision=revision, tracked_physics_patch_sha256=sha256(delta).hexdigest(), fix4=args.fix4, size_probe=args.size_probe, host=platform.node(), python=sys.version, torch=torch.__version__,
                mujoco=mujoco.__version__, source_sha256=manifest,
                jobs=[job[3].name for job in jobs])
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2))
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as pool:
        results = list(pool.map(partial(trial, complete_motion=args.projection_controls), jobs))
    if hashes(paths) != manifest:
        raise RuntimeError("Source changed during paired run")
    if not args.smoke and not args.projection_controls:
        for kind in TYPES:
            for mass in MASSES:
                group = [r for r in results if r["kind"] == kind and r["mass_kg"] == mass]
                if len({r["prefix"] for r in group}) != 1:
                    raise RuntimeError(f"Predecision trajectory mismatch for {kind} {mass}")
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
