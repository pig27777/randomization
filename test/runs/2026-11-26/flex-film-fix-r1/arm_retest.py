"""Record native pad–film loads in the Policy V3 Yam_B paths."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import mujoco
import numpy as np
from policy_v3.environment import MacroBagGraspEnv, MotionStop
from policy_v3.geometry import rotation_6d
from policy_v3.yam_backend import SafetyLimits, YamBackend
from policy_v3.yam_scene import CORNERS, PICK, FlexScene
from yam_platform.yam_model import build_yam_model, load_config, save_yam_xml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gpu-rl-gates-r1"))
from flex_fixture import TUNING

PROFILES = {
    "first": FlexScene.pad_film_contact,
    "harder": (.004, .95, .999, .0003),
    "softer": (.008, .95, .985, .0005),
}
OFFSETS = {"left": -.018, "center": 0., "right": .018}


def contact_model(scene, model):
    pad_fields = ("solref", "solimp", "friction", "margin", "gap", "priority", "contype", "conaffinity")
    report = {}
    for name in ("left_pad", "right_pad", "table", "table_flex_support", scene.payload_geom):
        idx = model.geom(name).id
        report[name] = {k: getattr(model, f"geom_{k}")[idx].tolist() for k in pad_fields}
    report["film"] = {k: getattr(model, f"flex_{k}")[0].tolist() for k in (*pad_fields, "radius")}
    report["device_pad_film_override"] = dict(solref=[scene.pad_film_contact[0], 1.],
            solimp=[*scene.pad_film_contact[1:], .5, 2.])
    return report


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def attempt(backend, case, out):
    out.mkdir()
    probe = case in OFFSETS
    rows, qpos = [], []
    stages = []
    native_tick, native_follow, native_episode = backend.tick, backend.follow, backend.scene.episode
    a, b = backend.flex_edges
    phase = "reset"
    reported = set()
    effective = None

    class Offset:
        def uniform(self, _low, _high):
            return OFFSETS[case]

    def run_episode(model, initial, rng, scripted):
        return native_episode(model, initial, Offset(), False)

    def monitor(command, effort, block):
        nonlocal effective
        try:
            native_tick(command, effort, block)
        finally:
            data = backend.drive.data
            if np.isfinite(data.qpos).all():
                center, rotation = backend.payload_box_pose()
                bottom = float((CORNERS * backend.scene.half @ rotation.T + center)[:, 2].min())
                stretch = np.linalg.norm(data.flexvert_xpos[a]-data.flexvert_xpos[b], axis=1) / backend.flex_rest_length
                inset, outside = backend.flex_core_counts()
                load, *pads = backend.contact_loads(check=False)
                left, right, depth = backend.drive.pad_stats.numpy().tolist()
                rows.append(dict(t_s=float(data.time), phase=phase, block=block, pad_N=pads, contact_N=load,
                     peak_pad_N=[left, right], min_pad_film_dist_m=depth,
                     arm_peak_N=float(backend.drive.peak_contact.numpy()[0]), inset=inset, outside=outside,
                     bottom_m=bottom, edge_ratio=float(stretch.max())))
                qpos.append(data.qpos.copy())
                if effective is None:
                    for c in data.contact:
                        if ((c.geom[0] in backend.pads and c.flex[1] >= 0) or
                                (c.geom[1] in backend.pads and c.flex[0] >= 0)):
                            effective = dict(solref=c.solref.tolist(), solimp=c.solimp.tolist(),
                                             friction=c.friction.tolist(), inclusion_margin_m=float(c.includemargin))
                            break
                if (left or right) and phase not in reported:
                    reported.add(phase)
                    print(case, phase, "first solved pad forces", left, right, "depth", depth, flush=True)

    def follow(plan, effort, *, record=None, stage="grasp", contact_cutoff_N=None):
        nonlocal phase
        if plan:
            phase = plan[0][2]
            stages.append(phase)
            print(case, phase, float(backend.drive.data.time), flush=True)
        result = native_follow(plan, effort, record=record, stage=stage, contact_cutoff_N=contact_cutoff_N)
        if not probe and phase == "lift":
            if rows[-1]["bottom_m"] - rows[0]["bottom_m"] < .010 and max(rows[-1]["pad_N"]) < .05:
                raise MotionStop("payload_not_captured_at_lift")
            phase = "hold"
            stages.append(phase)
            for _ in range(26):
                backend.tick(backend.last_command, effort, "grasp")
            if (rows[-1]["bottom_m"] - rows[0]["bottom_m"] < .010
                    or min(rows[-1]["pad_N"]) < .05):
                raise MotionStop("payload_not_captured_at_lift")
        return result

    env = MacroBagGraspEnv(backend, np.random.default_rng(7))
    backend.tick, backend.follow = monitor, follow
    if probe:
        backend.scene.episode = run_episode
    start = perf_counter()
    try:
        v0, _, _ = env.reset(seed=7)
        if probe:
            action = np.zeros(13)
            action[:2] = np.clip(((PICK[:2]+[0., -.015])-v0.center[:2]) @ v0.rotation[:2, :2] / v0.half_extent[:2], -1., 1.)
            action[8], action[9], action[12] = -.35, .12, .30
            history, count, v1, meta = env.execute_probe(action)
            np.savez_compressed(out/"history.npz", H_try=history, H_try_raw=env.raw_history,
                                phases=env.trace.stages, timestamps=env.trace.timestamps,
                                tcp_world=env.trace.tcp_world)
        else:
            xyz, gap = (PICK+[0., 0., .0165], .094) if case == "through" else (PICK+[0., -.055, .0125], .050)
            target = np.r_[xyz, rotation_6d(backend.hover_rotation), gap, 8.*backend.motor_Nm_per_N]
            meta = env.execute_grasp(target)
    except (MotionStop, RuntimeError) as exc:
        if not isinstance(exc, MotionStop) and str(exc) != "Reset failed to reach the common neutral pose":
            raise
        meta = dict(physics_validity=False, task_success=False, failure_reason=str(exc), stop_reason=str(exc),
                    invalid_stage="reset", retracted=False, probe_cutoff_trigger_N=None)
        if probe:
            v0, v1, count, action = None, None, 0, np.zeros(13)
    finally:
        backend.tick, backend.follow, backend.scene.episode = native_tick, native_follow, native_episode
    native_peak = float(backend.drive.peak_contact.numpy()[0])
    geometry = dict(inset=max((r["inset"] for r in rows), default=0), outside=max((r["outside"] for r in rows), default=0),
                    max_edge_ratio=max((r["edge_ratio"] for r in rows), default=1.))
    event = backend.drive.penetration_event()
    no_bypass = event is None or not event["direct_force_N"]
    result = dict(case=case, metadata=meta, stages=stages, geometry=geometry, native_peak_N=native_peak,
                 event=event, effective_pad_film_contact=effective, wall_s=perf_counter()-start, rows=rows)
    if probe:
        result.update(offset_m=OFFSETS[case], action=action.tolist(), history_frames=count,
                      native_soft_cutoff=backend.drive.probe_trigger_event,
                      v0=v0.to_dict() if v0 is not None else None, v1=v1.to_dict() if v1 is not None else None,
                      gate_pass=bool(meta["physics_validity"] and meta["retracted"] and v1 is not None
                                and meta["probe_cutoff_trigger_N"] is not None and .5 <= native_peak <= 3.
                                and not geometry["inset"] and not geometry["outside"] and no_bypass))
    else:
        hold = [r for r in rows if r["phase"] == "hold"]
        duration = hold[-1]["t_s"] - hold[0]["t_s"] if hold else 0.
        medians = np.median([r["pad_N"] for r in hold], axis=0).tolist() if hold else None
        clearance = 1000.*min(r["bottom_m"]-rows[0]["bottom_m"] for r in hold) if hold else None
        end = [r["bottom_m"] for r in rows if r["phase"] in ("release", "retreat")]
        drop = 1000.*(end[0]-min(end)) if end else None
        peak_pad = max((max(r["peak_pad_N"]) for r in rows), default=0.)
        result.update(hold_s=duration, medians_N=medians, hold_min_clearance_mm=clearance,
                      release_drop_mm=drop, peak_pad_N=peak_pad,
                      gate_pass=bool(meta["physics_validity"] and meta["task_success"] and duration >= .9998
                                     and clearance >= 40. and all(3. <= f <= 15. for f in medians)
                                     and peak_pad <= 25. and drop is not None and drop >= 5.
                                     and not geometry["inset"] and not geometry["outside"] and no_bypass))
    if qpos:
        np.savez_compressed(out/"trajectory.npz", qpos=qpos, timestamps=[r["t_s"] for r in rows])
    write(out/"summary.json", result)
    print({k:v for k,v in result.items() if k not in ("rows", "v0", "v1", "action")}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="first")
    parser.add_argument("--tuning", choices=("a2", "a3"))
    parser.add_argument("--young-scale", type=float, default=1.)
    parser.add_argument("--modes", nargs="+", choices=(*OFFSETS, "through", "film"))
    args = parser.parse_args()
    args.output.mkdir()

    class RetestScene(FlexScene):
        pad_film_contact = PROFILES[args.profile]
        config = dict(FlexScene.config)
        if args.tuning:
            config["physics_dt_s"] = TUNING[args.tuning][0]

        def add_scene(self, root):
            super().add_scene(root)
            if args.tuning:
                dt, iterations, search, ratio, margin = TUNING[args.tuning]
                root.find("option").attrib.update(timestep=str(dt), iterations=str(iterations),
                                                  ls_iterations=str(search), impratio=str(ratio))
                root.find(".//flexcomp/contact").set("margin", str(margin))
            if args.young_scale != 1.:
                elasticity = root.find(".//flexcomp/elasticity")
                elasticity.set("young", str(float(elasticity.get("young")) * args.young_scale))

    if not args.modes:
        scene = RetestScene()
        _, model = build_yam_model(load_config("nominal",scene.config), add_scene=scene.add_scene, compile_model=scene.compile_model)
        write(args.output/"model-contact.json", contact_model(scene, model))
        return
    with patch("policy_v3.yam_backend.FlexScene", RetestScene):
        backend = YamBackend(7, scene="flex", device="cuda:0", scripted=True,
                             limits=replace(SafetyLimits(), flex_edge_stretch=1.15))
    try:
        write(args.output/"model-contact.json", contact_model(backend.scene, backend.model))
        save_yam_xml(backend.drive.sim.xml, args.output/"scene.xml")
        for case in args.modes:
            attempt(backend, case, args.output/case)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
