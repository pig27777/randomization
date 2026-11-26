"""Measure Yam_B probe recovery and contact-only full-arm basket trajectories."""

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from policy_v3.environment import MacroBagGraspEnv, MotionStop
from policy_v3.geometry import exp_rotation, rotation_6d
from policy_v3.yam_backend import YamBackend
from policy_v3.yam_scene import BIN, CORNERS, PICK

OFFSETS = {"left": -.018, "center": 0., "right": .018}


def fixed_probe(v0, jaw_action=.30):
    action = np.zeros(13)
    xy = (PICK[:2] + [0., -.015] - v0.center[:2]) @ v0.rotation[:2, :2]
    action[:2] = np.clip(xy / v0.half_extent[:2], -1., 1.)
    action[8], action[9], action[12] = -.35, .12, jaw_action
    return action


def grasp_candidate(backend, mode, *, dz=0., dy=0., dx=0., effort_N=8., yaw_deg=0., opening_m=None, tilt_rad=(0., 0.)):
    xyz, gap = (PICK + [0., 0., .0165], .094) if mode == "through" else (
        PICK + [0., -.055, .0125], .050)
    rotation = exp_rotation(np.array([*tilt_rad, np.deg2rad(yaw_deg)])) @ backend.hover_rotation
    return np.r_[xyz + [dx, dy, dz], rotation_6d(rotation), gap if opening_m is None else opening_m,
                 effort_N * backend.motor_Nm_per_N]


def trial(backend, mode, offset, seed, output, *, dz=0., dy=0., dx=0., effort_N=8., after_probe=False,
          yaw_deg=0., opening_m=None, jaw_action=.30, probe_speed=.16, probe_cutoff_N=None,
          guard_full_probe_search=None, tilt_rad=(0., 0.), screen_lift=False, cruise_scale=.65,
          compact_grasp=False, approach_scale=.65):
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    env = MacroBagGraspEnv(backend, rng)
    drive = backend.drive
    previous_speed = backend.probe_contact_speed
    previous_cutoff = backend.probe_cutoff_N
    previous_search = backend.guard_full_probe_search
    previous_cruise = backend.grasp_cruise_scale
    previous_approach = backend.grasp_approach_scale
    previous_compact = backend.compact_grasp
    backend.grasp_cruise_scale = cruise_scale
    backend.grasp_approach_scale = approach_scale
    backend.compact_grasp = compact_grasp
    backend.probe_contact_speed = probe_speed
    if probe_cutoff_N is not None:
        backend.probe_cutoff_N = probe_cutoff_N
    if guard_full_probe_search is not None:
        backend.guard_full_probe_search = guard_full_probe_search
    native_tick, native_follow, native_episode = backend.tick, backend.follow, backend.scene.episode
    native_grasp = backend.execute_grasp
    rows, qpos = [], []
    recorded_stages = set()
    vision, probe = None, None
    start = perf_counter()

    class FixedOffset:
        def uniform(self, *_bounds):
            return offset

    def episode(model, initial, _rng, _scripted):
        return native_episode(model, initial, FixedOffset(), False)

    def tick(command, effort, block):
        try:
            native_tick(command, effort, block)
        finally:
            center, rotation = backend.payload_box_pose()
            corners = CORNERS * backend.scene.half @ rotation.T + center
            if np.isfinite(drive.data.qpos).all():
                inset, exposed = backend.frame_core_counts
                static_dist = {name: float(distance) for name, distance in zip(
                    drive.static_film_surfaces, drive.frame_static_film_min_m, strict=True) if np.isfinite(distance)}
                rows.append(dict(t=float(drive.data.time), stage=backend.motion_label, block=block,
                                 load_N=drive.frame_load.tolist(), peak_N=drive.frame_peak.tolist(),
                                 static_film_min_m=static_dist,
                                 max_edge_ratio=float(drive.frame_edge_peak), inset=inset, exposed=exposed,
                                 tcp_minus_payload_com_m=(
                                     drive.data.site_xpos[backend.site] - drive.data.xipos[backend.payload_id]).tolist(),
                                 jaw_actual_m=float(drive.data.qpos[backend.qadr[6:]].sum()),
                                 payload_min_z_m=float(corners[:, 2].min()),
                                 parcel_min_z_m=float(drive.data.flexvert_xpos[:, 2].min()),
                                 jaw_m=float(drive.observation[24]), jaw_target_m=float(drive.observation[27])))
                qpos.append(drive.data.qpos.copy())

    def follow(plan, effort, *, record=None, stage="grasp", contact_cutoff_N=None):
        for _, _, label in plan:
            if label not in recorded_stages:
                recorded_stages.add(label)
                print(f"{output.name} {label} {drive.data.time:.3f}s", flush=True)
        return native_follow(plan, effort, record=record, stage=stage, contact_cutoff_N=contact_cutoff_N)

    backend.scene.episode, backend.tick, backend.follow = episode, tick, follow
    if screen_lift:
        backend.execute_grasp = partial(native_grasp, screen_lift=True)
    try:
        v0, _, _ = env.reset(seed=seed)
        if mode == "probe" or after_probe:
            history, count, v1, probe = env.execute_probe(fixed_probe(v0, jaw_action))
            np.savez_compressed(output / "history.npz", H_try=history, H_try_raw=env.raw_history,
                                phases=env.trace.stages, timestamps=env.trace.timestamps,
                                tcp_world=env.trace.tcp_world)
            vision = dict(v0=v0.to_dict(), v1=v1.to_dict() if v1 is not None else None, history_frames=count,
                          history_dim=history.shape[1], trigger=drive.probe_trigger_event)
        if mode != "probe" and (probe is None or probe["retracted"]):
            target = grasp_candidate(backend, mode, dz=dz, dy=dy, dx=dx, effort_N=effort_N,
                                     yaw_deg=yaw_deg, opening_m=opening_m, tilt_rad=tilt_rad)
            result = env.execute_grasp(target)
        else:
            result = probe
    except MotionStop as error:
        result = dict(task_success=False, physics_validity=error.physics_valid, invalid_stage=getattr(backend, "motion_label", "reset"),
                      failure_reason=str(error))
    finally:
        backend.scene.episode, backend.tick, backend.follow = native_episode, native_tick, native_follow
        backend.execute_grasp = native_grasp
        backend.probe_contact_speed = previous_speed
        backend.probe_cutoff_N = previous_cutoff
        backend.guard_full_probe_search = previous_search
        backend.grasp_cruise_scale = previous_cruise
        backend.grasp_approach_scale = previous_approach
        backend.compact_grasp = previous_compact
    inset = max((r["inset"] for r in rows), default=0)
    exposed = max((r["exposed"] for r in rows), default=0)
    direct = max((r["peak_N"][4] for r in rows), default=0.)
    arm_peak = max((r["peak_N"][0] for r in rows if r["block"] == "probe"), default=0.)
    pad_peak = max((max(r["peak_N"][1:3]) for r in rows if r["block"] == "grasp"), default=0.)
    hold = [r for r in rows if r["stage"] == "close"]
    hold = hold[-min(8, len(hold)):]
    medians = np.median([r["load_N"][1:3] for r in hold], axis=0).tolist() if hold else None
    rim = [r for r in rows if r["stage"] == "transport"]
    static_min = {}
    for row in rows:
        for name, distance in row["static_film_min_m"].items():
            static_min[name] = min(distance, static_min.get(name, distance))
    clean = (inset, exposed, direct) == (0, 0, 0.)
    if mode == "probe":
        trigger = (vision["trigger"] or {}).get("load_N", 0.) if vision else 0.
        cutoff = previous_cutoff if probe_cutoff_N is None else probe_cutoff_N
        gate = clean and result.get("physics_validity") and result.get("retracted") and vision and vision["v1"] and (
            cutoff - 1e-6 <= trigger <= 2.8 and arm_peak < 3.5 and vision["history_dim"] == 48 and vision["history_frames"] > 0)
    else:
        gate = clean and result.get("physics_validity") and result.get("task_success") and pad_peak <= 25. and (
            medians is not None and all(3. <= median <= 15. for median in medians))
    metadata = dict(mode=mode, offset_m=offset, candidate_delta_m=[dx, dy, dz], effort_N=effort_N, seed=seed,
                    yaw_deg=yaw_deg, tilt_rad=tilt_rad, opening_m=opening_m, jaw_action=jaw_action, probe_speed=probe_speed,
                    screen_lift=screen_lift, cruise_scale=cruise_scale, compact_grasp=compact_grasp,
                    approach_scale=approach_scale,
                    probe_cutoff_N=previous_cutoff if probe_cutoff_N is None else probe_cutoff_N,
                    guard_full_probe_search=previous_search if guard_full_probe_search is None else guard_full_probe_search,
                    solver_profile=backend.scene.solver_profile, wall_s=perf_counter() - start, outcome=result, probe=probe, vision=vision, gate_pass=bool(gate),
                    max_inset=inset, max_exposed=exposed, robot_payload_direct_force_N=direct, arm_probe_peak_N=arm_peak,
                    pad_grasp_peak_N=pad_peak, hold_medians_N=medians,
                    min_static_film_separation_m=static_min,
                    min_transport_clearance_m=min((min(r["parcel_min_z_m"], r["payload_min_z_m"]) - BIN["rim_height_m"] for r in rim),
                                                  default=None),
                    max_edge_ratio=max((r["max_edge_ratio"] for r in rows), default=0.),
                    max_hold_edge_ratio=max((r["max_edge_ratio"] for r in hold), default=None),
                    grasp_clamp_counts=drive.clamp_counts.tolist(), event=drive.penetration_event())
    metadata["stage_frames"] = {s: sum(r["stage"] == s for r in rows)
                                for s in dict.fromkeys(r["stage"] for r in rows)}
    metadata["sim_elapsed_s"] = rows[-1]["t"] if rows else 0.
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / "forces.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
    if qpos:
        np.savez_compressed(output / "trajectory.npz", qpos=qpos, timestamps=[r["t"] for r in rows])
    print(output.name, json.dumps({k: v for k, v in metadata.items() if k != "vision"}), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", choices=(*OFFSETS, "through", "film"))
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--offset", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dz", type=float, default=0.)
    parser.add_argument("--dy", type=float, default=0.)
    parser.add_argument("--dx", type=float, default=0.)
    parser.add_argument("--effort-n", type=float, default=8.)
    parser.add_argument("--after-probe", action="store_true")
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--solver-profile", choices=("nominal", "a2"), default="nominal")
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--ls-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--solver", choices=("Newton", "CG"))
    parser.add_argument("--contacts", type=int)
    parser.add_argument("--constraints", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    from yam_platform.yam_model import save_yam_xml

    if bool(args.case_file) == bool(args.cases):
        parser.error("supply exactly one of --case-file or --cases")
    if args.case_file:
        options = json.loads(args.case_file.read_text())
    else:
        options = [dict(name=case, mode="probe" if case in OFFSETS else case,
                        offset=OFFSETS.get(case, args.offset), seed=args.seed, dz=args.dz, dy=args.dy,
                        dx=args.dx, effort_N=args.effort_n, after_probe=args.after_probe) for case in args.cases]
    jobs = [partial(trial, output=args.output / row["name"],
                    **{k: v for k, v in row.items() if k != "name"}) for row in options]
    physics = dict(flex_solver=args.solver_profile, contact_capacity=args.contacts,
                   constraint_capacity=args.constraints,
                   solver_options={key: getattr(args, key) for key in ("timestep", "iterations", "ls_iterations",
                                                                      "tolerance", "solver")
                                   if getattr(args, key) is not None})
    if args.batched:
        from policy_v3.yam_batch import BatchedYam

        batch = BatchedYam(args.seed, len(jobs), **physics)
        try:
            save_yam_xml(batch.drive.sim.xml, args.output / "scene.xml")
            def progress(state):
                print(f"{state['control_frames']} control frames, {state['env_steps']} environment steps, "
                      f"{state['completed_episodes']}/{len(jobs)} complete, {state['wall_s']:.1f}s", flush=True)

            _, timing = batch.run(jobs, progress=progress)
            (args.output / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        finally:
            batch.close()
    else:
        backend = YamBackend(args.seed, scene="flex", device="cuda:0", **physics)
        try:
            save_yam_xml(backend.drive.sim.xml, args.output / "scene.xml")
            for job in jobs:
                job(backend)
        finally:
            backend.close()


if __name__ == "__main__":
    main()
