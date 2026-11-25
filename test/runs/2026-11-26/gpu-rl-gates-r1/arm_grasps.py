"""Full GPU Yam_B Flex attempts using collision-checked Policy V3 trajectories."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from bag_sim.wrap import triangle_box_intersections, winding_number
from flex_fixture import TUNING
from policy_v3.environment import MacroBagGraspEnv, MotionStop
from policy_v3.geometry import rotation_6d
from policy_v3.workers import write_json
from policy_v3.yam_backend import SafetyLimits, YamBackend
from policy_v3.yam_scene import CORNERS, PICK, FlexScene


def run(b, episode, output, seed=7):
    env = MacroBagGraspEnv(b, np.random.default_rng(seed))
    env.reset(seed=seed)
    if episode == "through":
        point = PICK + [0., 0., .0165]
        r = b.hover_rotation
        gap = .094
    else:
        point = PICK + [0., -.055, .0125]
        r = b.hover_rotation
        gap = .050
    grasp = np.r_[point, rotation_6d(r), gap, 8. * b.motor_Nm_per_N]
    last_tick, follow_native = b.tick, b.follow
    rows, states = [], []
    motion = "initial"
    initial_z = float(b.drive.data.geom_xpos[b.payload_geom, 2])
    initial_floor = initial_z - b.scene.half[2]
    data = b.drive.data
    center, rotation = b.payload_box_pose()
    film = data.flexvert_xpos
    faces = b.scene.parcel.envelope["faces"]
    h = b.scene.half
    corners = CORNERS * h @ rotation.T + center
    outside = int((np.abs(winding_number(film, faces, corners)) < .5).sum())
    inset = int(triangle_box_intersections(film, faces, h - .002, center, rotation).sum())
    if outside or inset:
        write_json(output / f"{episode}.json", dict(arm="Yam_B", mode=episode, gate1_pass=False,
                   result=dict(physics_validity=False, task_success=False, invalid_stage="reset",
                               failure_reason="initial_film_core_geometry"),
                   initial_geometry=dict(outside_corners=outside, inset_triangles=inset,
                                         core_bottom_m=float(corners[:,2].min())), rows=[]))
        print(episode, "initial geometry invalid", outside, inset, flush=True)
        return

    def tick(command, effort, phase):
        hard_error = None
        try:
            last_tick(command, effort, phase)
        except MotionStop as error:
            hard_error = error
        finally:
            d = b.drive.data
            if np.isfinite(d.qpos).all():
                f = d.flexvert_xpos
                c, rotation = b.payload_box_pose()
                h = b.scene.half
                corners = CORNERS * h @ rotation.T + c
                faces = b.scene.parcel.envelope["faces"]
                n_out = int((np.abs(winding_number(f, faces, corners)) < .5).sum())
                n_in = int(triangle_box_intersections(f, faces, h - .002, c, rotation).sum())
                a, v = b.flex_edges
                stretch = max(np.linalg.norm(f[a]-f[v],axis=1) / b.flex_rest_length)
                _, *pads = b.contact_loads(check=False)
                rows.append(dict(time_s=float(d.time), phase=motion, pad_N=pads, core_bottom_m=float(corners[:, 2].min()),
                                 inset_triangles=n_in, outside_corners=n_out, max_edge_ratio=float(stretch)))
                states.append(d.qpos.copy())
        if hard_error:
            raise hard_error
        if rows and (rows[-1]["inset_triangles"] or rows[-1]["outside_corners"]):
            raise MotionStop("film_core_geometry", physics_valid=False)

    def follow(plan, effort, *, record=None, stage="grasp"):
        nonlocal motion
        if plan:
            motion = plan[0][2]
            print(episode, motion, float(b.drive.data.time), flush=True)
        result = follow_native(plan, effort, stage=stage, record=record)
        if motion == "lift":
            data = b.drive.data
            core_now = float(data.geom_xpos[b.payload_geom, 2])
            if core_now - initial_z < .006 and max(b.contact_loads(check=False)[1:]) < .05:
                raise MotionStop("payload_not_captured_at_lift")
            motion = "hold"
            for _ in range(26):
                b.tick(b.last_command, effort, "grasp")
        return result

    before = perf_counter()
    b.tick, b.follow = tick, follow
    try:
        result = env.execute_grasp(grasp)
    finally:
        b.tick, b.follow = last_tick, follow_native
    hold = [row for row in rows if row["phase"] == "hold"]
    peak = max((max(r["pad_N"]) for r in rows), default=0.)
    med = np.median([r["pad_N"] for r in hold], axis=0).tolist() if hold else None
    minimum = min((r["core_bottom_m"] - initial_floor for r in hold), default=None)
    held = hold[-1]["time_s"] - hold[0]["time_s"] if hold else 0.
    peak_edge = max((r["max_edge_ratio"] for r in rows), default=1.)
    valid_geometry = not any(r["inset_triangles"] or r["outside_corners"] for r in rows)
    down = [r for r in rows if r["phase"] in ("release", "retreat")]
    drop = max(0., down[0]["core_bottom_m"] - min(r["core_bottom_m"] for r in down)) if down else None
    through_gate = (result["task_success"] and result["physics_validity"] and valid_geometry and peak_edge <= 1.05
                    and peak <= 25. and held >= 1.-.0001 and minimum >= .04
                    and all(3. <= n <= 15. for n in med) and drop is not None and drop > .005)
    write_json(output / f"{episode}.json", dict(arm="Yam_B", mode=episode, position_m=point.tolist(),
                result=result, gate1_pass=bool(through_gate), elapsed_s=perf_counter()-before,
                penetration_event=b.drive.penetration_event(), hold_s=held, hold_min_lift_m=minimum,
                peak_pad_N=peak, hold_pad_median_N=med, release_drop_m=drop, rows=rows))
    if states:
        np.savez_compressed(output / f"{episode}.npz", qpos=np.asarray(states), time_s=[r["time_s"] for r in rows])
    print(episode, result, "strict pass", bool(through_gate), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--tuning", choices=("a1", "a2"), default="a2")
    p.add_argument("--modes", nargs="+", choices=("through", "film"), default=("through", "film"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    class TrialScene(FlexScene):
        def add_scene(self, root):
            super().add_scene(root)
            _dt, iterations, search, ratio, margin = TUNING[args.tuning]
            root.find("option").attrib.update(iterations=str(iterations), ls_iterations=str(search), impratio=str(ratio))
            root.find(".//flexcomp/contact").set("margin", str(margin))

    with patch("policy_v3.yam_backend.FlexScene", TrialScene):
        b = YamBackend(7, scene="flex", device="cuda:0", scripted=True,
                       limits=replace(SafetyLimits(), flex_edge_stretch=1.05))
    try:
        for case in args.modes:
            run(b, case, args.output)
    finally:
        b.close()
