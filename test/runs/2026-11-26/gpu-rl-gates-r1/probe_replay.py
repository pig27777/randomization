"""Measure the first Flex probe stop and replay the guarded Try over three inner states."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from bag_sim.wrap import triangle_box_intersections, winding_number
from policy_v3.environment import MacroBagGraspEnv
from policy_v3.workers import write_json
from policy_v3.yam_backend import YamBackend
from policy_v3.yam_scene import CORNERS, PICK
from yam_platform.yam_model import save_yam_xml

OLD = Path("test/runs/2026-11-25/policy/policy-v3-mvp-r1/flex-probe")


def run(backend, out, name, offset):
    out.mkdir()
    old_scene_episode = backend.scene.episode
    original_tick = backend.tick
    trace = []

    class Offset:
        def uniform(self, _lower, _upper):
            return offset

    def episode(model, initial, rng, scripted):
        return old_scene_episode(model, initial, Offset(), False)

    def tick(command, effort, phase):
        stamp = perf_counter()
        try:
            original_tick(command, effort, phase)
        finally:
            d = backend.drive.data
            if np.isfinite(d.qpos).all():
                film = d.flexvert_xpos
                center, rotation = backend.payload_box_pose()
                half = backend.scene.half
                faces = backend.scene.parcel.envelope["faces"]
                corners = CORNERS * half @ rotation.T + center
                outside = int(np.count_nonzero(np.abs(winding_number(film, faces, corners)) < .5))
                inset = int(triangle_box_intersections(film, faces, half - .002, center, rotation).sum())
                load, *pads = backend.contact_loads(check=False)
                a, b = backend.flex_edges
                stretch = max(np.linalg.norm(film[a] - film[b], axis=1) / backend.flex_rest_length)
                trace.append(dict(t_s=float(d.time), phase=phase, outside_corners=outside, inset_triangles=inset,
                                  contact_N=float(load), pads_N=pads, peak_substep_N=float(backend.drive.peak_contact.numpy()[0]),
                                  max_edge_ratio=float(stretch), payload_lowest_m=float(corners[:,2].min()),
                                  wall_s=perf_counter() - stamp))

    backend.probe_cutoff_N = None if name == "archived" else 2.
    backend.scene.episode = episode
    backend.tick = tick
    env = MacroBagGraspEnv(backend, np.random.default_rng(7))
    v0, _, _ = env.reset(seed=7)
    if name == "archived":
        import json
        action = np.array(json.loads((OLD / "scripted.json").read_text())["a_try"])
    else:
        action = np.zeros(13)
        local_xy = ((PICK[:2] + [0., -.015]) - v0.center[:2]) @ v0.rotation[:2, :2]
        action[:2] = np.clip(local_xy / v0.half_extent[:2], -1., 1.)
        action[8], action[9], action[12] = -.35, .12, .30
    start = perf_counter()
    failed_initial_geometry = any(row["inset_triangles"] or row["outside_corners"] for row in trace)
    if name != "archived" and failed_initial_geometry:
        h, n, v1 = np.empty((0, 48)), 0, None
        meta = dict(physics_validity=False, stop_reason="initial_film_core_geometry", retracted=backend.at_hover())
    else:
        h, n, v1, meta = env.execute_probe(action)
    stop = backend.drive.penetration_event()
    peak = float(backend.drive.peak_contact.numpy()[0])
    valid = (meta["physics_validity"] and meta["retracted"] and v1 is not None and .5 <= peak <= 3.5
             and not any(row["inset_triangles"] or row["outside_corners"] for row in trace))
    data = dict(case=name, payload_offset_m=offset, film_gate_pass=valid, probe=meta, frames=n, action=action.tolist(),
                v0=v0.to_dict(), v1=v1.to_dict() if v1 is not None else None, peak_substep_N=peak, stop_contact=stop,
                wall_s=perf_counter() - start, rows=trace)
    write_json(out / "summary.json", data)
    np.savez_compressed(out / "probe.npz", H_try=h, H_try_raw=env.raw_history, timestamps=env.trace.timestamps,
                        phases=env.trace.stages, trajectory_world=env.trace.tcp_world)
    backend.tick = original_tick
    backend.scene.episode = old_scene_episode
    print({key: value for key, value in data.items() if key != "rows"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--modes", nargs="+", choices=("archived", "left", "center", "right"),
                   default=("archived", "left", "center", "right"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    backend = YamBackend(7, scene="flex", device="cuda:0", scripted=True)
    try:
        save_yam_xml(backend.drive.sim.xml, a.output / "scene.xml")
        for name in a.modes:
            run(backend, a.output / name, name, {"archived": 0., "left": -.018, "center": 0., "right": .018}[name])
    finally:
        backend.close()
