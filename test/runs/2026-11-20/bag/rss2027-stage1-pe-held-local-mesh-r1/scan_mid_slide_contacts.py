"""Test whether jaw pose offsets recover the archived PE sliding-state loads."""

import argparse
from functools import partial
from itertools import product
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from replay_mid_slide_mesh import RUN, SLIDE, _ORIGINAL_SCENE, configure, experiment, original_mid_slide

backend = experiment.backend
replay = experiment.replay


def measure(robot, pull):
    feedback = backend.measure(robot)
    normal, _ = replay.inner_loads(replay.release_contacts(robot))
    tension = float(np.sum(feedback["force_on_fingers_N"], axis=0) @ pull)
    return dict(loads_N=[*normal.tolist(), tension], support_N=feedback["support_N"],
                penetration_mm=feedback["penetration_mm"], warnings=int(robot.sim.d.warning.number.sum()))


def scan(coarse, refined, pull):
    backend.refresh(coarse.sim)
    baseline = measure(coarse, pull)
    expected = np.asarray(baseline["loads_N"])
    if (min(expected) <= .001 or baseline["support_N"] > .05 or baseline["penetration_mm"] > 1.
            or baseline["warnings"]):
        raise ValueError(f"Archived coarse loads do not qualify: {baseline}")
    sim = refined.sim
    model, data = sim.m, sim.d
    state = experiment.integration_state(sim)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    original_film = data.flexvert_xpos.copy()
    original_payload = sim.com().copy()
    axis, center = replay.jaw_axis(refined)
    original_gap = refined.gap()
    position_jacobian, rotation_jacobian = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, position_jacobian, rotation_jacobian, refined.site)
    jacobian = np.vstack((position_jacobian[:, refined.dofs], rotation_jacobian[:, refined.dofs]))
    arm_displacement_per_m = np.linalg.solve(jacobian, np.r_[axis, np.zeros(3)])

    samples = []
    for opening, offset in product(np.linspace(-.0006, .0006, 9), np.linspace(-.0003, .0003, 9)):
        mujoco.mj_setState(model, data, state, spec)
        data.qpos[refined.qadr] += arm_displacement_per_m * offset
        data.qpos[refined.fingers] += opening / 2.
        backend.refresh(sim)
        np.testing.assert_array_equal(data.flexvert_xpos, original_film)
        np.testing.assert_array_equal(sim.com(), original_payload)
        _, actual_center = replay.jaw_axis(refined)
        observation = measure(refined, pull)
        relative = (np.asarray(observation["loads_N"]) - expected) / expected
        valid = observation["support_N"] <= .05 and observation["penetration_mm"] <= 1. and not observation["warnings"]
        samples.append(dict(opening_offset_m=float(opening), center_offset_m=float(offset),
                            measured_opening_offset_m=refined.gap() - original_gap,
                            measured_center_offset_m=float((actual_center - center) @ axis),
                            observation=observation, relative_difference=relative.tolist(), numerically_valid=valid,
                            normal_match=bool(valid and np.max(np.abs(relative[:2])) <= .1),
                            full_match=bool(valid and np.max(np.abs(relative)) <= .1)))

    valid = [row for row in samples if row["numerically_valid"]]
    if not valid:
        raise ValueError("No offsets meet the original numerical and support checks")
    closest = min(valid, key=lambda row: max(map(abs, row["relative_difference"])))
    normal_closest = min(valid, key=lambda row: max(map(abs, row["relative_difference"][:2])))
    return dict(source_time_s=float(data.time), original_gap_m=original_gap, coarse_mesh_vertices=coarse.sim.m.nflexvert,
                refined_mesh_vertices=model.nflexvert, load_order=("left_inner", "right_inner", "tangential_pull"),
                baseline=baseline, unadjusted_refined=samples[len(samples) // 2],
                normal_matches=sum(row["normal_match"] for row in samples),
                full_matches=sum(row["full_match"] for row in samples), closest=closest,
                closest_normal=normal_closest, samples=samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    configure()
    source = SLIDE / "local-tangent-exact"
    native = json.loads((source / "dt-25us.json").read_bytes())["native_load"]
    checkpoint = partial(original_mid_slide, _ORIGINAL_SCENE, source)
    mesh = RUN / "mid-slide-mesh/prepared_mesh.npz"
    geometry = "../" + mesh.relative_to(experiment.ROOT).as_posix()
    coarse, _, fingerprint = checkpoint()
    coarse.sim.m.opt.timestep = native["dt_s"]
    with patch.multiple(experiment, ASSET=mesh, GEOMETRY=geometry, COMPILED_DT=native["dt_s"]):
        refined, _, _ = experiment.refined_scene(source=checkpoint)
    result = scan(coarse, refined, np.asarray(native["force_direction"]))
    result.update(source_sha256=fingerprint, force_direction=native["force_direction"])
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Matching normal loads: {result['normal_matches']}; all three loads: {result['full_matches']}", flush=True)


if __name__ == "__main__":
    main()
