"""Compare several jaw-local mesh sizes at the same recorded folded state."""

from collections import Counter
from dataclasses import replace
from functools import partial
import hashlib
import json
from unittest.mock import patch

import numpy as np

from record_local_top import RUN, at_top_center, backend
from replay_ridge_mesh import load_pair, mesh
from record_physical_parcel import release_contacts

CASE = RUN / "pe-faster-ridge-source"
OUTPUT = CASE / "local-mesh-window/resolution-sweep"
STATIC_STEP_S = .00000078125


def refine(source_robot, config, height, output, *, prepare=True):
    state = mesh.integration_state(source_robot.sim)
    fingerprint = hashlib.sha256(state.tobytes()).hexdigest()

    def source():
        return source_robot, float(source_robot.sim.d.time), fingerprint

    asset = output / "prepared_mesh.npz"
    geometry = "../" + asset.relative_to(mesh.ROOT).as_posix()
    next_scene = partial(mesh.refined_scene, source=source, movable_support=False)
    with patch.multiple(mesh, RUN=output, ASSET=asset, GEOMETRY=geometry, COMPILED_DT=STATIC_STEP_S,
                        original_scene=source, get_config=lambda: config, refined_scene=next_scene), \
            at_top_center(height["target_center_xy_m"], height["top_reference_m"]):
        if prepare:
            output.mkdir(exist_ok=False)
            mesh.prepare()
        robot, _, _ = next_scene()
    return robot, replace(config, geometry=geometry, dt=STATIC_STEP_S)


def measure(robot, level, median_edge):
    robot.sim.m.opt.timestep = STATIC_STEP_S
    backend.refresh(robot.sim)
    feedback = backend.measure(robot)
    contacts = release_contacts(robot, include_all_fingers=True)
    vectors = np.zeros((2, 3))
    for point in contacts:
        vectors[point["pad"]] += point["force_on_finger_N"]
    np.testing.assert_allclose(vectors, feedback["force_on_fingers_N"], atol=1e-10, rtol=0.)
    return dict(level=level, film_vertices=robot.sim.m.nflexvert, median_edge_m=median_edge,
                actual_gap_m=robot.gap(), time_s=float(robot.sim.d.time), timestep_s=STATIC_STEP_S,
                penetration_mm=feedback["penetration_mm"], direct_support_N=feedback["direct_support_N"],
                normal_pressure_N=float(vectors[:, 2].sum()), force_on_fingers_N=vectors.tolist(),
                contacts_by_geom=dict(Counter(point["geom"] for point in contacts)))


def main():
    result = json.loads((CASE / "recording/trial/result.json").read_bytes())
    height = json.loads((CASE / "height.json").read_bytes())
    original = backend.Config(**result["config"])
    folder = OUTPUT.parent
    prep = json.loads((folder / "preparation.json").read_bytes())
    coarse, robot, *_ = load_pair(case=CASE, timestep=.000025)
    report = [measure(coarse, 0, prep["split_edge_median_m"]),
              measure(robot, 1, prep["refined_edge_median_m"])]
    config = replace(original, geometry="../" + (folder / "prepared_mesh.npz").relative_to(mesh.ROOT).as_posix())
    OUTPUT.mkdir(exist_ok=False)
    for level in (2, 3, 4):
        target = OUTPUT / f"level-{level}"
        robot, config = refine(robot, config, height, target)
        prep = json.loads((target / "preparation.json").read_bytes())
        report.append(measure(robot, level, prep["refined_edge_median_m"]))
    with (OUTPUT / "force-sweep.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
