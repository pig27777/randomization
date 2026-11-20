"""Halve the jaw-local mesh on the same measured whole-bag gathering window."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from replay_ridge_window import CASE, RUN, assemble, backend, load_window, replay
from record_local_top import at_top_center

sys.path.insert(0, str(RUN.with_name("rss2027-stage1-kraft-held-local-mesh-r1")))
import replay_refined as mesh  # noqa: E402

OUTPUT = CASE / "local-mesh-window"
DT = .0000125


def load_pair(prepare_only=False, *, case=CASE, timestep=DT):
    output = case / "local-mesh-window"
    result = json.loads((case / "recording/trial/result.json").read_bytes())
    reference = json.loads((case / "physical-comparison.json").read_bytes())
    height = json.loads((case / "height.json").read_bytes())
    original = backend.Config(**result["config"])
    coarse = assemble(case)
    time, command, state, spec = load_window(coarse, reference, case)

    def saved_scene():
        robot = assemble(case)
        mujoco.mj_setState(robot.sim.m, robot.sim.d, state, spec)
        backend.refresh(robot.sim)
        fingerprint = hashlib.sha256(state.tobytes()).hexdigest()
        return robot, float(time[0]), fingerprint

    asset = output / "prepared_mesh.npz"
    geometry = "../" + asset.relative_to(mesh.ROOT).as_posix()
    with patch.multiple(mesh, RUN=output, ASSET=asset, GEOMETRY=geometry, COMPILED_DT=timestep,
                        original_scene=saved_scene, get_config=lambda: original), \
            at_top_center(height["target_center_xy_m"], height.get("top_reference_m")):
        fine_scene = partial(mesh.refined_scene, source=saved_scene, movable_support=False)
        if prepare_only:
            output.mkdir(exist_ok=False)
            with patch.object(mesh, "refined_scene", fine_scene):
                mesh.prepare()
            return
        fine, _, _ = fine_scene()

    pressure = backend.measure(fine)
    np.testing.assert_array_equal(fine.sim.d.ctrl, command[0])
    if pressure["direct_support_N"] > .05:
        raise ValueError("The refined gripper directly loads the table in its initial state")
    refined_state = mesh.integration_state(fine.sim)
    return coarse, fine, time, command, state, refined_state, spec


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--step-us", type=float, choices=(12.5, 25., 50.), default=DT * 1e6)
    args = parser.parse_args()
    case = args.case.resolve()
    timestep = args.step_us / 1e6
    pair = load_pair(args.prepare, case=case, timestep=timestep)
    if args.prepare:
        return
    coarse, fine, time, command, state, refined_state, spec = pair
    coarse_result = replay(coarse, timestep, time, command, state, spec)
    refined_result = replay(fine, timestep, time, command, refined_state, spec)
    fits = [run["fit"] for run in (coarse_result, refined_result)]
    changes = {}
    if all(fit is not None for fit in fits):
        for key in ("slope_m_per_m", "growth_m", "displacement_m", "mean_pressure_N"):
            base, changed = (fit[key] for fit in fits)
            if base and changed is not None:
                changes[key] = changed / base - 1.
    passed = (len(changes) == 4 and all(fit["eligible"] for fit in fits)
              and all(abs(value) <= .10 for value in changes.values()))
    reference_changes = {}
    if all(fit is not None and fit["reference_eligible"] for fit in fits):
        for key in ("reference_growth_m", "reference_slope_m_per_m", "displacement_m", "mean_pressure_N"):
            if fits[0][key]:
                reference_changes[key] = fits[1][key] / fits[0][key] - 1.
    reference_passed = (len(reference_changes) == 4
                        and all(abs(value) <= .10 for value in reference_changes.values()))
    comparison = dict(coarse=coarse_result, refined=refined_result,
                      relative_change=changes, passed=passed,
                      reference_relative_change=reference_changes, reference_passed=reference_passed)
    with (case / "local-mesh-window/comparison.json").open("x") as stream:
        stream.write(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    print(passed, changes, flush=True)


if __name__ == "__main__":
    main()
