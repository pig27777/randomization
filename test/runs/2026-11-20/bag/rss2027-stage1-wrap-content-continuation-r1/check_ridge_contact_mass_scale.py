"""Check native contact scaling at three unchanged, recorded YAM poses."""

import json

import mujoco
import numpy as np

from replay_ridge_window import assemble, backend
from scan_ridge_mesh_resolution import CASE, OUTPUT, STATIC_STEP_S, refine


def native_contacts(robot):
    model, data = robot.sim.m, robot.sim.d
    model.opt.timestep = STATIC_STEP_S
    backend.refresh(robot.sim)
    primary = set().union(*robot.pad_sets)
    forces = np.zeros(6)
    rows = []
    for index, contact in enumerate(data.contact):
        address = contact.efc_address
        geoms = primary.intersection(contact.geom)
        if address < 0 or not geoms or 0 not in contact.flex:
            continue
        geom, = geoms
        impedance = data.efc_KBIP[address, 2]
        regularizer = data.efc_R[address]
        mass = (1. - impedance) / (impedance * regularizer)
        if not np.isfinite(mass) or mass <= 0.:
            raise ValueError("The primary contact row has no finite effective mass")
        mujoco.mj_contactForce(model, data, index, forces)
        rows.append(dict(geom=model.geom(geom).name, depth_m=float(-contact.dist),
                         effective_mass_kg=float(mass), normal_force_N=float(forces[0]),
                         reference_acceleration_m_s2=float(data.efc_aref[address]),
                         solref=contact.solref.tolist()))
    feedback = backend.measure(robot)
    vertical = [force[2] for force in feedback["force_on_fingers_N"]]
    return dict(contact_mass_kg=sum(row["effective_mass_kg"] for row in rows), vertical_force_N=vertical,
                total_pressure_N=sum(vertical), penetration_mm=feedback["penetration_mm"],
                direct_support_N=feedback["direct_support_N"], contacts=rows)


def scale_primary_contacts(robot, factor):
    model = robot.sim.m
    ids = sorted(set().union(*robot.pad_sets))
    time_constant, damping_ratio = model.geom_solref[ids].T.copy()
    if np.any(time_constant <= 0.) or np.any(damping_ratio <= 0.):
        raise ValueError("The original model requires positive time-constant parameters")
    # Scale both coefficients of the native reference acceleration.
    model.geom_solref[ids, 0] = -factor / (time_constant * damping_ratio)**2
    model.geom_solref[ids, 1] = -2. * factor / time_constant


def models(state, spec, cfg, height):
    robot = assemble(CASE)
    mujoco.mj_setState(robot.sim.m, robot.sim.d, state, spec)
    backend.refresh(robot.sim)
    yield robot
    for folder in (OUTPUT.parent, *(OUTPUT / f"level-{level}" for level in (2, 3, 4))):
        robot, cfg = refine(robot, cfg, height, folder, prepare=False)
        yield robot


def main():
    result = json.loads((CASE / "recording/trial/result.json").read_bytes())
    height = json.loads((CASE / "height.json").read_bytes())
    cfg = backend.Config(**result["config"])
    times = (9.2601, 9.3401, 9.4801)
    with np.load(CASE / "recording/trial/trajectory.npz") as archive:
        choices = [int(np.argmin(abs(archive["time"] - target))) for target in times]
        np.testing.assert_allclose(archive["time"][choices], times, atol=1e-8, rtol=0.)
        states = archive["integration_state"][choices]
        spec = int(archive["integration_state_spec"])
    factors, frames = None, []
    for time, state in zip(times, states, strict=True):
        robots = list(models(state, spec, cfg, height))
        native = [native_contacts(robot) for robot in robots]
        if factors is None:
            factors = [native[0]["contact_mass_kg"] / result["contact_mass_kg"] for result in native]
        result = []
        for robot, raw, factor in zip(robots, native, factors, strict=True):
            scale_primary_contacts(robot, factor)
            scaled = native_contacts(robot)
            np.testing.assert_allclose(scaled["penetration_mm"], raw["penetration_mm"], atol=1e-9, rtol=0.)
            assert [(row["geom"], row["depth_m"]) for row in scaled["contacts"]] == [
                (row["geom"], row["depth_m"]) for row in raw["contacts"]]
            result.append(dict(nodes=robot.sim.m.nflexvert, before=raw, after=scaled))
        frames.append(dict(t_s=time, samples=result))
    with (OUTPUT / "mass-scaled-contact.json").open("x") as stream:
        stream.write(json.dumps(dict(scale_factors=factors, frames=frames), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
