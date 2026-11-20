"""Isolate midpoint motion and lumped mass in a recorded PE contact state."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from scan_mid_slide_contacts import RUN, backend, load_scenes, measure


def tie_midpoints(spec, old_count, edges):
    joints = []
    for index in range(old_count + len(edges)):
        point = spec.body(f"bag_{index}")
        for axis, joint in enumerate(point.joints):
            joint.name = f"film_node_{index}_{axis}"
        joints.append([joint.name for joint in point.joints])
    for index, (left, right) in enumerate(edges, start=old_count):
        for axis in range(3):
            tendon = spec.add_tendon(name=f"edge_midpoint_{index}_{axis}")
            tendon.wrap_joint(joints[index][axis], 1.)
            tendon.wrap_joint(joints[left][axis], -.5)
            tendon.wrap_joint(joints[right][axis], -.5)
            spec.add_equality(type=mujoco.mjtEq.mjEQ_TENDON, objtype=mujoco.mjtObj.mjOBJ_TENDON,
                              name1=tendon.name, data=[0.] * mujoco.mjNEQDATA, active=1,
                              solref=[.0001, 1.], solimp=[.9999, .9999, .001, .5, 2.])


def preserve_coarse_mass(coarse, fine, fine_nodes, edges):
    old_count = coarse.nflexvert
    old_nodes = [coarse.body(f"bag_{i}").id for i in range(old_count)]
    shared = fine.body_mass.copy()
    midpoints = shared[fine_nodes[old_count:]] * .001
    original = coarse.body_mass[old_nodes].copy()
    for side in range(2):
        np.add.at(original, edges[:, side], -.5 * midpoints)
    if original.min() <= 0.:
        raise ValueError("The midpoint mass split would leave a nonpositive parent mass")
    shared[fine_nodes] = np.r_[original, midpoints]
    np.testing.assert_allclose(shared[fine_nodes].sum(), fine.body_mass[fine_nodes].sum(), atol=1e-12, rtol=0.)
    return shared


def forward_variant(robot, spec, mass, pull):
    source = robot.sim
    old_model, old_data = source.m, source.d
    rebuilt = spec.compile()
    np.testing.assert_array_equal(rebuilt.flex_vertbodyid, old_model.flex_vertbodyid)
    np.testing.assert_array_equal(rebuilt.jnt_qposadr, old_model.jnt_qposadr)
    np.testing.assert_array_equal(rebuilt.jnt_dofadr, old_model.jnt_dofadr)
    rebuilt.opt.timestep = old_model.opt.timestep
    rebuilt.body_mass[:] = mass
    data = mujoco.MjData(rebuilt)
    mujoco.mj_setConst(rebuilt, data)
    data.time = old_data.time
    for key in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied",
                "mocap_pos", "mocap_quat"):
        getattr(data, key)[:] = getattr(old_data, key)
    data.eq_active[:old_model.neq] = old_data.eq_active
    with patch.multiple(source, m=rebuilt, d=data):
        backend.refresh(source)
        np.testing.assert_allclose(data.flexvert_xpos, old_data.flexvert_xpos, atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(data.qvel, old_data.qvel)
        np.testing.assert_array_equal(data.xfrc_applied, old_data.xfrc_applied)
        result = measure(robot, pull)
    nodes = rebuilt.flex_vertbodyid
    center = np.average(data.flexvert_xpos, axis=0, weights=mass[nodes])
    old_center = np.average(old_data.flexvert_xpos, axis=0, weights=old_model.body_mass[nodes])
    selected = ((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id >= old_model.neq))
    return dict(**result, film_mass_kg=float(mass[nodes].sum()), film_center_m=center.tolist(),
                center_change_m=float(np.linalg.norm(center - old_center)),
                added_equalities=rebuilt.neq - old_model.neq, active_equalities=int(selected.sum()),
                max_midpoint_residual_m=float(np.abs(data.efc_pos[selected]).max(initial=0.)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compile_model = backend.Parcel.compile_model
    sources = []

    def capture_xml(parcel, xml):
        model = compile_model(parcel, xml)
        sources.append((model, xml))
        return model

    with patch.object(backend.Parcel, "compile_model", capture_xml):
        coarse, fine, native, fingerprint, wrench = load_scenes()
    xml, = [xml for model, xml in sources if model is fine.sim.m]
    pull = np.asarray(native["force_direction"])
    for robot in (coarse, fine):
        backend.refresh(robot.sim)
    baseline = measure(coarse, pull)
    original = measure(fine, pull)
    model = fine.sim.m
    nodes = model.flex_vertbodyid
    with np.load(RUN / "mid-slide-mesh/prepared_mesh.npz") as mesh:
        edges = mesh["old_edges"]
    spec = mujoco.MjSpec.from_string(xml)
    reference = forward_variant(fine, spec, model.body_mass, pull)
    np.testing.assert_allclose(reference["loads_N"], original["loads_N"], atol=1e-8, rtol=0.)
    tie_midpoints(spec, coarse.sim.m.nflexvert, edges)
    shared_mass = preserve_coarse_mass(coarse.sim.m, model, nodes, edges)
    runs = {}
    for label, mass in (("original_refined_mass", model.body_mass), ("coarse_mass_distribution", shared_mass)):
        result = forward_variant(fine, spec, mass, pull)
        assert result["active_equalities"] == 3 * len(edges)
        assert result["max_midpoint_residual_m"] < 1e-10
        relative = np.asarray(result["loads_N"]) / baseline["loads_N"] - 1.
        result["relative_difference"] = relative.tolist()
        result["matched"] = bool(np.abs(relative).max() <= .1 and result["support_N"] <= .05
                                 and result["penetration_mm"] <= 1. and not result["warnings"])
        runs[label] = result
    record = dict(source_sha256=fingerprint, payload_wrench=wrench.tolist(), coarse=baseline,
                  refined=reference, constrained_refined=runs)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print({key: row["relative_difference"] for key, row in runs.items()}, flush=True)


if __name__ == "__main__":
    main()
