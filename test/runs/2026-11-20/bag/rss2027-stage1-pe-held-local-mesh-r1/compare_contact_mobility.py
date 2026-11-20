"""Compute the effect of midpoint velocities on native normal contact mobility."""

import argparse
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np

from scan_mid_slide_contacts import RUN, backend, load_scenes, measure as measure_loads


def coarse_velocity_map(coarse, fine, edges):
    original, refined = coarse.film_dofs, fine.film_dofs
    shared = np.zeros((fine.m.nv, coarse.m.nv))
    shared[refined[:len(original)], original] = 1.
    shared[refined[len(original):], original[edges[:, 0]]] = .5
    shared[refined[len(original):], original[edges[:, 1]]] = .5
    coarse_other = np.setdiff1d(np.arange(coarse.m.nv), original)
    fine_other = np.setdiff1d(np.arange(fine.m.nv), refined)

    def joint_names(model, dofs):
        return [model.joint(index).name for index in model.dof_jntid[dofs]]

    if joint_names(coarse.m, coarse_other) != joint_names(fine.m, fine_other):
        raise ValueError("Non-film degrees of freedom differ between the two native models")
    shared[fine_other, coarse_other] = 1.
    np.testing.assert_allclose(shared @ coarse.d.qvel, fine.d.qvel, atol=1e-12, rtol=0.)
    return shared


def constraints(robot):
    sim = robot.sim
    model, data = sim.m, sim.d
    pads = set().union(*robot.pad_sets)
    backend.refresh(sim)
    if model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
        raise ValueError("A normal row must be the first constraint of an elliptic contact")
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    sparse = mujoco.mj_isSparse(model)
    dense = None if sparse else data.efc_J.reshape(data.nefc, model.nv)
    rows, records = [], []
    force = np.empty(6)
    for index, contact in enumerate(data.contact):
        ids = set(contact.geom).intersection(pads)
        address = int(contact.efc_address)
        if address < 0 or not ids or 0 not in contact.flex:
            continue
        pad, = ids
        if sparse:
            first = data.efc_J_rowadr[address]
            end = first + data.efc_J_rownnz[address]
            row = np.zeros(model.nv)
            row[data.efc_J_colind[first:end]] = data.efc_J[first:end]
        else:
            row = dense[address].copy()
        rows.append(row)
        mujoco.mj_contactForce(model, data, index, force)
        records.append(dict(index=index, geom=model.geom(pad).name, point_m=contact.pos.tolist(),
                            normal=contact.frame[:3].tolist(), normal_N=float(force[0]),
                            normal_regularizer=float(data.efc_R[address])))
    return mass, np.asarray(rows), records


def mobility_diagonal(mass, jacobian):
    return np.einsum("ij,ji->i", jacobian, np.linalg.solve(mass, jacobian.T))


def measure(coarse, fine, edges):
    velocity_map = coarse_velocity_map(coarse.sim, fine.sim, edges)
    coarse_mass, coarse_jac, base = constraints(coarse)
    fine_mass, fine_jac, split = constraints(fine)
    shared_jac = fine_jac @ velocity_map
    shared_mass = velocity_map.T @ fine_mass @ velocity_map
    coarse_mobility = mobility_diagonal(coarse_mass, coarse_jac)
    fine_mobility = mobility_diagonal(fine_mass, fine_jac)
    shared_mobility = mobility_diagonal(shared_mass, shared_jac)
    options = []
    for i, j in product(range(len(base)), range(len(split))):
        a, b = base[i], split[j]
        separation = np.linalg.norm(np.subtract(a["point_m"], b["point_m"]))
        normal_dot = np.dot(a["normal"], b["normal"])
        if a["geom"] == b["geom"] and separation <= 1e-6 and normal_dot >= np.cos(np.deg2rad(1.)):
            options.append((float(separation), i, j))
    matched, used_coarse, used_fine = [], set(), set()
    for distance, i, j in sorted(options):
        if i in used_coarse or j in used_fine:
            continue
        used_coarse.add(i)
        used_fine.add(j)
        jac_change = np.linalg.norm(shared_jac[j] - coarse_jac[i]) / np.linalg.norm(coarse_jac[i])
        matched.append(dict(coarse_index=i, fine_index=j, distance_m=distance,
                            normal_jacobian_relative_difference=float(jac_change),
                            coarse_inverse_mass_per_kg=float(coarse_mobility[i]),
                            fine_inverse_mass_per_kg=float(fine_mobility[j]),
                            projected_inverse_mass_per_kg=float(shared_mobility[j])))
    return dict(coarse_contacts=base, fine_contacts=split, matched=matched,
                unmatched_coarse=[i for i in range(len(base)) if i not in used_coarse],
                unmatched_fine=[j for j in range(len(split)) if j not in used_fine])


def solve_contact_subset(robot, pull, unmatched):
    sim = robot.sim
    m, d = sim.m, sim.d
    samples = []
    for excluded in ([], unmatched):
        backend.refresh(sim)
        before = measure_loads(robot, pull)
        for i in excluded:
            assert d.contact[i].efc_address >= 0
            d.contact[i].exclude = 1
            d.contact[i].efc_address = -1
        mujoco.mj_makeConstraint(m, d)
        mujoco.mj_projectConstraint(m, d)
        mujoco.mj_fwdVelocity(m, d)
        sim.parcel.apply_internal(m, d, sim.film_dofs)
        mujoco.mj_forwardSkip(m, d, mujoco.mjtStage.mjSTAGE_VEL, 1)
        contact_types = (mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS,
                         mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)
        active_ids = d.efc_id[np.isin(d.efc_type, contact_types)]
        assert set(excluded).isdisjoint(active_ids)
        assert all(d.contact[i].efc_address == -1 for i in excluded)
        after = measure_loads(robot, pull)
        if not excluded:
            np.testing.assert_allclose(after["loads_N"], before["loads_N"], atol=1e-8, rtol=0.)
        samples.append(dict(excluded_contact_ids=excluded, **after))
    return samples


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--paired-only", action="store_true", help="also solve with unpaired jaw contacts omitted")
    args = parser.parse_args()
    coarse, fine, native, fingerprint, wrench = load_scenes()
    with np.load(RUN / "mid-slide-mesh/prepared_mesh.npz") as mesh:
        report = measure(coarse, fine, mesh["old_edges"])
    report.update(source_sha256=fingerprint, payload_wrench=wrench.tolist())
    if args.paired_only:
        pull = np.asarray(native["force_direction"])
        solves = [solve_contact_subset(robot, pull, [report[f"{mesh}_contacts"][i]["index"]
                                                    for i in report[f"unmatched_{mesh}"]])
                  for mesh, robot in (("coarse", coarse), ("fine", fine))]
        relative = [(np.asarray(second["loads_N"]) / first["loads_N"] - 1.).tolist()
                    for first, second in zip(*solves, strict=True)]
        report["shared_contact_study"] = dict(coarse=solves[0], fine=solves[1], relative_difference=relative)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Compared {len(report['matched'])} pairs of matching native contacts", flush=True)


if __name__ == "__main__":
    main()
