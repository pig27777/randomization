"""Check whether the existing articulated bag can contain and support a free core."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "bag_sim"))
from loose_setup import BOX
from mj_adapter import OMEGA_MAX
from mj_loose import CONTENT_RELEASE_SHA256, LooseBag, LooseSim, loose_scene
from model import build, load_table


BAG_SIZE = (.35, .25)
DT = .0004
CASES = (
    ("loose_left", (.12, .08, .030), .150, -.065),
    ("loose_right", (.12, .08, .030), .150, .065),
    ("medium_center", (.18, .13, .035), .275, 0.),
    ("flat_full", (.29, .19, .020), .400, 0.),
)


def audit_geometry(sim):
    m = sim.m
    core_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, sim.obj_bodies)))
    bag_geoms = set(np.flatnonzero(np.isin(m.geom_bodyid, sim.bag_bodies)))
    core_bodies = set(sim.obj_bodies)
    free = [j for j in range(m.njnt) if m.jnt_bodyid[j] in core_bodies]
    if len(free) != 1 or m.jnt_type[free[0]] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("C0 requires a single six-DOF core")
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_CONNECT or m.eq_objtype[i] != mujoco.mjtObj.mjOBJ_SITE:
            raise RuntimeError(f"Unexpected equality in qualification scene: {m.equality(i).name}")
        a, b = m.eq_obj1id[i], m.eq_obj2id[i]
        if m.site_bodyid[a] not in sim.bag_bodies or m.site_bodyid[b] not in sim.bag_bodies:
            raise RuntimeError(f"Non-bag equality: {m.equality(i).name}")
    for i in range(m.nwrap):
        if (m.wrap_type[i] != mujoco.mjtWrap.mjWRAP_SITE
                or m.site_bodyid[m.wrap_objid[i]] not in sim.bag_bodies):
            raise RuntimeError("A tendon in the qualification scene reaches outside the bag")
    if m.npair:
        raise RuntimeError("Unexpected explicit collision pair in the static fixture")
    return core_geoms, bag_geoms


def measure(sim, core_geoms, bag_geoms):
    d = sim.d
    finite = all(np.isfinite(a).all() for a in (d.qpos, d.qvel, d.qacc))
    result = dict(time_s=float(d.time), finite=finite,
                  warnings={k.name: d.warning[k].number for k in mujoco.mjtWarning.__members__.values()
                            if k.name != "mjNWARNING" and d.warning[k].number})
    if not finite:
        return result

    force = np.zeros(6)
    vertical_bag_load = 0.
    unexpected_core_contact_force = 0.
    peak_bag_core_overlap = 0.
    for i, contact in enumerate(d.contact):
        a, b = contact.geom
        if a not in core_geoms and b not in core_geoms:
            continue
        other = b if a in core_geoms else a
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(sim.m, d, i, force)
        if other in bag_geoms:
            wrench_world = contact.frame.reshape(3, 3).T @ force[:3]
            vertical_bag_load += float(wrench_world[2]) * (1 if b in core_geoms else -1)
            peak_bag_core_overlap = max(peak_bag_core_overlap, -float(contact.dist))
        else:
            unexpected_core_contact_force += float(np.linalg.norm(force[:3]))

    center_out, corners_out = sim.outside_fraction()
    result.update(core_center_outside=bool(center_out), inset_core_corners_outside=int(round(8 * corners_out)),
                  reconstructed_penetration_m=sim.penetration(), bag_core_contact_overlap_m=peak_bag_core_overlap,
                  bag_vertical_load_on_core_N=vertical_bag_load,
                  unexpected_core_contact_force_N=unexpected_core_contact_force,
                  core_z_m=float(d.xpos[sim.obj_bodies[0], 2]))
    return result


def qualify(regime, size, mass, y, compress):
    # loose_scene takes the absolute core position and a fractional offset,
    # which together keep the nominal exterior center fixed across all cases.
    offset = y / ((BAG_SIZE[1] - size[1] - size[2]) / 2)
    core = replace(BOX["loose"], size=size, mass=mass)
    bag = LooseBag(size=BAG_SIZE, profile="pe", compress=compress, offset=(0., offset))
    # Use the existing wrapper with no pad bodies and no gripper bindings.
    desc = build(core, omega_max=OMEGA_MAX, table=load_table())
    sim = LooseSim(loose_scene(desc, bag, dt=DT, content_xy=(0., y)))
    core_geoms, bag_geoms = audit_geometry(sim)
    sim.wrap()
    samples = [measure(sim, core_geoms, bag_geoms)]
    for _ in range(5):
        for _ in range(round(.04 / DT)):
            sim.step()
        mujoco.mj_forward(sim.m, sim.d)
        samples.append(measure(sim, core_geoms, bag_geoms))
    weight = mass * abs(sim.m.opt.gravity[2])
    return dict(regime=regime, compression_fraction=compress, core_size_m=size, core_mass_kg=mass,
                core_weight_N=weight, core_offset_y_m=y, bag_size_m=BAG_SIZE, step_s=DT,
                plate_count=len(bag_geoms), equalities=sim.m.neq, tendons=sim.m.ntendon,
                core_dofs=6, samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        for case in CASES:
            for compression in (.9, 0.):
                result = qualify(*case, compression)
                result["content_archive_sha256"] = CONTENT_RELEASE_SHA256
                print(json.dumps(result, allow_nan=False), file=stream, flush=True)
                print(case[0], compression, result["samples"][-1], flush=True)


if __name__ == "__main__":
    main()
