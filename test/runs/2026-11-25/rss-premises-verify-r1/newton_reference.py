"""Probe the native Newton MJWarp/VBD feedback path with the repository's Yam_B."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import newton
import numpy as np
import warp as wp
from isaaclab_contrib.coupling import (
    CouplerEntryCfg,
    CouplerProxyCfg,
    CouplerProxyMappingCfg,
    NewtonCouplerManager,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager, VBDSolverCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from src.isaac_yam_parcel.yam_scene import YAM_USD, load_scene, load_workcell


BAG_SIZE = np.array([.220, .160, .036])
PAD_WIDTH = .0144
POLICY_DT = .04
RADIUS = .00025
MATERIAL = dict(young_pa=6.427e6, thickness_m=.000193, density_kg_m3=851., poisson=.3,
                bend_nm=2.e-6, friction=.35)


def closed_mesh(cells):
    """Generate shared, outward-facing vertices on a sealed rectangular envelope."""
    vertices, faces, labels = [], [], {}
    for normal in range(3):
        a, b = (normal + 1) % 3, (normal + 2) % 3
        for upper in (False, True):
            for i in range(cells[a]):
                for j in range(cells[b]):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        lattice = [0] * 3
                        lattice[normal] = cells[normal] if upper else 0
                        lattice[a], lattice[b] = i + di, j + dj
                        key = tuple(lattice)
                        if key not in labels:
                            labels[key] = len(vertices)
                            vertices.append((np.asarray(key) / cells - .5) * BAG_SIZE)
                        quad.append(labels[key])
                    if not upper:
                        quad.reverse()
                    faces.extend((quad[:3], [quad[0], quad[2], quad[3]]))
    v, f = np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    edges, uses = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
                            axis=0, return_counts=True)
    if not np.all(uses == 2):
        raise ValueError("The bag must be a closed manifold")
    rest = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    return v, f, edges.astype(np.int32), rest.astype(np.float32)


@wp.kernel
def max_stretch(positions: wp.array(dtype=wp.vec3), edges: wp.array(dtype=wp.vec2i),
                rest: wp.array(dtype=float), maximum: wp.array(dtype=float)):
    i = wp.tid()
    edge = edges[i]
    ratio = wp.length(positions[edge[1]] - positions[edge[0]]) / rest[i]
    if wp.isfinite(ratio):
        wp.atomic_max(maximum, 0, ratio)
    else:
        wp.atomic_max(maximum, 0, 1.e10)


@wp.kernel
def direct_core_contacts(count: wp.array(dtype=int), geoms: wp.array(dtype=wp.vec2i),
                         shape_map: wp.array(dtype=int), address: wp.array2d(dtype=int),
                         constraint_force: wp.array2d(dtype=float), core: int,
                         first_robot_shape: int, last_robot_shape: int, hits: wp.array(dtype=int)):
    i = wp.tid()
    if i < count[0]:
        pair = geoms[i]
        if pair[0] >= 0 and pair[1] >= 0:
            a, b = shape_map[pair[0]], shape_map[pair[1]]
            if ((a == core and first_robot_shape <= b and b <= last_robot_shape)
                    or (b == core and first_robot_shape <= a and a <= last_robot_shape)):
                wp.atomic_add(hits, 0, 1)
                for slot in range(address.shape[1]):
                    row = address[i, slot]
                    if row >= 0 and wp.abs(constraint_force[0, row]) > 1.e-6:
                        wp.atomic_add(hits, 1, 1)
                        break


def new_builder():
    return NewtonMJWarpManager.create_builder(up_axis="Z", physics_cfg=NewtonCfg(), gravity=(0., 0., -9.81))


def ydofs(builder):
    joints = {name.rsplit("/", 1)[-1]: j for j, name in enumerate(builder.joint_label)}
    ids = [joints[f"joint{i}"] for i in range(1, 9)]
    return np.asarray([builder.joint_q_start[j] for j in ids]), np.asarray([builder.joint_qd_start[j] for j in ids])


def robot_scene():
    scene, workcell = load_scene(), load_workcell()
    arm = workcell["arms"]["arm_a"]
    if arm["prim"] != "Yam_B":
        raise ValueError("Expected the right-arm Yam_B asset configuration")
    builder = new_builder()
    builder.default_shape_cfg.configure_sdf(force_sdf=True, target_voxel_size=.0025)
    transform = wp.transform(wp.vec3(*scene["arm_bases_m"][arm["abc_index"]]), wp.quat_identity())
    builder.add_usd(str(YAM_USD), xform=transform, floating=False, load_visual_shapes=False,
                    load_static_visual_shapes=False, parse_mujoco_options=False,
                    skip_mesh_approximation=True,
                    override_root_xform=True, force_position_velocity_actuation=True)
    qidx, didx = ydofs(builder)
    target = np.asarray([*arm["ready_joints_rad"], scene["ready_rail_m"], scene["ready_rail_m"]], dtype=np.float32)
    for i, (q, d) in enumerate(zip(qidx, didx, strict=True)):
        builder.joint_q[q] = float(target[i])
        builder.joint_target_q[q] = float(target[i])
        builder.joint_target_ke[d] = (80., 80., 80., 10., 10., 10., 1000., 1000.)[i]
        builder.joint_target_kd[d] = (5., 5., 5., 1.5, 1.5, 1.5, 16.7, 16.7)[i]
        builder.joint_effort_limit[d] = (28., 28., 28., 10., 10., 10., 50., 50.)[i]
        builder.joint_target_mode[d] = int(newton.JointTargetMode.POSITION_VELOCITY)
    pads = [next(i for i, label in enumerate(builder.shape_label) if label.endswith(suffix))
            for suffix in ("/tip_left_pad", "/tip_right_pad")]
    bare = builder.finalize(device="cuda:0")
    s = bare.state()
    newton.eval_fk(bare, bare.joint_q, bare.joint_qd, s)
    body_q = s.body_q.numpy()
    centres = np.asarray([wp.transform_point(wp.transform(*body_q[builder.shape_body[i]]), builder.shape_transform[i].p)
                          for i in pads])
    return builder, qidx, didx, target, pads, centres


def add_bag(builder, centres, args):
    cells = {"coarse": (12, 9, 2), "medium": (22, 16, 4), "fine": (33, 24, 6)}[args.mesh]
    v, f, edges, rest = closed_mesh(cells)
    if args.mesh != "coarse" and rest.max() >= PAD_WIDTH:
        raise ValueError("The mesh edge bound exceeds the narrow pad face")
    center = centres.mean(axis=0)
    close_axis = centres[1] - centres[0]
    close_axis[2] = 0.
    close_axis /= np.linalg.norm(close_axis)
    z = np.array([0., 0., 1.])
    length = np.cross(z, close_axis)
    # Horizontal 36 mm probe width, 220 mm length, 160 mm height.
    frame = np.column_stack((length, z, close_axis))
    rotation = wp.quat_from_matrix(wp.mat33(*frame.astype(np.float32).ravel()))
    # Probe a point 55 mm from the inner core's long-axis boundary.
    center += .055 * length
    center[2] -= args.lower_m
    lower_z = float(center[2] - BAG_SIZE[1] / 2)
    floor = builder.add_link(mass=0., lock_inertia=True, is_kinematic=True, label="Fixture/support")
    fixed = builder.add_joint_fixed(-1, floor)
    builder.add_articulation([fixed])
    support_cfg = newton.ModelBuilder.ShapeConfig(density=0., mu=.35, ke=800., kd=1.)
    support_xform = wp.transform(wp.vec3(float(center[0]), float(center[1]), lower_z - .0055), wp.quat_identity())
    # A collidable finite top is owned by MJWarp and copied into the VBD proxy view.
    builder.add_shape_box(floor, xform=support_xform, hx=.17, hy=.15, hz=.005,
                          cfg=support_cfg, label="Fixture/table")
    half = np.array([.100, .077, .0165])
    core_center = center + np.array([0., 0., -.0025])
    inertia = args.mass / 3. * (half @ half - half**2)
    core = builder.add_body(xform=wp.transform(wp.vec3(*core_center), rotation), mass=args.mass,
                            inertia=wp.mat33(*np.diag(inertia).astype(np.float32).ravel()), lock_inertia=True,
                            com=wp.vec3(args.offset_m, 0., 0.), label="Fixture/core")
    core_shape = builder.add_shape_box(core, hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
                                       cfg=newton.ModelBuilder.ShapeConfig(density=0., mu=args.friction,
                                                                          ke=800., kd=1.), label="Fixture/core_box")
    yy, thickness, poisson = (MATERIAL[k] for k in ("young_pa", "thickness_m", "poisson"))
    stretch = yy * thickness
    builder.add_cloth_mesh(pos=wp.vec3(*center), rot=rotation, scale=1., vel=wp.vec3(0.),
                           vertices=[wp.vec3(*row) for row in v], indices=f.ravel().tolist(),
                           density=MATERIAL["density_kg_m3"] * thickness, tri_ke=stretch / (2 * (1 + poisson)),
                           tri_ka=stretch * poisson / (1 - poisson**2), tri_kd=1.e-4,
                           edge_ke=MATERIAL["bend_nm"], edge_kd=1.e-9, particle_radius=RADIUS)
    return core, core_shape, edges, rest, f, center


def run(args):
    wp.config.enable_backward = False
    wp.set_device("cuda:0")
    b, qids, dids, desired, pads, centres = robot_scene()
    robot_shape_range = (0, len(b.shape_label) - 1)
    core, core_shape, edges, rest, faces, center = add_bag(b, centres, args)
    b.color(include_bending=True)
    model = b.finalize(device="cuda:0")
    model.soft_contact_ke = 800.
    model.soft_contact_kd = 1.
    model.soft_contact_mu = MATERIAL["friction"]
    model_rigids = list(range(model.body_count))
    # MJWarp owns rigid-rigid contacts. The cloth pass only builds rigid-soft contacts.
    rigid_pairs = wp.empty(0, dtype=wp.vec2i, device=model.device)
    cfg = CouplerProxyCfg(entries=[
        CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg(integrator="implicitfast", nconmax=512, njmax=2048),
                        bodies=[".*"], include_static_shapes=True),
        CouplerEntryCfg(name="cloth", solver_cfg=VBDSolverCfg(particle_enable_self_contact=True,
                        particle_self_contact_radius=RADIUS, particle_self_contact_margin=2 * RADIUS,
                        rigid_body_particle_contact_buffer_size=4096, iterations=10), all_particles=True),
    ], proxies=[CouplerProxyMappingCfg(source="rigid", destination="cloth", bodies=model_rigids, mode="lagged",
                collision_pipeline=lambda view: newton.CollisionPipeline(view, broad_phase="explicit",
                    shape_pairs_filtered=rigid_pairs, soft_contact_gap=RADIUS,
                    enable_rigid_soft_full_surface_contact=True),
                proxy_relaxation=args.feedback)])
    NewtonManager.backend = SimpleNamespace(model=model)
    NewtonManager._report_contacts = False
    NewtonCouplerManager._build_solver(model, cfg)
    solver = NewtonManager._solver
    rigid = solver.solver("rigid")
    contacts = solver.get_proxy_contacts("rigid", "cloth")
    cloth_view = solver.view("cloth")
    shape_labels = cloth_view.shape_label
    pad_shapes = [next(i for i, label in enumerate(shape_labels) if label.endswith(b.shape_label[pad]))
                  for pad in pads]
    core_contact = next(i for i, label in enumerate(shape_labels) if label.endswith(b.shape_label[core_shape]))
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver.reset(s0)
    solver.solver("cloth").rebuild_bvh(solver.entry_state("cloth"))
    edge_dev = wp.array(edges, dtype=wp.vec2i, device=model.device)
    rest_dev = wp.array(rest, dtype=float, device=model.device)
    maximum = wp.zeros(1, dtype=float, device=model.device)
    targets = ctrl.joint_target_q.numpy()
    motor_map = rigid.mjc_dof_to_newton_dof.numpy()[0]
    motor_dofs = [int(np.flatnonzero(motor_map == j)[0]) for j in dids]
    geom_shape = rigid.mjc_geom_to_newton_shape.numpy()[0]
    geom_dev = wp.array(geom_shape, dtype=int, device=model.device)
    direct_dev = wp.zeros(2, dtype=int, device=model.device)
    native = rigid.mjw_data
    print(f"Built {model.particle_count} bag vertices, pads {centres.tolist()}, bag {center.tolist()}", flush=True)

    rows = []
    stride = round(POLICY_DT / args.dt)
    sim_time = 0.
    start = time.perf_counter()
    for tick in range(args.ticks):
        t = tick * POLICY_DT
        target = desired.copy()
        target[-2:] += args.close_m * np.clip((t - .12) / .64, 0., 1.)
        # A slow joint-3/4 coordinated rock after the grip settles.
        phase = 2 * np.pi * max(t - .88, 0.) / 1.6
        target[3] += .012 * np.sin(phase)
        target[2] -= .006 * np.sin(phase)
        targets[qids] = target
        ctrl.joint_target_q.assign(targets)
        direct_dev.zero_()
        for _ in range(stride):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, args.dt)
            s0, s1 = s1, s0
            wp.launch(max_stretch, dim=len(edges), inputs=[s0.particle_q, edge_dev, rest_dev, maximum],
                      device=model.device)
            wp.launch(direct_core_contacts, dim=native.contact.geom.shape[0],
                      inputs=[native.nacon, native.contact.geom, geom_dev, native.contact.efc_address,
                              native.efc.force, core_shape, *robot_shape_range, direct_dev],
                      device=model.device)
        sim_time = (tick + 1) * POLICY_DT
        qpos = s0.joint_q.numpy()[qids]
        torque = rigid.mjw_data.qfrc_actuator.numpy()[0, motor_dofs]
        n = int(contacts.soft_contact_count.numpy()[0])
        rigid_in_cloth = int(contacts.rigid_contact_count.numpy()[0])
        idx = contacts.soft_contact_shape.numpy()[:n]
        soft_count = [int(np.count_nonzero(idx == i)) for i in (*pad_shapes, core_contact)]
        ids, counts = np.unique(idx, return_counts=True)
        counts_by_shape = {str(shape_labels[i]): int(c) for i, c in zip(ids, counts, strict=True) if i >= 0}
        wrenches = solver.entry_state("rigid", "input").body_f.numpy()
        pad_forces = wrenches[[b.shape_body[p] for p in pads]]
        overflow = int(np.bitwise_or.reduce(native.overflow.numpy())) if hasattr(native, "overflow") else None
        native_count = int(native.nacon.numpy()[0])
        geoms = native.contact.geom.numpy()[:native_count]
        shape_pairs = geom_shape[geoms[(geoms >= 0).all(axis=1)]]
        arm_pair = (shape_pairs == core_shape).any(axis=1) & (
            (shape_pairs >= robot_shape_range[0]) & (shape_pairs <= robot_shape_range[1])).any(axis=1)
        direct_shape_labels = sorted({b.shape_label[i] for pair in shape_pairs[arm_pair] for i in pair
                                      if robot_shape_range[0] <= i <= robot_shape_range[1]})
        direct_pad_core = int(np.count_nonzero((shape_pairs == core_shape).any(axis=1)
                                              & np.isin(shape_pairs, pads).any(axis=1)))
        direct_candidates, direct_loaded = [int(x) for x in direct_dev.numpy()]
        relevant = np.flatnonzero((geoms >= 0).all(axis=1))[arm_pair]
        native_direct_dist = native.contact.dist.numpy()[relevant]
        native_direct_addr = native.contact.efc_address.numpy()[relevant, 0]
        active_pairs = [
            dict(shapes=[b.shape_label[i] for i in pair], distance_m=float(d))
            for pair, d, addr in zip(shape_pairs[arm_pair], native_direct_dist, native_direct_addr, strict=True)
            if addr >= 0
        ]
        ratio = float(maximum.numpy()[0])
        rows.append(dict(t_s=sim_time, q_des=target.tolist(), q_act=qpos.tolist(),
                         error=(target - qpos).tolist(), motor_efforts=torque.tolist(), soft_counts=soft_count,
                         input_wrenches=pad_forces.tolist(), core_wrench=wrenches[core].tolist(),
                         core_pose=s0.body_q.numpy()[core].tolist(), stretch=ratio, native_overflow=overflow,
                         soft_contacts=n, rigid_contacts_in_cloth=rigid_in_cloth,
                         soft_counts_by_shape=counts_by_shape, native_rigid_contact_count=native_count,
                         native_direct_pad_core_count=direct_pad_core,
                         native_direct_arm_core_samples_per_tick=direct_candidates,
                         native_loaded_arm_core_samples_per_tick=direct_loaded,
                         native_direct_arm_core_dist_m=native_direct_dist.tolist(),
                         native_direct_arm_core_rows=native_direct_addr.tolist(),
                         native_direct_arm_core_constraint_pairs=active_pairs,
                         native_direct_arm_core_shapes=direct_shape_labels))
        finite = np.isfinite(qpos).all() and np.isfinite(torque).all() and np.isfinite(wrenches).all()
        if (ratio > 1.05 or not finite or overflow or native_count >= native.contact.geom.shape[0]
                or n >= contacts.soft_contact_max or rigid_in_cloth):
            break
        if tick % 12 == 0:
            print(f"t={sim_time:.2f} s counts={soft_count} stretch={ratio:.5f}", flush=True)
    wp.synchronize_device(model.device)
    wall = time.perf_counter() - start
    valid = (len(rows) == args.ticks and rows[-1]["stretch"] <= 1.05 and finite and not overflow
             and native_count < native.contact.geom.shape[0] and n < contacts.soft_contact_max and not rigid_in_cloth)
    return dict(valid=bool(valid), dt_s=args.dt, policy_dt_s=POLICY_DT, sim_time_s=sim_time, wall_s=wall,
                mass_kg=args.mass, friction=args.friction, offset_m=args.offset_m, feedback=args.feedback,
                fixture_lower_m=args.lower_m, closure_m=args.close_m,
                direct_arm_core_samples=sum(row["native_direct_arm_core_samples_per_tick"] for row in rows),
                loaded_arm_core_samples=sum(row["native_loaded_arm_core_samples_per_tick"] for row in rows),
                vertices=model.particle_count, faces=len(faces), edge_max_m=float(rest.max()),
                material=MATERIAL, pad_centers_m=centres.tolist(), bag_center_m=center.tolist(), rows=rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--headless", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=62)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--mass", type=float, default=.200)
    parser.add_argument("--offset-m", type=float, default=0.)
    parser.add_argument("--lower-m", type=float, default=.074)
    parser.add_argument("--close-m", type=float, default=.023)
    parser.add_argument("--friction", type=float, default=.35)
    parser.add_argument("--feedback", type=float, choices=(0., 1.), default=1.)
    parser.add_argument("--mesh", choices=("coarse", "medium", "fine"), default="medium")
    args = parser.parse_args()
    if args.output.exists() or args.dt <= 0. or args.ticks < 1 or args.mass <= 0.:
        parser.error("Require fresh output, positive limits and positive mass")
    if not np.isclose(POLICY_DT / args.dt, round(POLICY_DT / args.dt)):
        parser.error("The physics step must divide the 25 Hz policy period")
    result = run(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print({k: v for k, v in result.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    main()
