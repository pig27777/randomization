"""Stage 1 prerequisites: dynamic geometric containment on the ABC table."""

import argparse
import json
import platform
import time
from pathlib import Path

import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from newton_reference import (
    MATERIAL, POLICY_DT, RADIUS, CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg,
    MJWarpSolverCfg, NewtonCouplerManager, NewtonManager, SimpleNamespace, VBDSolverCfg,
    direct_core_contacts, newton, np, robot_scene, wp,
)
from bag_sim.wrap import triangle_box_intersections, winding_number

SAMPLES = {
    "1A": (.300, (.120, .100, .030), (-.070, 0.)),
    "1B": (.600, (.140, .100, .035), (0., 0.)),
    "1C-empty": (0., (.140, .100, .035), (0., 0.)),
    "1C-500": (.500, (.140, .100, .035), (0., 0.)),
    "1C-1000": (1., (.140, .100, .035), (0., 0.)),
}
BAG_XY = np.array([-.03, -.225])
BAG_SIZE = np.array([.390, .290])
CORNER_SIGNS = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1
KP = np.array([80., 80., 80., 10., 10., 10., 1000., 1000.])
KD = np.array([5., 5., 5., 1.5, 1.5, 1.5, 16.7, 16.7])
LIMIT = np.array([28., 28., 28., 10., 10., 10., 50., 50.])


def pouch_mesh(half, offset, loaded, fold=False):
    nx, ny = 25, 19
    xy = np.stack(np.meshgrid(np.linspace(-.195, .195, nx + 1),
                              np.linspace(-.145, .145, ny + 1)), axis=-1)
    z = RADIUS + .00015
    bottom = np.full((ny + 1, nx + 1), z)
    sx = np.minimum(np.arange(nx + 1), np.arange(nx + 1)[::-1])
    sy = np.minimum(np.arange(ny + 1), np.arange(ny + 1)[::-1])
    rim = np.minimum(sy[:, None], sx)
    top = bottom + np.minimum(rim, 1) * .0015
    if loaded:
        # Keep full triangles over the payload outside the box at t=0.
        step = BAG_SIZE / [nx, ny]
        d = np.abs(xy - offset) - half[:2] - step
        fade = np.clip(d / .042, 0, 1)
        top += 2 * half[2] * np.prod(1 - fade * fade * (3 - 2 * fade), axis=-1) * (rim > 0)
    if fold:
        ridge = np.maximum(1 - np.abs(xy[:, :, 0] - .1014) / .029, 0.)
        ridge *= np.clip((.105 - np.abs(xy[:, :, 1])) / .06, 0., 1.)
        bottom += .035 * ridge
        top += .035 * ridge
    vertices = np.dstack((xy + BAG_XY, bottom)).reshape(-1, 3).tolist()
    bottom_ids = np.arange(len(vertices)).reshape(bottom.shape)
    top_ids = bottom_ids.copy()
    for j, i in zip(*np.nonzero(rim), strict=True):
        top_ids[j, i] = len(vertices)
        vertices.append([*(xy[j, i] + BAG_XY), top[j, i]])
    faces = []
    for ids, flip in ((top_ids, False), (bottom_ids, True)):
        for j in range(ny):
            for i in range(nx):
                a, b, c, d = ids[j, i], ids[j, i + 1], ids[j + 1, i + 1], ids[j + 1, i]
                cross_corner = (i, j) in ((nx - 1, 0), (0, ny - 1))
                tris = ((a, b, d), (b, c, d)) if cross_corner else ((a, b, c), (a, c, d))
                faces.extend(tri[::-1] if flip else tri for tri in tris)
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    edges, uses = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                                    faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    if not np.all(uses == 2):
        raise ValueError("The seam must join exactly two triangles per edge")
    rest = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return vertices, faces, edges, rest


def geometry(film, faces, half, pose):
    if pose is None:
        return None
    center, quat = pose[:3], pose[3:]
    rotation = Rotation.from_quat(quat).as_matrix()
    corners = CORNER_SIGNS * half @ rotation.T + center
    inside = np.abs(winding_number(film, faces, corners)) >= .5
    hits = triangle_box_intersections(film, faces, half, center, rotation)
    depth = 0.
    if hits.any():
        low, high = 0., float(half.min())
        for _ in range(17):
            middle = (low + high) / 2
            if triangle_box_intersections(film, faces[hits], half - middle, center, rotation).any():
                low = middle
            else:
                high = middle
        depth = high
    return dict(intersecting_triangles=int(hits.sum()), outside_corners=int((~inside).sum()),
                depth_upper_m=depth, clearance_m=float(corners[:, 2].min()))



def grasp_targets(model, qidx, pad_geoms, ready, case, seconds, height_retry, through_approach):
    """Plan the same open-loop ABC trajectory at 25 Hz on the actual rigid asset."""
    data = mujoco.MjData(model)
    data.qpos[qidx] = ready
    mujoco.mj_kinematics(model, data)
    pad_ref = data.geom_xmat[pad_geoms[0]].reshape(3, 3).copy()
    joints = [int(np.flatnonzero(model.jnt_qposadr == adr)[0]) for adr in qidx[:6]]
    use_limits = model.jnt_limited[joints].astype(bool)
    ranges = model.jnt_range[joints]
    lo = np.where(use_limits, ranges[:, 0], -2 * np.pi) + 1.e-6
    hi = np.where(use_limits, ranges[:, 1], 2 * np.pi) - 1.e-6
    film_pinch = case != "1B"
    xy = BAG_XY + ([.1014, 0.] if film_pinch else [0., 0.])
    close_z = .017 if film_pinch else .020
    open_rail = -.039
    dwell, close_until, lift_until = .08, .40, 2.40
    if height_retry:
        close_z = .012 if film_pinch else .039
    if through_approach:
        dwell, close_until, lift_until = .88, 1.20, 3.20
    start_q = np.clip(ready[:6], lo, hi)

    def blend(t, start, end):
        return (1. - np.cos(np.pi * np.clip((t - start) / (end - start), 0., 1.))) / 2.

    targets = []
    for tick in range(round(seconds / POLICY_DT) + 1):
        t = tick * POLICY_DT
        closing = blend(t, dwell, close_until)
        lifting = blend(t, close_until, lift_until)
        rail = open_rail + (.001 - open_rail) * closing
        entry = (.075 - close_z) * (1. - blend(t, 0., .80)) if through_approach else 0.
        desired_p = np.r_[xy, close_z + entry + .190 * lifting]
        desired_r = Rotation.from_rotvec([0., -.24 * lifting, 0.]).as_matrix() @ pad_ref

        def error(q):
            data.qpos[qidx] = np.r_[q, rail, rail]
            mujoco.mj_kinematics(model, data)
            p = data.geom_xpos[pad_geoms].mean(0)
            r = data.geom_xmat[pad_geoms[0]].reshape(3, 3)
            return np.r_[8. * (p - desired_p), Rotation.from_matrix(r @ desired_r.T).as_rotvec()]

        result = least_squares(error, start_q, bounds=(lo, hi), max_nfev=75, gtol=1.e-9)
        pose_error = error(result.x)
        if np.linalg.norm(pose_error[:3]) / 8. > .003:
            raise RuntimeError(f"pad IK exceeds 3 mm at {t:.3f} s: {pose_error.tolist()}")
        start_q = result.x
        targets.append(np.r_[start_q, rail, rail])
    return np.asarray(targets)


def run(args, case):
    mass, size, offset = SAMPLES[case]
    if args.grasp and case == "1B":
        size = (.064, .140, .035)
    if args.grasp and case.startswith("1C") and mass:
        offset = SAMPLES["1A"][2]
    half = np.asarray(size) / 2.
    vertices, faces, edges, rest = pouch_mesh(half, offset, mass > 0., fold=args.grasp and case != "1B")
    b, qids, dids, target, pads, centres = robot_scene()
    arm_shapes = len(b.shape_label)
    for dof in dids:
        b.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
        b.joint_target_ke[dof] = b.joint_target_kd[dof] = 0.
    table = b.add_link(mass=0., lock_inertia=True, is_kinematic=True, label="ABC/table")
    b.add_articulation([b.add_joint_fixed(-1, table)])
    table_shape = b.add_shape_box(table, xform=wp.transform(wp.vec3(0., 0., -.005), wp.quat_identity()),
                    hx=.2975, hy=.65, hz=.005,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0., mu=.35, ke=800., kd=1.))
    core = core_shape = None
    if mass:
        pos = np.r_[BAG_XY + offset, half[2] + 2 * RADIUS + .00015]
        inertia = mass / 3. * (half @ half - half**2)
        core = b.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass,
                          inertia=wp.mat33(*np.diag(inertia).astype(np.float32).ravel()),
                          lock_inertia=True, label="Payload")
        core_shape = b.add_shape_box(core, hx=half[0], hy=half[1], hz=half[2],
                                     cfg=newton.ModelBuilder.ShapeConfig(density=0., mu=.35, ke=800., kd=1.),
                                     label="Payload/box")
    tri = vertices[faces].astype(float)
    film_area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    areal_density = MATERIAL["density_kg_m3"] * MATERIAL["thickness_m"]
    if args.film_mass_g is not None:
        areal_density = args.film_mass_g / (1000. * film_area)
    poisson = MATERIAL["poisson"]
    stretch = MATERIAL["young_pa"] * MATERIAL["thickness_m"]
    b.add_cloth_mesh(pos=wp.vec3(0.), rot=wp.quat_identity(), scale=1., vel=wp.vec3(0.),
                     vertices=[wp.vec3(*v) for v in vertices], indices=faces.ravel().tolist(),
                     density=areal_density,
                     tri_ke=stretch / (2 * (1 + poisson)), tri_ka=stretch * poisson / (1 - poisson**2),
                     tri_kd=1.e-4, edge_ke=MATERIAL["bend_nm"], edge_kd=1.e-9, particle_radius=RADIUS)
    film_mass = float(sum(b.particle_mass))
    if args.film_mass_g is not None and not np.isclose(film_mass, args.film_mass_g / 1000., rtol=1.e-5, atol=0.):
        raise RuntimeError(f"Film mass differs from prescribed value: {film_mass:g} kg")
    b.color(include_bending=True)
    model = b.finalize(device="cuda:0")
    model.soft_contact_ke = args.contact_ke
    model.soft_contact_kd = 1.
    model.soft_contact_mu = MATERIAL["friction"]
    shape_pairs = wp.empty(0, dtype=wp.vec2i, device=model.device)
    def cloth_collision_pipeline(view):
        return newton.CollisionPipeline(
            view, broad_phase="explicit", shape_pairs_filtered=shape_pairs,
            soft_contact_gap=RADIUS, enable_rigid_soft_full_surface_contact=True)

    cfg = CouplerProxyCfg(entries=[
        CouplerEntryCfg(name="rigid", solver_cfg=MJWarpSolverCfg(integrator="implicitfast", nconmax=512, njmax=2048),
                        bodies=[".*"], include_static_shapes=True),
        CouplerEntryCfg(name="cloth", solver_cfg=VBDSolverCfg(particle_enable_self_contact=True,
            particle_self_contact_radius=RADIUS, particle_self_contact_margin=2 * RADIUS,
            rigid_body_particle_contact_buffer_size=4096, iterations=10), all_particles=True),
    ], proxies=[CouplerProxyMappingCfg(source="rigid", destination="cloth", bodies=list(range(model.body_count)),
        mode="lagged", proxy_relaxation=1., collision_pipeline=cloth_collision_pipeline)])
    NewtonManager.backend = SimpleNamespace(model=model)
    NewtonManager._report_contacts = False
    NewtonCouplerManager._build_solver(model, cfg)
    solver = NewtonManager._solver
    rigid = solver.solver("rigid")
    contacts = solver.get_proxy_contacts("rigid", "cloth")
    native = rigid.mjw_data
    s0, s1, ctrl = model.state(), model.state(), model.control()
    # Use the MJWarp rigid model's identical CPU template for gravity only.
    gravity_model = rigid.mj_model
    gravity_state = mujoco.MjData(gravity_model)
    motor_map = rigid.mjc_dof_to_newton_dof.numpy()[0]
    motor_dofs = np.array([np.flatnonzero(motor_map == j)[0] for j in dids])
    qpos_addresses = gravity_model.jnt_qposadr[gravity_model.dof_jntid[motor_dofs]]
    geom_map = rigid.mjc_geom_to_newton_shape.numpy()[0]
    pad_geoms = np.array([np.flatnonzero(geom_map == p)[0] for p in pads])
    if args.grasp:
        targets = grasp_targets(gravity_model, qpos_addresses, pad_geoms, target, case, args.seconds,
                                args.height_retry, args.through_approach)
    else:
        targets = np.tile(target, (round(args.seconds / POLICY_DT) + 1, 1))
    arm_q = model.joint_q.numpy()
    arm_q[qids] = targets[0]
    s0.joint_q.assign(arm_q)
    newton.eval_fk(model, s0.joint_q, model.joint_qd, s0)
    reset_flags = newton.StateFlags.ALL & ~(newton.StateFlags.JOINT_Q | newton.StateFlags.JOINT_QD)
    solver.reset(s0, flags=reset_flags)
    if not np.allclose(s0.joint_q.numpy()[qids], targets[0], atol=1.e-6, rtol=0.):
        raise RuntimeError("Coupled reset did not preserve the planned initial joint coordinates")
    solver.solver("cloth").rebuild_bvh(solver.entry_state("cloth"))
    shape_map = wp.array(geom_map, dtype=int, device=model.device)
    cloth_labels = solver.view("cloth").shape_label
    cloth_pads = [cloth_labels.index(b.shape_label[p]) for p in pads]
    pad_bodies = [b.shape_body[p] for p in pads]
    pad_half = np.asarray([list(b.shape_scale[p]) for p in pads])
    pad_local = np.asarray([[*b.shape_transform[p].p, *b.shape_transform[p].q] for p in pads])
    jac = np.zeros((3, gravity_model.nv))
    jac_rot = np.zeros_like(jac)
    direct = wp.zeros(2, dtype=int, device=model.device)
    arm_table = wp.zeros(2, dtype=int, device=model.device)
    effort = ctrl.joint_f.numpy()
    print(f"{case} reset: {len(vertices)} vertices, {len(faces)} triangles, {args.dt:g} s dt", flush=True)

    rows, snapshots = [], []
    stride = round(POLICY_DT / args.dt)
    start = time.perf_counter()
    failure = None
    for step in range(round(args.seconds / args.dt) + 1):
        target = targets[step // stride]
        q = s0.joint_q.numpy()[qids]
        dq = s0.joint_qd.numpy()[dids]
        film = s0.particle_q.numpy()
        bodies = s0.body_q.numpy()
        pose = bodies[core] if core is not None else None
        ratios = np.linalg.norm(film[edges[:, 0]] - film[edges[:, 1]], axis=1) / rest
        finite = np.isfinite(film).all() and np.isfinite(bodies).all() and np.isfinite(q).all() and np.isfinite(dq).all()
        geo = geometry(film, faces, half, pose) if finite else None
        hits = direct.numpy().tolist() if core_shape is not None else [0, 0]
        overflow = int(np.bitwise_or.reduce(native.overflow.numpy())) if hasattr(native, "overflow") and step else 0
        soft_count = int(contacts.soft_contact_count.numpy()[0]) if step else 0
        native_count = int(native.nacon.numpy()[0]) if step else 0
        room = soft_count < contacts.soft_contact_max and native_count < native.contact.geom.shape[0]
        table_hits = arm_table.numpy().tolist() if args.grasp else [0, 0]
        row = dict(time_s=step * args.dt, q_des=target.tolist(), q_act=q.tolist(), dq=dq.tolist(),
                   error=(target - q).tolist(), applied_effort=effort[dids].tolist(),
                   max_edge_ratio=float(ratios.max()) if finite else None,
                   min_film_z_m=float(film[:, 2].min()) if finite else None,
                   core_pose=pose.tolist() if pose is not None and finite else None,
                   geometry=geo, direct_arm_core_candidates=hits[0], direct_arm_core_loaded=hits[1],
                   soft_contacts=soft_count, finite=bool(finite), native_overflow=overflow)
        if args.grasp:
            row.update(loaded_arm_table=table_hits[1], native_contact_count=native_count)
        if step and (step % stride == 0 or args.grasp and (geo and geo["intersecting_triangles"])):
            soft_shapes = contacts.soft_contact_shape.numpy()[:soft_count]
            wrench = solver.entry_state("rigid", "input").body_f.numpy()
            native_effort = native.qfrc_applied.numpy()[0, motor_dofs]
            row.update(native_applied_effort=native_effort.tolist(),
                       pad_soft_candidates=[int(np.sum(soft_shapes == p)) for p in cloth_pads],
                       pad_proxy_wrench=wrench[pad_bodies].tolist(),
                       core_proxy_wrench=wrench[core].tolist() if core is not None else None)
            if args.grasp:
                gravity_state.qpos[qpos_addresses] = q
                mujoco.mj_fwdPosition(gravity_model, gravity_state)
                mujoco.mj_rne(gravity_model, gravity_state, 0, gravity_state.qfrc_bias)
                pad_mid = gravity_state.geom_xpos[pad_geoms].mean(0)
                tool = int(gravity_model.body_parentid[gravity_model.geom_bodyid[pad_geoms[0]]])
                mujoco.mj_jac(gravity_model, gravity_state, jac, jac_rot, pad_mid, tool)
                jac_z = jac[2, motor_dofs[:6]]
                gravity_effort = gravity_state.qfrc_bias[motor_dofs[:6]]
                row.update(pad_mid_m=pad_mid.tolist(), vertical_jacobian_m=jac_z.tolist(),
                           vertical_load_est_N=float(jac_z @ (native_effort[:6] - gravity_effort)
                                                     / (jac_z @ jac_z)))
        rows.append(row)
        if not finite:
            failure = "nonfinite state"
        elif overflow or not room:
            failure = "contact capacity"
        elif hits[1]:
            failure = "direct arm–payload load"
        elif table_hits[1]:
            failure = "direct arm–table load"
        elif geo and (geo["intersecting_triangles"] or geo["outside_corners"]):
            failure = "film–payload geometry"
        elif case == "1A" and ratios.max() > 1.05:
            failure = "maximum edge ratio"
        if failure and step:
            native_effort = native.qfrc_applied.numpy()[0, motor_dofs]
            row["native_applied_effort"] = native_effort.tolist()
            if native_count:
                native_geom = native.contact.geom.numpy()[:native_count]
                addresses = native.contact.efc_address.numpy()[:native_count]
                forces = native.efc.force.numpy()[0]
                dist = native.contact.dist.numpy()[:native_count]
                pairs = []
                for pair, slots, distance in zip(native_geom, addresses, dist, strict=True):
                    if min(pair) < 0:
                        continue
                    shapes = geom_map[pair]
                    active = slots[slots >= 0]
                    if np.any(np.abs(forces[active]) > 1.e-6):
                        pairs.append(dict(shapes=[b.shape_label[i] for i in shapes], distance_m=float(distance)))
                row["loaded_native_pairs"] = pairs
        if step == 0 or step % stride == 0 or failure:
            snapshots.append((step, film.copy(), bodies.copy()))
        if failure:
            print(f"{case} stopped at {row['time_s']:.6f} s: {failure}", flush=True)
            break
        if step < round(args.seconds / args.dt):
            if args.grasp or step % stride == 0:
                gravity_state.qpos[qpos_addresses] = q
                mujoco.mj_fwdPosition(gravity_model, gravity_state)
                mujoco.mj_rne(gravity_model, gravity_state, 0, gravity_state.qfrc_bias)
                command = np.clip(KP * (target - q) - KD * dq + gravity_state.qfrc_bias[motor_dofs], -LIMIT, LIMIT)
                effort[dids] = command
                ctrl.joint_f.assign(effort)
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, args.dt)
            s0, s1 = s1, s0
            if args.grasp:
                arm_table.zero_()
                wp.launch(direct_core_contacts, dim=native.contact.geom.shape[0],
                          inputs=[native.nacon, native.contact.geom, shape_map, native.contact.efc_address,
                                  native.efc.force, table_shape, 0, arm_shapes - 1, arm_table], device=model.device)
            if core_shape is not None:
                direct.zero_()
                wp.launch(direct_core_contacts, dim=native.contact.geom.shape[0],
                          inputs=[native.nacon, native.contact.geom, shape_map, native.contact.efc_address,
                                  native.efc.force, core_shape, 0, arm_shapes - 1, direct], device=model.device)
    out = args.output / case
    wp.synchronize_device(model.device)
    np.savez_compressed(out.with_suffix(".npz"), steps=np.array([s[0] for s in snapshots]),
                        film=np.stack([s[1] for s in snapshots]), bodies=np.stack([s[2] for s in snapshots]),
                        faces=faces, half=half, core_id=-1 if core is None else core,
                        pad_body_ids=np.asarray(pad_bodies),
                        pad_local_centers=pad_local[:, :3], pad_local_rotation=pad_local[:, 3:],
                        pad_half_sizes=pad_half,
                        q_targets=targets)
    data = dict(case=case, mass_kg=mass, core_size_m=size if mass else None, initial_offset_xy_m=offset,
                bag_size_m=BAG_SIZE.tolist(), vertices=len(vertices), faces=len(faces),
                max_reference_edge_m=float(rest.max()), material=MATERIAL, soft_contact_ke=args.contact_ke,
                particle_film_mass_kg=film_mass, film_areal_density_kg_m2=float(areal_density),
                specified_film_mass_g=args.film_mass_g,
                dt_s=args.dt, control_dt_s=POLICY_DT, substeps=stride, vbd_iterations=10,
                pad_initial_centres_m=centres.tolist(), arm_base_m=[-.3475, -.3100, 0.],
                robot_q_target_initial_rad_m=targets[0].tolist(), requested_sim_s=args.seconds,
                trial="grasp" if args.grasp else "preflight", planned_reset_verified=True,
                height_retry=args.height_retry, through_approach=args.through_approach,
                lift_end_s=(3.20 if args.through_approach else 2.40) if args.grasp else None,
                wall_s=time.perf_counter() - start, failure=failure, machine=platform.node(), rows=rows)
    with out.with_suffix(".json").open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", required=True, action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", required=True, choices=SAMPLES, nargs="+")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--height-retry", action="store_true")
    parser.add_argument("--through-approach", action="store_true")
    parser.add_argument("--dt", type=float, default=.0005)
    parser.add_argument("--seconds", type=float, default=.12)
    parser.add_argument("--contact-ke", type=float, default=800.)
    parser.add_argument("--film-mass-g", type=float)
    args = parser.parse_args()
    if args.film_mass_g is not None and (not np.isfinite(args.film_mass_g) or args.film_mass_g <= 0.):
        parser.error("film mass must be finite and positive")
    if args.through_approach and (not args.height_retry or args.cases != ["1B"]):
        parser.error("through approach applies to the 1B height retry only")
    if args.height_retry and (not args.grasp or any(case.startswith("1C") for case in args.cases)):
        parser.error("height retry applies to 1A/1B grasp only")
    if args.dt <= 0 or not np.isclose(POLICY_DT / args.dt, round(POLICY_DT / args.dt)):
        parser.error("dt must divide the 25 Hz control period")
    if args.seconds < POLICY_DT or not np.isclose(args.seconds / args.dt, round(args.seconds / args.dt)):
        parser.error("duration must span at least one control period and be an integer multiple of dt")
    args.output.mkdir(parents=True, exist_ok=False)
    wp.config.enable_backward = False
    wp.set_device("cuda:0")
    for case in args.cases:
        run(args, case)


if __name__ == "__main__":
    main()
