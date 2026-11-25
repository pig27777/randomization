"""Replay the archived 10 g YAM targets with a reduced geometry audit."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

R1 = Path(__file__).resolve().parent.parent / "rss-premises-verify-r1"
sys.path.insert(0, str(R1))
from preflight import (
    BAG_SIZE, BAG_XY, CORNER_SIGNS, KD, KP, LIMIT, MATERIAL, POLICY_DT, RADIUS,
    SAMPLES, CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg,
    MJWarpSolverCfg, NewtonCouplerManager, NewtonManager, Rotation, SimpleNamespace,
    VBDSolverCfg, direct_core_contacts, mujoco, newton, np, pouch_mesh, robot_scene,
    triangle_box_intersections, winding_number, wp,
)
from bag_sim.wrap import point_triangle_distance
from newton_reference import max_stretch
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact

INSET = .002
LIGHT_JOINT_MASS = {"1C-150": .150, "1C-250": .250}
ARCHIVE_CASE = {"1C-150": "1C-500", "1C-250": "1C-1000"}


def geometry(film, faces, half, pose):
    q = Rotation.from_quat(pose[3:]).as_matrix()
    local = (film - pose[:3]) @ q
    all_tri = local[faces]
    bounding = ((all_tri.min(1) < half).all(1) & (all_tri.max(1) > -half).all(1))
    faces_near = faces[bounding]
    origin = np.zeros(3)
    nominal = triangle_box_intersections(local, faces_near, half, origin)
    hit_faces = faces_near[nominal]
    lo, hi = 0., 0.
    inset_count = 0
    if len(hit_faces):
        inset_count = int(triangle_box_intersections(local, hit_faces, half - INSET, origin).sum())
        hi = float(half.min())
        for _ in range(17):
            mid = (lo + hi) / 2
            if triangle_box_intersections(local, hit_faces, half - mid, origin).any():
                lo = mid
            else:
                hi = mid
    corner_local = CORNER_SIGNS * half
    corner_world = corner_local @ q.T + pose[:3]
    outside = np.abs(winding_number(local, faces, corner_local)) < .5
    distances = point_triangle_distance(corner_local[outside], local, faces)
    return dict(clearance_m=float(corner_world[:, 2].min()), nominal_triangles=int(len(hit_faces)),
                nominal_pen_um=hi * 1.e6, inset_triangles=inset_count,
                inset_pen_um=max(0., hi - INSET) * 1.e6 if inset_count else 0.,
                outside_corners=int(outside.sum()), outside_max_um=float(max(distances, default=0.) * 1.e6))


@wp.kernel
def body_health(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
                particle_v: wp.array(dtype=wp.vec3), failed: wp.array(dtype=int)):
    i = wp.tid()
    if i < q.shape[0]:
        for k in range(7):
            if not wp.isfinite(q[i][k]):
                wp.atomic_max(failed, 0, 1)
        for k in range(6):
            if not wp.isfinite(qd[i][k]):
                wp.atomic_max(failed, 0, 1)
    if i < particle_v.shape[0]:
        for k in range(3):
            if not wp.isfinite(particle_v[i][k]):
                wp.atomic_max(failed, 0, 1)


@wp.kernel
def pad_audit(
    pads: wp.array(dtype=int), dt: float, eps: float, count: wp.array(dtype=int),
    corners: wp.array(dtype=wp.vec3i), bary: wp.array(dtype=wp.vec3), shapes: wp.array(dtype=int),
    local_points: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3), particle_prev: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float), body_q: wp.array(dtype=wp.transform), body_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector), com: wp.array(dtype=wp.vec3),
    stiffness: wp.array(dtype=float), damping: wp.array(dtype=float), friction: wp.array(dtype=float),
    margin: wp.array(dtype=float), shape_body: wp.array(dtype=int),
    capacity: int, adj_count: wp.array(dtype=int), adjacency: wp.array(dtype=int),
    out_force: wp.array2d(dtype=float), marked: wp.array2d(dtype=int),
):
    p, lane = wp.tid()
    shape = pads[p]
    body = shape_body[shape]
    for slot in range(lane, wp.min(adj_count[body], capacity), 64):
        i = adjacency[body * capacity + slot]
        if i < count[0] and shapes[i] == shape and corners[i][0] >= 0:
            fp, _hess, _point = _eval_soft_ef_contact(
                i, corners[i], bary[i], particle_q, particle_prev, radius,
                stiffness[i], damping[i], friction[i], eps, shape_body, body_q, body_prev,
                body_qd, com, shapes, local_points, velocities, normal, margin, dt,
            )
            fn = wp.dot(fp, normal[i])
            if wp.abs(fn) > 1.e-7:
                ft = fp - fn * normal[i]
                wp.atomic_add(out_force, p, 0, wp.abs(fn))
                wp.atomic_add(out_force, p, 1, wp.length(ft))
                for k in range(3):
                    v = corners[i][k]
                    if v >= 0 and bary[i][k] > 1.e-7:
                        wp.atomic_max(marked, p, v, 1)


def run(args, case):
    args.through_approach = case == "1B"
    archived_case = ARCHIVE_CASE.get(case, case)
    mass, size, offset = SAMPLES[archived_case]
    mass = LIGHT_JOINT_MASS.get(case, mass)
    if case == "1B" and args.light_through:
        mass = .200
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
    source = R1 / ("through-10g" if case == "1B" else "joint-10g") / f"{archived_case}.npz"
    with np.load(source) as original:
        targets = original["q_targets"].copy()
        if not (np.array_equal(original["film"][0], vertices) and np.array_equal(original["faces"], faces)):
            raise RuntimeError("Restored geometry differs from the original trajectory")
    original_goals = targets.copy()
    if case == "1B":
        release = np.tile(targets[-1], (20, 1))
        release[:, 6:] += np.clip(np.arange(1, 21)[:, None] / 8, 0, 1) * (targets[0, 6:] - targets[-1, 6:])
        targets = np.concatenate((targets, release))
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

    vbd = solver.solver("cloth")
    pad_ids = wp.array(cloth_pads, dtype=int, device=model.device)
    shapes_n = 2
    pad_forces = wp.zeros((shapes_n, 2), dtype=float, device=model.device)
    marked = wp.zeros((shapes_n, model.particle_count), dtype=int, device=model.device)
    edge_dev = wp.array(edges, dtype=wp.vec2i, device=model.device)
    rest_dev = wp.array(rest, dtype=float, device=model.device)
    maximum = wp.zeros(1, dtype=float, device=model.device)
    invalid = wp.zeros(1, dtype=int, device=model.device)
    rows, snapshots = [], []
    stride = round(POLICY_DT / args.dt)
    n_steps = (len(targets) - 1) * stride
    audit_stride = args.audit_stride
    start = time.perf_counter()
    geometry_wall = 0.
    failure = None
    direct_total = 0
    candidate_total = 0
    table_total = 0
    next_progress = 1.
    for step in range(n_steps + 1):
        target = targets[step // stride]
        q = s0.joint_q.numpy()[qids]
        dq = s0.joint_qd.numpy()[dids]
        hits = direct.numpy().tolist() if core_shape is not None and step else [0, 0]
        table_hits = int(arm_table.numpy()[1]) if step else 0
        direct_total += hits[1]
        candidate_total += hits[0]
        table_total += table_hits
        bad = not (np.isfinite(q).all() and np.isfinite(dq).all())
        if step:
            bad = bad or bool(invalid.numpy()[0]) or maximum.numpy()[0] >= 1.e10
        if bad:
            failure = "nonfinite state"
        elif hits[1]:
            failure = "direct arm–payload load"
        elif table_hits:
            failure = "direct arm–table load"
        if step % audit_stride == 0 or step == n_steps or failure:
            audit_start = time.perf_counter()
            film = s0.particle_q.numpy()
            bodies = s0.body_q.numpy()
            pose = bodies[core] if core is not None else None
            geo = geometry(film, faces, half, pose) if pose is not None and not bad else None
            geometry_wall += time.perf_counter() - audit_start
            nsoft = int(contacts.soft_contact_count.numpy()[0]) if step else 0
            nnative = int(native.nacon.numpy()[0]) if step else 0
            overflow = bool(np.bitwise_or.reduce(native.overflow.numpy())) if hasattr(native, "overflow") and step else False
            if not failure and (nsoft >= contacts.soft_contact_max or nnative >= native.contact.geom.shape[0] or overflow):
                failure = "contact capacity"
            if not failure and geo and (geo["inset_triangles"] or geo["outside_max_um"] > 2000.):
                failure = "film–payload 2 mm geometry"
            row = dict(step=step, t_s=step * args.dt, finite=not bad,
                       q_des=target.tolist(), q_act=q.tolist() if not bad else None,
                       dq=dq.tolist() if not bad else None, error=(target - q).tolist() if not bad else None,
                       film_min_m=float(film[:, 2].min()) if not bad else None,
                       geometry=geo, direct_core_loaded=hits[1], direct_core_candidates=hits[0],
                       arm_table_loaded=table_hits, native_contacts=nnative, soft_candidates=nsoft,
                       max_edge_ratio=float(maximum.numpy()[0]) if not bad else None)
            if step and not bad and row["t_s"] >= ({"1B": .70}.get(case, .07)):
                vbd_state = solver.entry_state("cloth", "output")
                pad_forces.zero_()
                marked.zero_()
                wp.launch(pad_audit, dim=(2, 64), inputs=[
                    pad_ids, args.dt, vbd.friction_epsilon, contacts.soft_contact_count,
                    contacts.soft_contact_indices, contacts.soft_contact_barycentric, contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos, contacts.soft_contact_body_vel, contacts.soft_contact_normal,
                    vbd_state.particle_q, vbd.particle_q_prev, vbd.model.particle_radius,
                    vbd_state.body_q, vbd._coupling_body_q_prev_snapshot, vbd_state.body_qd, vbd.model.body_com,
                    vbd.body_particle_contact_penalty_k, vbd.body_particle_contact_material_kd,
                    vbd.body_particle_contact_material_mu, vbd.model.shape_margin, vbd.model.shape_body,
                    vbd.body_particle_contact_buffer_pre_alloc, vbd.body_particle_contact_counts,
                    vbd.body_particle_contact_indices, pad_forces, marked], device=model.device)
                row["pad_N_normal_tangent"] = pad_forces.numpy().tolist()
                row["pad_loaded_vertices"] = marked.numpy().sum(axis=1).tolist()
            if step and not bad and (step % stride == 0 or failure):
                native_effort = native.qfrc_applied.numpy()[0, motor_dofs]
                gravity_state.qpos[qpos_addresses] = q
                mujoco.mj_fwdPosition(gravity_model, gravity_state)
                mujoco.mj_rne(gravity_model, gravity_state, 0, gravity_state.qfrc_bias)
                pad_mid = gravity_state.geom_xpos[pad_geoms].mean(0)
                tool = int(gravity_model.body_parentid[gravity_model.geom_bodyid[pad_geoms[0]]])
                mujoco.mj_jac(gravity_model, gravity_state, jac, jac_rot, pad_mid, tool)
                jac_z = jac[2, motor_dofs[:6]]
                gravity_effort = gravity_state.qfrc_bias[motor_dofs[:6]]
                row.update(applied_effort=native_effort.tolist(), pad_mid_m=pad_mid.tolist(),
                           F_z_est_N=float(jac_z @ (native_effort[:6] - gravity_effort) / (jac_z @ jac_z)),
                           pad_proxy_wrench=solver.entry_state("rigid", "input").body_f.numpy()[pad_bodies].tolist())
            rows.append(row)
            if step % stride == 0 or step == n_steps or failure:
                snapshots.append((step, film.copy(), bodies.copy()))
        if step * args.dt >= next_progress:
            print(f"{case}: sim={step * args.dt:.3f}s wall={time.perf_counter()-start:.1f}s", flush=True)
            next_progress += 1.
        if failure or step == n_steps:
            print(f"{case}: {step * args.dt:.4f}s, {failure or 'complete'}", flush=True)
            break

        gravity_state.qpos[qpos_addresses] = q
        mujoco.mj_fwdPosition(gravity_model, gravity_state)
        mujoco.mj_rne(gravity_model, gravity_state, 0, gravity_state.qfrc_bias)
        command = np.clip(KP * (target - q) - KD * dq + gravity_state.qfrc_bias[motor_dofs], -LIMIT, LIMIT)
        effort[dids] = command
        ctrl.joint_f.assign(effort)
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, args.dt)
        s0, s1 = s1, s0
        arm_table.zero_()
        wp.launch(direct_core_contacts, dim=native.contact.geom.shape[0], inputs=[
            native.nacon, native.contact.geom, shape_map, native.contact.efc_address,
            native.efc.force, table_shape, 0, arm_shapes - 1, arm_table], device=model.device)
        if core_shape is not None:
            direct.zero_()
            wp.launch(direct_core_contacts, dim=native.contact.geom.shape[0], inputs=[
                native.nacon, native.contact.geom, shape_map, native.contact.efc_address,
                native.efc.force, core_shape, 0, arm_shapes - 1, direct], device=model.device)
        wp.launch(max_stretch, dim=len(edges), inputs=[s0.particle_q, edge_dev, rest_dev, maximum], device=model.device)
        wp.launch(body_health, dim=max(model.body_count, model.particle_count),
                  inputs=[s0.body_q, s0.body_qd, s0.particle_qd, invalid], device=model.device)

    wp.synchronize_device(model.device)
    wall = time.perf_counter() - start
    out = args.output / case
    np.savez_compressed(out.with_suffix(".npz"), steps=np.array([t for t, *_ in snapshots]),
                        film=np.stack([p for _, p, _ in snapshots]), bodies=np.stack([b for _, _, b in snapshots]),
                        faces=faces, half=half, core_id=-1 if core is None else core, pad_body_ids=pad_bodies,
                        pad_local_centers=pad_local[:, :3], pad_local_rotation=pad_local[:, 3:],
                        pad_half_sizes=pad_half, q_targets=targets, original_q_targets=original_goals)
    data = dict(case=case, machine=platform.node(), film_mass_g=film_mass * 1000., payload_mass_g=mass * 1000.,
                dt_s=args.dt, audit_stride=audit_stride, steps=step, wall_s=wall, geometry_wall_s=geometry_wall,
                requested_s=n_steps * args.dt, original_target_sha256=sha256(original_goals.tobytes()).hexdigest(),
                failure=failure, solid_direct_count=direct_total, solid_direct_candidate_count=candidate_total,
                arm_table_loaded_count=table_total, lift_end_s=3.2 if case == "1B" else 2.4,
                release_begin_s=(len(original_goals)-1)*POLICY_DT if case == "1B" else None,
                max_edge_ratio=float(maximum.numpy()[0]) if not bad else None,
                rows=rows)
    with out.with_suffix(".json").open("x") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=("1B", "1C-empty", "1C-500", "1C-1000", *LIGHT_JOINT_MASS), required=True)
    p.add_argument("--light-through", action="store_true", help="Use 200 g with the archived 1B geometry and targets")
    p.add_argument("--audit-stride", type=int, choices=(1, 10), default=10)
    args = p.parse_args()
    args.grasp, args.dt, args.film_mass_g, args.contact_ke = True, .0005, 10., 800.
    args.output.mkdir(parents=True, exist_ok=False)
    wp.config.enable_backward = False
    wp.set_device("cuda:0")
    for case in args.cases:
        run(args, case)


if __name__ == "__main__":
    main()
