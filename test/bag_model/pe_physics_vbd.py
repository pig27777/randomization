"""Newton 1.6.1 adapter for the standardized PE physics scenarios."""
import io
import json
import sys

import newton
import numpy as np
import warp as wp

from pe_physics import BASE, FIXTURES, PAD, PAIRS, ROOT, pair_name, parcel_config, prepared_envelope, yaw_quaternion
from bag_sim.bag_model import dihedrals, parameters

sys.path.insert(0, str(ROOT / "bag_sim"))
from newton_bag import NewtonBag, _world_drag, content_shape_config

from newton._src.geometry.kernels import triangle_closest_point
from newton._src.geometry.tri_mesh_collision import (
    TriMeshCollisionInfo, get_edge_colliding_edges_count, get_vertex_colliding_triangles_count,
)
from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_neo_hookean_membrane_force_hessian, evaluate_self_contact_force_norm,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact


@wp.kernel
def soft_reactions(
    dt: float, epsilon: float, pose: wp.array[wp.transform],
    previous_pose: wp.array[wp.transform], body_vel: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3], q: wp.array[wp.vec3], previous_q: wp.array[wp.vec3],
    radius: wp.array[float], margins: wp.array[float], shape_body: wp.array[int],
    shapes: wp.array[int], indices: wp.array[wp.vec3i], barycentric: wp.array[wp.vec3],
    positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3], normals: wp.array[wp.vec3],
    ke: wp.array[float], kd: wp.array[float], mu: wp.array[float],
    wrench: wp.array[wp.spatial_vector], normal: wp.array[float],
    depth: wp.array[float], active: wp.array[int],
):
    i = wp.tid()
    shape = shapes[i]
    if shape < 0:
        return
    body = shape_body[shape]
    if body < 0:
        return
    nodes = indices[i]
    if nodes[0] < 0:
        return
    bary = barycentric[i]
    force, _, point = _eval_soft_ef_contact(
        i, nodes, bary, q, previous_q, radius, ke[i], kd[i], mu[i], epsilon,
        shape_body, pose, previous_pose, body_vel, com, shapes, positions, velocities, normals, margins, dt,
    )
    magnitude = wp.dot(force, normals[i])
    if magnitude <= 0.0:
        return
    film = q[nodes[0]] * bary[0]
    skin = radius[nodes[0]]
    for k in range(1, 3):
        if nodes[k] >= 0:
            film += q[nodes[k]] * bary[k]
            skin = wp.max(skin, radius[nodes[k]])
    penetration = skin + margins[shapes[i]] - wp.dot(film - point, normals[i])
    torque = wp.cross(point - wp.transform_point(pose[body], com[body]), -force)
    wp.atomic_add(wrench, body, wp.spatial_vector(-force, torque))
    wp.atomic_add(normal, body, magnitude)
    wp.atomic_max(depth, body, penetration)
    wp.atomic_add(active, body, 1)


@wp.func
def add_self_pair(distance: float, thickness: float, ke: float,
                  result: wp.array[float], contacts: wp.array[int]):
    if 0.0 < distance and distance < thickness:
        force, _ = evaluate_self_contact_force_norm(distance, thickness, ke)
        wp.atomic_add(result, 0, -force)
        wp.atomic_max(result, 1, thickness - distance)
        wp.atomic_add(contacts, 0, 1)


@wp.kernel
def self_reactions(q: wp.array[wp.vec3], triangles: wp.array2d[int], edges: wp.array2d[int],
                   infos: wp.array[TriMeshCollisionInfo], thickness: float, stiffness: float,
                   epsilon: float, result: wp.array[float], contacts: wp.array[int]):
    i = wp.tid()
    info = infos[0]
    if i < info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        offset = info.vertex_colliding_triangles_offsets[i]
        for j in range(get_vertex_colliding_triangles_count(info, i)):
            t = info.vertex_colliding_triangles[2 * (offset + j) + 1]
            if t >= 0:
                nearest, barycentric, feature = triangle_closest_point(
                    q[triangles[t, 0]], q[triangles[t, 1]], q[triangles[t, 2]], q[i])
                add_self_pair(wp.length(q[i] - nearest), thickness, stiffness, result, contacts)
    if i < edges.shape[0]:
        offset = info.edge_colliding_edges_offsets[i]
        for j in range(get_edge_colliding_edges_count(info, i)):
            edge = info.edge_colliding_edges[2 * (offset + j) + 1]
            if edge > i:
                st = wp.closest_point_edge_edge(q[edges[i, 2]], q[edges[i, 3]],
                                                q[edges[edge, 2]], q[edges[edge, 3]], epsilon)
                add_self_pair(st[2], thickness, stiffness, result, contacts)


@wp.kernel
def clamp_membrane(positions: wp.array[wp.vec3], previous: wp.array[wp.vec3],
                   faces: wp.array2d[int], inverse: wp.array[wp.mat22], properties: wp.array2d[float],
                   areas: wp.array[float], clamps: wp.array[int], dt: float, result: wp.array[wp.vec3]):
    face, side = wp.tid()
    if clamps[faces[face, side]]:
        force, _ = evaluate_neo_hookean_membrane_force_hessian(
            face, side, positions, previous, faces, inverse[face], areas[face],
            properties[face, 0], properties[face, 1], properties[face, 2], dt,
        )
        wp.atomic_add(result, 0, -force)


def transform(position, quaternion=(1., 0., 0., 0.)):
    w, x, y, z = quaternion
    return wp.transform(wp.vec3(*position), wp.quat(x, y, z, w))


class VBDBagBackend:
    @classmethod
    def reset(cls, config, seed=0):
        return cls(config, seed)

    def __init__(self, config, seed):
        if newton.__version__ != "1.6.1":
            raise NotImplementedError("the VBD force readback is qualified for Newton 1.6.1")
        if not wp.is_cuda_available():
            raise NotImplementedError("CUDA is required for this VBD run")
        if config.dt < 5e-5:
            raise NotImplementedError("existing VBD PE model sets a 50 µs Float32 step floor")
        self.device = wp.get_device("cuda:0")
        wp.set_device(self.device)
        cfg = parcel_config(config)
        film = self.envelope = prepared_envelope(config, cfg)
        self.spec = config
        self.seed = seed
        file = io.BytesIO()
        np.savez(file, vertices=film["reference"], faces=film["faces"], panel_ids=film["panels"],
                 masses_kg=film["masses"], uv=np.zeros((len(film["reference"]), 2)))
        file.seek(0)
        params = json.loads((ROOT / "bag_sim/profiles/pe.json").read_text())["params"]
        params.update(membrane_model="elastic",
                      young_pa=config.vbd_extensional_scale * BASE.membrane / BASE.material_thickness,
                      membrane_thickness_m=BASE.material_thickness, bending_rigidity_nm=BASE.bending,
                      collision_radius_m=BASE.radius, membrane_damping=BASE.rayleigh,
                      precreased_panel_boundaries=True, precreased_stiffness_ratio=BASE.precreased_ratio,
                      gravity=bool(config.gravity[2]), self_contact=config.self_collision,
                      frame_dt_s=2 * config.dt, newton_substep_s=config.dt)
        params = parameters(params)

        def extra(builder, contact):
            bodies = {}
            shapes = {}
            defs = {
                "table": ((0., 0., -.01), "box", (.35, .35, .01), BASE.friction),
                "contact_probe": ((0., 0., 1.), "sphere", (.01,), config.tool_mu),
                "platen": ((0., 0., 2.), "box", (.20, .15, .005), config.tool_mu),
                "left_pad": ((0., 0., 3.), "box", (.003, *PAD / 2), config.tool_mu),
                "right_pad": ((0., 0., 4.), "box", (.003, *PAD / 2), config.tool_mu),
                "obstacle": ((0., 0., 5.), "box", (.025, .025, .015), config.tool_mu),
            }
            if not config.table:
                del defs["table"]
            if config.mass:
                defs["payload"] = (film["payload_origin"], "box", np.asarray(config.size) / 2, BASE.friction)
            for name, (center, shape, size, friction) in defs.items():
                center = config.initial_tools.get(name, center)
                kinematic = name != "payload"
                material = content_shape_config(contact, params, mu=friction, density=0., primitive=False)
                material.has_shape_collision = name in ("table", "payload")
                if name == "platen" and config.case.startswith(("B05-B", "B05-C")):
                    material.has_shape_collision = True
                mass, inertia = 0., wp.mat33(0.)
                if not kinematic:
                    mass = config.mass
                    squared = np.square(size)
                    inertia = wp.mat33(*np.diag(mass / 3 * (squared.sum() - squared)).ravel())
                orientation = yaw_quaternion(config.yaw) if name == "payload" else (1., 0., 0., 0.)
                bodies[name] = builder.add_body(xform=transform(center, orientation), mass=mass, inertia=inertia,
                                                is_kinematic=kinematic, lock_inertia=True, label=name)
                if shape == "sphere":
                    shapes[name] = builder.add_shape_sphere(bodies[name], radius=size[0], cfg=material)
                else:
                    shapes[name] = builder.add_shape_box(bodies[name], hx=size[0], hy=size[1], hz=size[2], cfg=material)
            return bodies, shapes

        self.bag = NewtonBag(file, params, support=False, initial_vertices=film["current"], extra=extra,
                             device=self.device)
        bag = self.bag
        if not np.isclose(bag.dt, config.dt, rtol=0, atol=1e-12):
            raise NotImplementedError(f"the existing VBD wave-step bound selects {bag.dt:.8g} s")
        self.bodies, self.shapes = bag.extra
        self.clamps = config.clamped_nodes(film)
        self.fixed = config.fixed_nodes(film)
        bag.fix(config.fixed_nodes(film))
        if "moving" in self.clamps:
            bag.fix(self.clamps["moving"])
        self.table_id = self.bodies.get("table", self.bodies.get("platen") if config.case.startswith(("B05-B", "B05-C")) else None)
        self.payload_id = self.bodies.get("payload")
        self.previous_poses = wp.clone(bag.s0.body_q)
        self.fixed_mask = wp.array(np.isin(np.arange(len(film["reference"])), self.fixed).astype(np.int32),
                                   dtype=int, device=self.device)
        size = bag.model.body_count
        self.wrenches = wp.zeros(size, dtype=wp.spatial_vector, device=self.device)
        self.normal = wp.zeros(size, dtype=float, device=self.device)
        self.depth = wp.zeros(size, dtype=float, device=self.device)
        self.active = wp.zeros(size, dtype=int, device=self.device)
        self.self_load = wp.zeros(2, dtype=float, device=self.device)
        self.self_active = wp.zeros(1, dtype=int, device=self.device)
        self.clamp_load = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.time = 0.
        self.frame = 0
        self.graphs = {}
        self.metadata = dict(newton=newton.__version__, warp=wp.__version__, device=self.device.name,
                             params=bag.params, geometry=film["metadata"], substep=bag.dt,
                             extensional_calibration_scale=config.vbd_extensional_scale,
                             wave_step_membrane_scale=float(bag.membrane_scale), shape_ke=float(bag.contact["shape_ke"]),
                             shape_kd=float(bag.contact["shape_kd"]),
                             rigid_contact="full-surface analytic primitives, no tool–payload rigid pair",
                             wrench_readback="native barycentric particle-side AVBD force law on retained records",
                             clamp_force="membrane component from the native VBD law")
        self.rest_angles, _ = dihedrals(film["reference"], bag.hinges, gradient=False)

    def step(self, action):
        bag = self.bag
        if action["tools"]:
            current = bag.s0.body_q.numpy()
            velocity = bag.s0.body_qd.numpy()
            for name, (position, derivative, q) in action["tools"].items():
                body = self.bodies[name]
                current[body] = [*position, q[1], q[2], q[3], q[0]]
                velocity[body] = [*derivative, 0., 0., 0.]
            bag.s0.body_q.assign(current)
            bag.s0.body_qd.assign(velocity)
        if action["clamps"]:
            q = bag.s0.particle_q.numpy()
            for label, delta in action["clamps"].items():
                ids = self.clamps[label]
                q[ids] = self.envelope["current"][ids] + delta
            bag.s0.particle_q.assign(q)
        nodes = self.spec.loaded_nodes(self.envelope)
        load = None
        if len(nodes):
            load = np.zeros_like(self.envelope["reference"], dtype=np.float32)
            load[nodes] = action["load"] / len(nodes)
        ramp = bag.params["newton_self_contact_ramp_s"]
        during_ramp = bag.self_contact and self.time < ramp
        if during_ramp:
            bag.solver.particle_self_contact_margin = bag.self_contact_margin * min(1., (self.time + bag.dt) / ramp)**2

        def integrate():
            wp.copy(self.previous_poses, bag.solver.body_q_prev if self.frame else bag.s0.body_q)
            bag.s0.clear_forces()
            if load is not None:
                bag.s0.particle_f.assign(load)
            wp.launch(_world_drag, dim=bag.model.particle_count,
                      inputs=[bag.s0.particle_qd, bag.model.particle_mass,
                              bag.params["velocity_damping_per_s"], bag.s0.particle_f])
            bag.pipeline.collide(bag.s0, bag.contacts)
            bag.solver.step(bag.s0, bag.s1, bag.control, bag.contacts, bag.dt)

        if self.frame and not during_ramp and load is None:
            key = self.frame % 2
            if key not in self.graphs:
                with wp.ScopedCapture() as capture:
                    integrate()
                self.graphs[key] = capture.graph
            wp.capture_launch(self.graphs[key])
        else:
            integrate()
        bag.s0, bag.s1 = bag.s1, bag.s0
        self.frame += 1
        self.time = self.frame * bag.dt

    @property
    def contact_time(self):
        return self.time

    def get_bag_state(self):
        return self.bag.positions, self.bag.velocities

    def get_payload_state(self):
        if self.payload_id is None:
            return np.empty(0), np.empty(0)
        pose = self.bag.s0.body_q.numpy()[self.payload_id]
        return pose[[0, 1, 2, 6, 3, 4, 5]], self.bag.s0.body_qd.numpy()[self.payload_id]

    def get_fixture_state(self):
        pose = self.bag.s0.body_q.numpy()[[self.bodies[name] for name in FIXTURES]]
        return pose[:, [0, 1, 2, 6, 3, 4, 5]]

    def get_contact_wrench(self):
        counts = np.zeros(len(PAIRS), dtype=int)
        normal = np.zeros(len(PAIRS))
        depth = np.zeros(len(PAIRS))
        wrench = np.zeros((len(FIXTURES) + 2, 6))
        clamp = np.zeros(3)
        bag = self.bag
        model, solver, c = bag.model, bag.solver, bag.contacts
        if not self.frame:
            return dict(wrench=wrench, pairs=counts, pair_normal=normal, pair_depth=depth, clamp_force=clamp)
        for array in (self.wrenches, self.normal, self.depth, self.active, self.self_load,
                      self.self_active, self.clamp_load):
            array.zero_()
        # The rigid adjacency list excludes kinematic bodies. Particle-side
        # records still act on the film there and carry the tool reaction.
        count = min(c.soft_contact_max, int(c.soft_contact_count.numpy()[0]))
        if count:
            wp.launch(soft_reactions, dim=count,
                      inputs=[bag.dt, solver.friction_epsilon, bag.s0.body_q, self.previous_poses,
                              bag.s0.body_qd, model.body_com, bag.s0.particle_q, solver.particle_q_prev,
                              model.particle_radius, model.shape_margin, model.shape_body, c.soft_contact_shape,
                              c.soft_contact_indices, c.soft_contact_barycentric, c.soft_contact_body_pos,
                              c.soft_contact_body_vel, c.soft_contact_normal, solver.body_particle_contact_penalty_k,
                              solver.body_particle_contact_material_kd, solver.body_particle_contact_material_mu,
                              self.wrenches, self.normal, self.depth, self.active])
        if bag.self_contact:
            wp.launch(self_reactions, dim=max(model.particle_count, model.edge_count),
                      inputs=[bag.s0.particle_q, model.tri_indices, model.edge_indices,
                              solver.trimesh_collision_info, solver.particle_self_contact_margin,
                              model.soft_contact_ke, solver._self_contact_edge_edge_parallel_epsilon,
                              self.self_load, self.self_active])
            normal[0], depth[0] = self.self_load.numpy()
            counts[0] = self.self_active.numpy()[0]
        if len(self.fixed) and self.spec.case.startswith("B02"):
            wp.launch(clamp_membrane, dim=(model.tri_count, 3),
                      inputs=[bag.s0.particle_q, solver.particle_q_prev, model.tri_indices, model.tri_poses,
                              model.tri_materials, model.tri_areas, self.fixed_mask, bag.dt, self.clamp_load])
            clamp = self.clamp_load.numpy()[0]
        reaction, soft_normal, soft_depth, soft_count = (array.numpy() for array in
                                                       (self.wrenches, self.normal, self.depth, self.active))
        labels = (*FIXTURES, "table", "payload")
        for label, body in self.bodies.items():
            row = labels.index(label)
            wrench[row] += reaction[body]
            name = pair_name(("film", label))
            if name is not None:
                index = PAIRS.index(name)
                counts[index], normal[index], depth[index] = soft_count[body], soft_normal[body], soft_depth[body]
        if self.payload_id is not None and self.table_id is not None:
            a, b, pa, pb, f, n = solver.collect_rigid_contact_forces(
                bag.s0.body_q, self.previous_poses, c, bag.dt)
            num = int(n.numpy()[0])
            forces = f.numpy()[:num]
            pos = bag.s0.body_q.numpy()[:, :3]
            index = PAIRS.index("payload_table")
            for body_a, body_b, point_a, point_b, force in zip(
                    a.numpy()[:num], b.numpy()[:num], pa.numpy()[:num], pb.numpy()[:num], forces):
                if not np.any(force):
                    continue
                for body, point, sign in ((body_a, point_a, -1), (body_b, point_b, 1)):
                    name = "table" if body == self.table_id else "payload"
                    row = labels.index(name)
                    wrench[row] += sign * np.r_[force, np.cross(point - pos[body], force)]
                counts[index] += 1
                normal[index] += np.linalg.norm(force)
        # The table wrench is on the table in both adapters.
        return dict(wrench=wrench, pairs=counts, pair_normal=normal, pair_depth=depth, clamp_force=clamp)

    def get_physics_diagnostics(self, positions):
        bag = self.bag
        solver, c = bag.solver, bag.contacts
        angles, _ = dihedrals(positions, bag.hinges, gradient=False)
        delta = np.arctan2(np.sin(angles - self.rest_angles), np.cos(angles - self.rest_angles))
        overflow = 0
        if self.frame:
            overflow = int(c.soft_contact_count.numpy()[0] > c.soft_contact_max)
            overflow += int(c.rigid_contact_count.numpy()[0] > c.rigid_contact_max)
            overflow += int(solver.body_particle_contact_overflow_max.numpy()[0] >
                            solver.body_particle_contact_buffer_pre_alloc)
            overflow += int(solver.body_body_contact_overflow_max.numpy()[0] > solver.body_body_contact_buffer_pre_alloc)
            if bag.self_contact:
                overflow += int(np.any(solver.trimesh_collision_detector.resize_flags.numpy()))
        return dict(bending_energy=float(np.sum(.5 * bag.stiffness * delta**2)),
                    solver_potential=np.nan, warnings=0, contact_overflow=overflow)
