"""Contact-transfer gate. Everything written by this script stays in this folder.

Example (run with the existing bubble_mailer Newton interpreter):
  python -B benchmark.py --cases plate finger finger_off direct_plate membrane --tag base
  python -B benchmark.py --cases plate finger finger_off --dt .0005 --iterations 40 --tag refined
  python -B benchmark.py --cases plate --drive position --tool-z-mm 60.148 --dense-s .1 --tag pos
  python -B benchmark.py --cases membrane_finger --membrane bubble --dt .0001 --friction .5 --seconds 8 --checkpoints 2 --tag bag_a3

membrane_* cases put the guided tool on a free membrane lying on the slab (../bagged_contact/README.md, part A):
--ke/--kd are the tool-membrane contact and --membrane-ke/--membrane-kd the membrane-slab contact.

With --drive position the tool is a kinematic body moved along a prescribed height
profile, so it takes no part in the VBD solve. Comparing it with the force-driven tool
separates contact generation from rigid-soft coupling iteration.

The slab bottom is clamped. Gravity is disabled; a known force is ramped onto a
vertically guided rigid tool, a membrane, or a consistent nodal pressure load. This isolates
the two contact interfaces from ground contact, body weight and drive damping.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import warp as wp
import newton
from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact
from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_self_contact_force_and_hessian

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'contents' / 'newton'))
import unified_soft as us
from metrics import pressure_weights, surface_height, tet_response, transform_points

# Newton 1.7 added the rigid-soft log barrier as a trailing argument of the contact law.
HAS_LOG_BARRIER = 'use_log_barrier' in _eval_soft_ef_contact.input_types


@wp.kernel
def _load_particle(base: wp.array[wp.vec3], scale: wp.array[float], mass: wp.array[float],
                   vel: wp.array[wp.vec3], drag: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = scale[0] * base[i] - drag * mass[i] * vel[i]


@wp.kernel
def _load_body(base: wp.array[wp.spatial_vector], scale: wp.array[float], out: wp.array[wp.spatial_vector]):
    i = wp.tid()
    out[i] = scale[0] * base[i]


@wp.kernel
def _place_tool(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body: int,
                x: float, y: float, z_from: wp.array[float], z_to: wp.array[float], fraction: float, frame_dt: float):
    z = z_from[0] + fraction * (z_to[0] - z_from[0])
    body_q[body] = wp.transform(wp.vec3(x, y, z), wp.quat_identity())
    body_qd[body] = wp.spatial_vector(wp.vec3(0., 0., (z_to[0] - z_from[0]) / frame_dt), wp.vec3(0., 0., 0.))


@wp.kernel
def _contact_probe(
    count: wp.array[int], corners: wp.array[wp.vec3i], bary: wp.array[wp.vec3],
    q: wp.array[wp.vec3], prev: wp.array[wp.vec3], radius: wp.array[float],
    ke: wp.array[float], kd: wp.array[float], mu: wp.array[float], eps: float,
    shape_body: wp.array[int], body_q: wp.array[wp.transform], body_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
    shape: wp.array[int], cp: wp.array[wp.vec3], cv: wp.array[wp.vec3], normal: wp.array[wp.vec3],
    margin: wp.array[float], dt: float, use_log_barrier: bool,
    particle_force: wp.array[wp.vec3], shape_force: wp.array[wp.vec3], kinds: wp.array[int],
):
    k = wp.tid()
    if k >= count[0]:
        return
    ids, weights = corners[k], bary[k]
    if wp.static(HAS_LOG_BARRIER):
        force, hessian, point = _eval_soft_ef_contact(k, ids, weights, q, prev, radius, ke[k], kd[k], mu[k], eps,
            shape_body, body_q, body_prev, body_qd, com, shape, cp, cv, normal, margin, dt, use_log_barrier)
    else:
        force, hessian, point = _eval_soft_ef_contact(k, ids, weights, q, prev, radius, ke[k], kd[k], mu[k], eps,
            shape_body, body_q, body_prev, body_qd, com, shape, cp, cv, normal, margin, dt)
    wp.atomic_add(shape_force, shape[k], force)
    kind = int(0)
    for j in range(3):
        if ids[j] >= 0:
            wp.atomic_add(particle_force, ids[j], weights[j] * force)
            kind += 1
    wp.atomic_add(kinds, kind - 1, 1)


@dataclass
class Config:
    case: str = 'plate'
    dt: float = .002
    iterations: int = 10
    cells: tuple = (12, 10, 4)
    ke: float = 3000.
    kd: float = 3.
    friction: float = 0.
    full_surface: bool = True
    tile: bool = True
    seconds: float = 2.
    ramp_s: float = .2
    material_scale: float = 1.
    tag: str = 'base'
    k_start: float = 100.
    guided: bool = True
    drag: float = 0.
    sdf_voxel: float = .0015
    drive: str = 'force'  # 'position': kinematic tool on a prescribed height profile
    tool_z_mm: float = 0.  # final tool-centre height for position drive
    dense_s: float = 0.  # extra time at the end sampled after every substep
    log_barrier: bool = False  # Newton >= 1.7: rigid-soft normal contact uses the C2 log barrier
    friction_epsilon: float = 1e-2  # VBD slip-speed scale (m/s) below which friction is regularized
    friction_zero_fix: bool = False
    collision_iterations: bool = False  # Newton 1.7: re-detect contacts inside the solve
    planar_quadrature: bool = False  # flat-tool diagnostic: integrate over the covered top triangles
    membrane: str = 'bench'  # membrane_* cases: 'bench' sheet or 'bubble' (bag_sim/profiles/bubble.json)
    membrane_ke: float = 300.  # membrane_* cases: particle (membrane-slab) contact; ke/kd stay tool-membrane
    membrane_kd: float = .3
    checkpoints: tuple = ()  # extra end times (s) whose last-0.5 s window is summarized separately


# Rigid-particle contact uses the mean of shape and particle materials (VBD
# init_body_particle_contacts), so membrane_* tools carry 2*ke - membrane_ke.
MEMBRANES = {
    # spacing m, height m, tri_ke, tri_ka, tri_kd, edge_ke, areal density kg/m^2, radius m, self-contact distance m
    'bench': dict(spacing=.02, z=.059, tri_ke=5000., tri_ka=2000., tri_kd=.01, edge_ke=.01,
                  density=.012 / (.24 * .20), radius=.004, distance=.008),
}


def bubble_membrane():
    """bubble.json E*t, nu, areal density, radius; softened as in bag_sim/newton_bag.py.

    Bending uses K / spacing; the bag line's per-hinge stiffness and crease softening are not reproduced.
    """
    et, nu, density, spacing = 2e7 * .002, .3, .012 / .0864, .01
    wave = lambda e_t: spacing * np.sqrt(density / (e_t / (2 * (1 + nu))))
    scale = min(1., (.7 * wave(et) / 5e-5) ** 2)  # newton_substep_courant, newton_min_substep_s
    et *= scale
    distance = 2 * .00175  # newton_self_contact_scale * collision_radius_m
    return dict(spacing=spacing, z=.05 + distance + .0002, tri_ke=et / (2 * (1 + nu)), tri_ka=et * nu / (1 - nu * nu),
                tri_kd=.004, edge_ke=1.33e-4 / spacing, density=density, radius=.00175, distance=distance,
                softening=scale)


MEMBRANES['bubble'] = bubble_membrane()


def triangle_areas(q, faces):
    p = q[faces]
    return .5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def membrane_mesh(spacing=.02, z=.059):
    nx, ny = round(.24 / spacing), round(.20 / spacing)
    q = np.array([(x, y, z) for x in np.linspace(-.12, .12, nx + 1)
                  for y in np.linspace(-.10, .10, ny + 1)])
    faces = []
    for a in range(nx):
        for b in range(ny):
            i = a * (ny + 1) + b
            faces.extend([(i, i + ny + 1, i + 1), (i + 1, i + ny + 1, i + ny + 2)])
    return q, np.array(faces, np.int32)


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.friction_zero_fix:
            from contact_fixes import enable_friction_zero_fix
            enable_friction_zero_fix()
        direct = cfg.case.startswith('direct_')
        # membrane_<tool>: the guided tool presses a free membrane lying on the slab.
        self.layered = cfg.case.startswith('membrane_')
        kind = cfg.case.removeprefix('direct_').removeprefix('membrane_')
        self.membrane = kind == 'membrane' or self.layered
        sheet = MEMBRANES[cfg.membrane] if self.layered else MEMBRANES['bench']
        self.half = (.12, .10, .01) if kind in ('plate', 'membrane') else (.01, .01, .01)
        self.xy = (.0125, .0125) if kind == 'finger_off' else (0., 0.)
        self.load = 19.62 if kind in ('plate', 'membrane') else 2.
        builder = newton.ModelBuilder(gravity=wp.vec3(0., 0., 0.))
        shape = us.SoftShape('slab', (.30, .25, .05), tuple(cfg.cells), .3, layers=('core',) * cfg.cells[2],
                            materials={'core': (1500. * cfg.material_scale, 1000. * cfg.material_scale, 1.)})
        info = us.add_item(builder, shape, base_z=0.)
        self.n_tet_nodes = info['end']
        q_tet = np.asarray(builder.particle_q, dtype=float)
        self.fixed = np.flatnonzero(np.abs(q_tet[:, 2]) < 1e-7)
        for i in self.fixed:
            builder.particle_flags[i] &= ~int(newton.ParticleFlags.ACTIVE)
        all_faces = np.asarray(builder.tri_indices, dtype=int)
        self.top_faces = all_faces[np.all(np.abs(q_tet[all_faces, 2] - .05) < 1e-7, axis=1)]
        self.top_weights = pressure_weights(q_tet, self.top_faces, self.half, self.xy)
        self.body, self.tool_shape = None, None
        if self.membrane:
            cloth_q, cloth_f = membrane_mesh(sheet['spacing'], sheet['z'])
            start = builder.particle_count
            builder.add_cloth_mesh(pos=wp.vec3(0., 0., 0.), rot=wp.quat_identity(), scale=1., vel=wp.vec3(),
                vertices=cloth_q.tolist(), indices=cloth_f.ravel().tolist(), density=sheet['density'],
                tri_ke=sheet['tri_ke'], tri_ka=sheet['tri_ka'], tri_kd=sheet['tri_kd'], edge_ke=sheet['edge_ke'],
                edge_kd=1e-5, particle_radius=sheet['radius'])
            self.cloth_slice = slice(start, builder.particle_count)
            self.cloth_faces = cloth_f + start
            self.sheet = sheet
        if self.layered or not (direct or self.membrane):
            self.position = cfg.drive == 'position'
            # A tool on a membrane starts 0.2 mm above the membrane's contact surface.
            self.tool_z0 = (sheet['z'] + sheet['radius'] + .0002 if self.layered else .055) + self.half[2]
            start = wp.transform(wp.vec3(*self.xy, self.tool_z0), wp.quat_identity())
            if self.position:
                self.body = builder.add_body(xform=start, is_kinematic=True, label=kind)
            else:
                self.body = (builder.add_link if cfg.guided else builder.add_body)(xform=start, label=kind)
            ke, kd = (2 * cfg.ke - cfg.membrane_ke, 2 * cfg.kd - cfg.membrane_kd) if self.layered else (cfg.ke, cfg.kd)
            scfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, mu=cfg.friction,
                density=0. if self.position else float((self.load / 9.81) / (8 * np.prod(self.half))))
            if cfg.full_surface:
                scfg.configure_sdf(force_sdf=True, target_voxel_size=cfg.sdf_voxel)
            self.tool_shape = builder.add_shape_box(self.body, hx=self.half[0], hy=self.half[1], hz=self.half[2], cfg=scfg)
            if cfg.guided and not self.position:
                joint = builder.add_joint_prismatic(parent=-1, child=self.body, axis=wp.vec3(0., 0., 1.),
                    parent_xform=wp.transform(wp.vec3(*self.xy, self.tool_z0), wp.quat_identity()),
                    child_xform=wp.transform_identity(), target_ke=0., target_kd=0., damping=0., friction=0., armature=0.,
                    label='vertical_guide')
                builder.add_articulation([joint])
        builder.color(include_bending=True)
        self.model = m = builder.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = ((cfg.membrane_ke, cfg.membrane_kd) if self.layered
                                                                   else (cfg.ke, cfg.kd)) + (cfg.friction,)
        if cfg.log_barrier and not HAS_LOG_BARRIER:
            raise SystemExit(f'--log-barrier needs Newton >= 1.7, found {newton.__version__}')
        barrier = dict(rigid_soft_contact_use_log_barrier=True) if cfg.log_barrier else {}
        if cfg.planar_quadrature and (self.body is None or cfg.collision_iterations or not cfg.guided):
            raise ValueError('planar quadrature requires a flat vertically guided tool and external collision scheduling')
        quadrature = {}
        solver_type = newton.solvers.SolverVBD
        if cfg.planar_quadrature:
            from planar_contact import PlanarVBD
            solver_type = PlanarVBD
            quadrature['quadrature_weights'] = wp.zeros(len(self.top_faces) * 15, dtype=float)
        self.pipeline = newton.CollisionPipeline(m, broad_phase='nxn', soft_contact_gap=.004,
            enable_rigid_soft_full_surface_contact=cfg.full_surface,
            **({'soft_contact_max': len(self.top_faces) * 15} if cfg.planar_quadrature else {}),
            **({'contact_matching': 'latest'} if cfg.collision_iterations else {}))
        schedule = {}
        if cfg.collision_iterations:
            if not HAS_LOG_BARRIER:
                raise ValueError('in-iteration rigid collision detection requires Newton 1.7')
            base = newton.solvers.SolverBase
            schedule = dict(collision_pipeline=self.pipeline,
                collision_frequency={base.CollisionSlot.RIGID: 1},
                collision_frequency_type={base.CollisionSlot.RIGID: base.CollisionFrequencyType.ITERATIONS})
        self.solver = solver_type(m, iterations=cfg.iterations, particle_enable_tile_solve=cfg.tile, **barrier,
            **schedule, **quadrature,
            rigid_compliant_alm=True, rigid_contact_k_start=cfg.k_start, friction_epsilon=cfg.friction_epsilon,
            rigid_body_particle_contact_buffer_size=8192, rigid_body_contact_buffer_size=256,
            particle_enable_self_contact=self.membrane, particle_self_contact_margin=sheet['distance'],
            particle_self_contact_gap=sheet['distance'] / 2)
        self.contacts = self.solver.contacts if cfg.collision_iterations else self.pipeline.contacts()
        if cfg.planar_quadrature:
            from planar_contact import install_planar_contacts
            install_planar_contacts(self, quadrature['quadrature_weights'])
        self.s0, self.s1 = m.state(), m.state()
        self.control = m.control()
        self.scale = wp.zeros(1, dtype=float)
        self.q0 = m.particle_q.numpy().astype(float)
        self.tets = m.tet_indices.numpy()
        self.inv = m.tet_poses.numpy().astype(float)
        self.materials = m.tet_materials.numpy().astype(float)
        self.rest_energy = tet_response(self.q0, self.tets, self.inv, self.materials)[0]
        self.external = np.zeros((m.particle_count, 3), np.float32)
        if direct:
            self.external[:self.n_tet_nodes, 2] = -self.load * self.top_weights
        elif self.membrane and not self.layered:
            weights = pressure_weights(self.q0, self.cloth_faces, self.half, self.xy)
            self.external[:, 2] = -self.load * weights
        self.external_gpu = wp.array(self.external, dtype=wp.vec3)
        self.position = getattr(self, 'position', False)
        if self.position:
            self.z_from = wp.array([self.tool_z0], dtype=float)
            self.z_to = wp.array([self.tool_z0], dtype=float)
        body_load = np.zeros((m.body_count, 6), np.float32)
        if self.body is not None and not self.position:
            body_load[self.body, 2] = -self.load
        self.body_load = wp.array(body_load, dtype=wp.spatial_vector)
        self.contact_f = wp.zeros(m.particle_count, dtype=wp.vec3)
        self.self_f = wp.zeros_like(self.contact_f)
        self.self_h = wp.zeros(m.particle_count, dtype=wp.mat33)
        self.shape_f = wp.zeros(m.shape_count, dtype=wp.vec3)
        self.kinds = wp.zeros(3, dtype=int)
        self.body_prev_measure = wp.clone(m.body_q)
        # Even substep count keeps the two state buffers stable across CUDA graph launches.
        self.substeps = 2 * max(1, int(np.ceil(.01 / cfg.dt / 2)))
        self.dt = .01 / self.substeps
        self.graph = None

    def frame(self):
        for k in range(self.substeps):
            self.substep(k)

    def substep(self, k):
        if self.position:
            # VBD never moves a kinematic body, so its pose is written before every substep.
            wp.launch(_place_tool, dim=1, inputs=[self.s0.body_q, self.s0.body_qd, self.body, *map(float, self.xy),
                self.z_from, self.z_to, (k + 1) / self.substeps, self.dt * self.substeps])
        self.s0.clear_forces()
        wp.launch(_load_particle, dim=self.model.particle_count,
            inputs=[self.external_gpu, self.scale, self.model.particle_mass, self.s0.particle_qd, self.cfg.drag, self.s0.particle_f])
        if self.model.body_count:
            wp.launch(_load_body, dim=self.model.body_count, inputs=[self.body_load, self.scale, self.s0.body_f])
        if not self.cfg.collision_iterations:
            self.pipeline.collide(self.s0, self.contacts)
        if self.model.body_count:
            wp.copy(self.body_prev_measure, self.solver.body_q_prev)
        self.solver.step(self.s0, self.s1, self.control,
                         None if self.cfg.collision_iterations else self.contacts, self.dt)
        self.s0, self.s1 = self.s1, self.s0
        if self.position:
            # The solver does not write kinematic poses to the output state; keep the placed pose for measurement.
            wp.copy(self.s0.body_q, self.s1.body_q)

    def advance(self, frame):
        f = min(1., (frame + 1) * .01 / self.cfg.ramp_s)
        f = f * f * (3 - 2 * f)
        self.scale.assign(np.array([f], np.float32))
        if self.position:
            self.z_from.assign(self.z_to.numpy())
            self.z_to.assign(np.array([self.tool_z0 + f * (self.cfg.tool_z_mm * 1e-3 - self.tool_z0)], np.float32))
        if frame == 0 or not self.model.device.is_cuda:
            self.frame()
        else:
            if self.graph is None:
                with wp.ScopedCapture() as cap:
                    self.frame()
                self.graph = cap.graph
            wp.capture_launch(self.graph)

    def measure(self, t):
        m, s, c, v = self.model, self.s0, self.contacts, self.solver
        q = s.particle_q.numpy().astype(float)
        if not np.isfinite(q).all() or np.abs(q).max() > 2:
            raise FloatingPointError(f'diverged at {t}s')
        self.contact_f.zero_(); self.shape_f.zero_(); self.kinds.zero_(); self.self_f.zero_(); self.self_h.zero_()
        count = int(c.soft_contact_count.numpy()[0])
        if count > c.soft_contact_max:
            raise RuntimeError(f'contact buffer overflow {count} > {c.soft_contact_max}')
        if count:
            wp.launch(_contact_probe, dim=count, inputs=[c.soft_contact_count, c.soft_contact_indices, c.soft_contact_barycentric,
                s.particle_q, v.particle_q_prev, m.particle_radius, v.body_particle_contact_penalty_k,
                v.body_particle_contact_material_kd, v.body_particle_contact_material_mu, v.friction_epsilon,
                m.shape_body, s.body_q, self.body_prev_measure, s.body_qd, m.body_com, c.soft_contact_shape,
                c.soft_contact_body_pos, c.soft_contact_body_vel, c.soft_contact_normal, m.shape_margin, self.dt, self.cfg.log_barrier,
                self.contact_f, self.shape_f, self.kinds])
        if self.membrane:
            for color in range(len(m.particle_color_groups)):
                wp.launch(accumulate_self_contact_force_and_hessian, dim=v.particle_self_contact_evaluation_kernel_launch_size,
                    inputs=[self.dt, color, v.particle_q_prev, s.particle_q, m.particle_colors, m.tri_indices, m.edge_indices,
                        v.trimesh_collision_info, v.particle_self_contact_margin, m.soft_contact_ke, m.soft_contact_kd,
                        m.soft_contact_mu, v.friction_epsilon, v._self_contact_edge_edge_parallel_epsilon],
                    outputs=[self.self_f, self.self_h])
        cf = self.contact_f.numpy().astype(float) + self.self_f.numpy().astype(float)
        energy, ef, jac = tet_response(q, self.tets, self.inv, self.materials)
        fixed = self.fixed
        active = np.ones(self.n_tet_nodes, bool)
        active[fixed] = False
        ext = self.external * min(1., t / self.cfg.ramp_s)
        residual = ef[:self.n_tet_nodes] + cf[:self.n_tet_nodes] + ext[:self.n_tet_nodes]
        support = -float((ef[fixed] + cf[fixed] + ext[fixed]).sum(axis=0)[2])
        speed = np.linalg.norm(s.particle_qd.numpy(), axis=1)
        row = dict(t=round(t, 6), support_N=support, sink_mm=float(self.top_weights @ (.05 - q[:self.n_tet_nodes, 2]) * 1e3),
            slab_contact_N=-float(cf[:self.n_tet_nodes, 2].sum()), active_residual_L1_N=float(np.linalg.norm(residual[active], axis=1).sum()),
            active_residual_z_N=float(residual[active, 2].sum()), energy_J=energy-self.rest_energy, min_J=float(jac.min()),
            inverted_tets=int(np.sum(jac <= 0)), max_speed_mm_s=float(speed.max() * 1e3),
            median_active_speed_mm_s=float(np.median(speed[:self.n_tet_nodes][active]) * 1e3),
            nodes_over_1mm_s=int(np.sum(speed[:self.n_tet_nodes][active] > 1e-3)),
            soft_contacts=count, contact_kinds=self.kinds.numpy().tolist(),
            self_pair_force_imbalance_N=float(np.linalg.norm(self.self_f.numpy().sum(axis=0))))
        if self.layered:
            # Particle (self) contact on slab nodes is the membrane's push; rigid contact there bypasses it.
            self_f = self.self_f.numpy().astype(float)
            row.update(slab_max_speed_mm_s=float(speed[:self.n_tet_nodes].max() * 1e3),
                membrane_max_speed_mm_s=float(speed[self.cloth_slice].max() * 1e3),
                membrane_to_slab_N=-float(self_f[:self.n_tet_nodes, 2].sum()),
                tool_to_slab_N=-float(self.contact_f.numpy()[:self.n_tet_nodes, 2].sum()),
                membrane_max_area_stretch=float((triangle_areas(q, self.cloth_faces)
                                                 / triangle_areas(self.q0, self.cloth_faces)).max() - 1))
        if self.body is not None:
            pose = s.body_q.numpy()[self.body].astype(float)
            samples = np.array([(x, y, -self.half[2]) for x in np.linspace(-.99*self.half[0], .99*self.half[0], 7)
                                for y in np.linspace(-.99*self.half[1], .99*self.half[1], 7)])
            world = transform_points(samples, pose)
            h = surface_height(q, self.cloth_faces if self.layered else self.top_faces, world[:, :2])
            clearance = world[:, 2] - h
            if not np.isfinite(clearance).any():
                raise RuntimeError('tool footprint left the measurable top surface')
            row.update(tool_reaction_N=-float(self.shape_f.numpy()[self.tool_shape, 2]),
                tool_z_mm=float(pose[2] * 1e3), tool_speed_mm_s=float(np.linalg.norm(s.body_qd.numpy()[self.body, :3]) * 1e3),
                min_interface_clearance_mm=float(np.nanmin(clearance) * 1e3),
                tool_shift_mm=float(np.linalg.norm(pose[:2] - self.xy) * 1e3),
                tool_tilt_deg=float(np.rad2deg(2 * np.arccos(min(1., abs(pose[6]))))))
            counts = v.body_particle_contact_counts.numpy()
            if counts.max() > v.body_particle_contact_buffer_pre_alloc:
                raise RuntimeError('per-body contact buffer overflow')
        if self.membrane:
            cloth = q[self.cloth_slice]
            h = surface_height(q, self.top_faces, cloth[:, :2])
            if not np.isfinite(h).any():
                raise RuntimeError('membrane left the measurable top surface')
            row['min_membrane_clearance_mm' if self.layered else 'min_interface_clearance_mm'] = float(np.nanmin(cloth[:, 2] - h) * 1e3)
        return row, q


def summarize(exp, rows, end, completed):
    """Last-0.5 s means and acceptance checks for the samples up to `end` seconds."""
    late = [r for r in rows if r['t'] > (rows[-1]['t'] if rows else end) - .5]
    summary = {}
    if late:
        for key in late[0]:
            if isinstance(late[0][key], (int, float)):
                summary[key] = float(np.mean([r[key] for r in late]))
        summary['support_error_pct'] = abs(summary['support_N'] / exp.load - 1) * 100
        summary['material_residual_pct'] = summary['active_residual_L1_N'] / exp.load * 100
        summary['sink_drift_last_half_second_mm'] = late[-1]['sink_mm'] - late[0]['sink_mm']
        if 'tool_reaction_N' in summary:
            summary['tool_force_error_pct'] = abs(summary['tool_reaction_N'] / exp.load - 1) * 100
        summary['min_J_all_samples'] = min(r['min_J'] for r in rows)
        summary['inverted_tets_all_samples'] = max(r['inverted_tets'] for r in rows)
        summary['minimum_interface_clearance_all_samples_mm'] = min((r.get('min_interface_clearance_mm', float('inf')) for r in rows), default=None)
        if not np.isfinite(summary['minimum_interface_clearance_all_samples_mm']):
            summary['minimum_interface_clearance_all_samples_mm'] = None
    if exp.layered and late:
        summary['membrane_transfer_error_pct'] = abs(summary['membrane_to_slab_N'] / exp.load - 1) * 100
        summary['minimum_membrane_clearance_all_samples_mm'] = min(r['min_membrane_clearance_mm'] for r in rows)
    if exp.position and late:
        # A prescribed tool has no target load: both sides of the interface must agree instead.
        reaction = max(abs(summary['tool_reaction_N']), 1e-9)
        summary['interface_balance_pct'] = abs(summary['support_N'] - summary['tool_reaction_N']) / reaction * 100
        summary['support_over_reference_load'] = summary['support_N'] / exp.load
        summary['material_residual_pct'] = summary['active_residual_L1_N'] / reaction * 100
    checks = dict(finished=completed and bool(late) and rows[-1]['t'] >= end - .001,
        support_balance=summary.get('interface_balance_pct' if exp.position else 'support_error_pct', 1e9) <= 5.,
        material_equilibrium=summary.get('material_residual_pct', 1e9) <= 5.,
        no_inversion=bool(rows) and summary.get('inverted_tets_all_samples', 1) == 0,
        settled=bool(late) and abs(summary.get('sink_drift_last_half_second_mm', 1e9)) <= .05
            and max(summary.get('max_speed_mm_s', 1e9), summary.get('tool_speed_mm_s', 0.)) <= 1.)
    if exp.body is not None and not exp.position:
        checks['tool_force_balance'] = summary.get('tool_force_error_pct', 1e9) <= 5.
    if exp.body is not None or exp.membrane:
        checks['no_geometric_penetration'] = summary.get('minimum_interface_clearance_all_samples_mm', -1e9) >= -.05
    if exp.layered:
        checks['membrane_transfer'] = summary.get('membrane_transfer_error_pct', 1e9) <= 5.
        checks['no_geometric_penetration'] &= summary.get('minimum_membrane_clearance_all_samples_mm', -1e9) >= -.05
    return summary, checks


def run(cfg, out):
    t0 = time.perf_counter()
    exp = Experiment(cfg)
    rows, frames = [], []
    failure = None
    for frame in range(round(cfg.seconds / .01)):
        exp.advance(frame)
        if (frame + 1) % 5 == 0:
            try:
                row, q = exp.measure((frame + 1) * .01)
                if frames:
                    row['max_speed_50ms_mm_s'] = float(np.linalg.norm(q - frames[-1], axis=1).max() / .05 * 1e3)
                rows.append(row)
                frames.append(q.astype(np.float32))
                if row['inverted_tets']:
                    failure = f"{row['inverted_tets']} inverted tetrahedra at {row['t']} s"
                    break
            except (FloatingPointError, RuntimeError) as err:
                failure = str(err)
                break
    dense = []
    if failure is None and cfg.dense_s > 0:
        if exp.position:
            exp.z_from.assign(exp.z_to.numpy())
        for k in range(round(cfg.dense_s / exp.dt)):
            exp.substep(k % exp.substeps)
            try:
                row, _ = exp.measure(cfg.seconds + (k + 1) * exp.dt)
            except (FloatingPointError, RuntimeError) as err:
                failure = f'dense window: {err}'
                break
            dense.append({key: row[key] for key in ('t', 'support_N', 'slab_contact_N', 'sink_mm', 'max_speed_mm_s',
                                                    'active_residual_L1_N', 'soft_contacts', 'contact_kinds', 'tool_reaction_N')
                          if key in row})
    wp.synchronize()
    summary, checks = summarize(exp, rows, cfg.seconds, failure is None)
    if dense:
        for key in ('support_N', 'tool_reaction_N', 'sink_mm'):
            if key in dense[0]:
                series = np.array([r[key] for r in dense])
                summary[f'dense_{key}_peak_to_peak'] = float(np.ptp(series))
                spectrum = np.abs(np.fft.rfft(series - series.mean()))
                freqs = np.fft.rfftfreq(len(series), exp.dt)
                summary[f'dense_{key}_dominant_hz'] = float(freqs[1:][spectrum[1:].argmax()]) if len(series) > 2 else None
        summary['dense_contact_count_range'] = [min(r['soft_contacts'] for r in dense), max(r['soft_contacts'] for r in dense)]
    checkpoints = {f'{end:g}s': dict(zip(('summary', 'checks'),
                   summarize(exp, [r for r in rows if r['t'] <= end + 1e-9], end, True)))
                   for end in cfg.checkpoints if end < cfg.seconds}
    result = dict(config=asdict(cfg), actual_dt_s=exp.dt, load_N=exp.load,
        nodes=exp.model.particle_count, tets=exp.model.tet_count, fixed_nodes=len(exp.fixed),
        environment=dict(python=platform.python_version(), newton=newton.__version__, warp=wp.__version__, device=str(exp.model.device),
            newton_path=str(Path(newton.__file__).parent), executable=sys.executable,
            solver_sha256=hashlib.sha256(Path(newton.__file__).parent.joinpath('_src/solvers/vbd/solver_vbd.py').read_bytes()).hexdigest()),
        seconds=cfg.seconds, wall_s=round(time.perf_counter()-t0, 3), failure=failure,
        summary=summary, checks=checks, passed=all(checks.values()), checkpoints=checkpoints, trace=rows, dense_trace=dense)
    source_names = ['benchmark.py']
    if cfg.friction_zero_fix:
        source_names.append('contact_fixes.py')
    if cfg.planar_quadrature:
        source_names.append('planar_contact.py')
    result['environment']['project_sources_sha256'] = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{cfg.tag}_{cfg.case}'
    (out / f'{stem}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if frames:
        np.savez_compressed(out / f'{stem}.npz', frames=np.array(frames), top_faces=exp.top_faces,
                            tets=exp.tets, fixed_nodes=exp.fixed, initial=exp.q0,
                            cloth_faces=exp.cloth_faces if exp.membrane else np.empty((0, 3), int))
    print(json.dumps(dict(name=stem, summary=summary, checks=checks, wall_s=result['wall_s']), allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['plate', 'finger', 'finger_off', 'direct_plate', 'membrane'],
        choices=['plate', 'finger', 'finger_off', 'direct_plate', 'direct_finger', 'direct_finger_off', 'membrane',
                 'membrane_plate', 'membrane_finger', 'membrane_finger_off'])
    parser.add_argument('--dt', type=float, default=.002)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--cells', type=int, nargs=3, default=(12, 10, 4))
    parser.add_argument('--ke', type=float, default=3000.)
    parser.add_argument('--kd', type=float, default=3.)
    parser.add_argument('--friction', type=float, default=0.)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--material-scale', type=float, default=1.)
    parser.add_argument('--k-start', type=float, default=100.)
    parser.add_argument('--vertex-only', action='store_true')
    parser.add_argument('--no-tile', action='store_true')
    parser.add_argument('--free-tool', action='store_true', help='diagnostic: remove vertical guide')
    parser.add_argument('--drag', type=float, default=0.)
    parser.add_argument('--sdf-voxel', type=float, default=.0015)
    parser.add_argument('--tag', default='base')
    parser.add_argument('--drive', choices=['force', 'position'], default='force')
    parser.add_argument('--tool-z-mm', type=float, default=0., help='final tool-centre height for --drive position')
    parser.add_argument('--dense-s', type=float, default=0., help='extra seconds sampled after every substep')
    parser.add_argument('--log-barrier', action='store_true', help='Newton >= 1.7: log-barrier rigid-soft contact')
    parser.add_argument('--friction-epsilon', type=float, default=1e-2, help='VBD friction regularization speed, m/s')
    parser.add_argument('--friction-zero-fix', action='store_true', help='project-local continuous zero-slip stiffness')
    parser.add_argument('--collision-iterations', action='store_true', help='Newton 1.7: re-detect rigid contacts each iteration')
    parser.add_argument('--planar-quadrature', action='store_true', help='area-weighted contact for the flat vertical tool benchmark')
    parser.add_argument('--membrane', choices=sorted(MEMBRANES), default='bench', help='sheet under the tool in membrane_* cases')
    parser.add_argument('--membrane-ke', type=float, default=300., help='membrane_* cases: membrane-slab contact ke')
    parser.add_argument('--membrane-kd', type=float, default=.3, help='membrane_* cases: membrane-slab contact kd')
    parser.add_argument('--checkpoints', type=float, nargs='*', default=(), help='also summarize the 0.5 s windows ending here')
    args = parser.parse_args()
    if args.drive == 'position' and args.tool_z_mm <= 0:
        parser.error('--drive position needs --tool-z-mm')
    if args.seconds < 1. or args.dt <= 0 or args.iterations < 1 or any(n < 1 for n in args.cells):
        parser.error('seconds >= 1, dt > 0, iterations >= 1, positive cell counts required')
    wp.config.quiet = True
    for case in args.cases:
        run(Config(case=case, dt=args.dt, iterations=args.iterations, cells=tuple(args.cells), ke=args.ke, kd=args.kd,
            friction=args.friction, full_surface=not args.vertex_only, tile=not args.no_tile,
            seconds=args.seconds, material_scale=args.material_scale, tag=args.tag, k_start=args.k_start,
            guided=not args.free_tool, drag=args.drag, sdf_voxel=args.sdf_voxel, drive=args.drive,
            tool_z_mm=args.tool_z_mm, dense_s=args.dense_s, log_barrier=args.log_barrier,
            friction_epsilon=args.friction_epsilon, friction_zero_fix=args.friction_zero_fix,
            collision_iterations=args.collision_iterations, planar_quadrature=args.planar_quadrature,
            membrane=args.membrane, membrane_ke=args.membrane_ke, membrane_kd=args.membrane_kd,
            checkpoints=tuple(args.checkpoints)), HERE / 'results')


if __name__ == '__main__':
    main()
