"""Newton (VBD) backend of the unified bag model: same profiles, same material arrays.

Stage 1 of the migration. Implemented here:
  mass        identical lumped node masses (load_material)
  membrane    Neo-Hookean membrane from E*t and Poisson ratio. Newton has no hard
              edge constraints, so "inextensible" also uses E*t; "tension_only" is absent.
              E*t is reduced (membrane_scale) when the accurate substep would be below
              newton_min_substep_s
  bending     identical per-hinge stiffness and damping, written per edge as
              edge_ke = K/l and edge_kd = c/l because VBD multiplies both by the edge length
  thickness   particle radius = collision_radius_m; a linear penalty per particle,
              ke = modulus * mean vertex area, so pressure = modulus * depth on a uniform mesh.
              The modulus is cushion_soft_modulus_pa_per_m with the cushion on, otherwise
              newton_contact_modulus_pa_per_m. The cushion lock-up stiffening is not represented yet
  contact     VBD self contact covers every non-adjacent vertex-triangle and edge-edge pair,
              same-side folds included; the front/back panel groups are not needed. With
              newton_full_surface_contact, rigid shapes also contact bag edges and faces
  damping     velocity_damping_per_s as an explicit world drag through particle_f

  history     stage 2: bilinear knee, plasticity, Maxwell memory and crease softening are
              advanced after every frame and written back as per-edge rest angle and stiffness
              (bending_history.py); one frame of lag

Not yet: tension_only membranes, the cushion lock-up, grippers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from bag_model import HERE, load_material, parameters
from bending_history import BendingHistory



@wp.kernel
def _world_drag(qd: wp.array[wp.vec3], mass: wp.array[float], rate: float, f: wp.array[wp.vec3]):
    i = wp.tid()
    f[i] = f[i] - mass[i] * rate * qd[i]


class NewtonBag:
    """Owns a Newton model with one bag; extra(builder) may add bodies before finalize."""

    def __init__(self, geometry, params, *, name="bag", support=True, initial_vertices=None, extra=None, device=None):
        p = self.params = parameters(params)
        if p["membrane_model"] == "tension_only":
            raise NotImplementedError("tension_only membranes are not available in the Newton backend yet")
        self.name = name
        material = load_material(geometry, p, initial_vertices)
        initial = material.pop("initial")
        self.__dict__.update(material)
        count = len(self.rest)

        builder = newton.ModelBuilder(gravity=-9.81 if p["gravity"] else 0.0)
        vertex_area = np.zeros(count)
        np.add.at(vertex_area, self.faces.ravel(), np.repeat(self.face_areas / 3, 3))
        self.vertex_area = vertex_area
        modulus = p["cushion_soft_modulus_pa_per_m"] if p["cushion"] else p["newton_contact_modulus_pa_per_m"]
        # With full-surface contact, edge and face contacts add to the vertex contacts: a box on a
        # sheet sank 8.7x less than modulus*depth on both 10 and 20 mm meshes. Dividing ke by that
        # was not linear (deeper than the radius at 6 kPa, areal-density dependent), so the
        # stiffer, uncompensated law is kept; calibrate the modulus with this setting on.
        area, mass = float(vertex_area.mean()), float(self.masses_kg.mean())
        zeta = p["cushion_damping_ratio"] if p["cushion"] else 1.0
        ke = modulus * area
        kd = 2 * zeta * np.sqrt(ke * mass)
        self_modulus = p["newton_self_contact_modulus_pa_per_m"]
        self_ke = ke if self_modulus is None else self_modulus * area
        self_kd = kd if self_modulus is None else 2 * zeta * np.sqrt(self_ke * mass)
        if self_ke > 2 * ke:
            raise ValueError("newton_self_contact_modulus_pa_per_m may be at most twice the body contact modulus")
        # VBD averages the particle and shape materials: shapes carry 2*body - self so body contact
        # keeps ke while the particles' own value sets self contact.
        self.contact = dict(modulus_pa_per_m=modulus, ke=ke, kd=kd, self_ke=self_ke,
                            shape_ke=2 * ke - self_ke, shape_kd=max(0.0, 2 * kd - self_kd))
        if support:
            ground = newton.ModelBuilder.ShapeConfig(ke=self.contact["shape_ke"], kd=self.contact["shape_kd"],
                                                     mu=max(0.0, 2 * p["table_friction"] - p["friction"]))
            builder.add_ground_plane(cfg=ground)
        self.extra = extra(builder, self.contact) if extra is not None else None

        et = p["young_pa"] * p["membrane_thickness_m"]
        nu = p["poisson"]
        # Substep from the membrane wave step on the typical edge (see newton_substep_courant);
        # soften the membrane when that step would fall below the float32 floor.
        density = float(self.masses_kg.sum() / self.face_areas.sum())
        edge = float(np.median(self.rest_lengths))
        wave = lambda e_t: edge * np.sqrt(density / (e_t / (2 * (1 + nu))))
        self.membrane_scale = min(1.0, (p["newton_substep_courant"] * wave(et) / p["newton_min_substep_s"]) ** 2)
        et *= self.membrane_scale
        self.wave_step = float(wave(et))
        start = builder.particle_count
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0),
                               vertices=[wp.vec3(*v) for v in self.rest], indices=self.faces.ravel().tolist(),
                               density=1.0, tri_ke=et / (2 * (1 + nu)), tri_ka=et * nu / (1 - nu * nu),
                               tri_kd=p["membrane_damping"], edge_ke=0.0, edge_kd=0.0,
                               particle_radius=p["collision_radius_m"])
        if builder.particle_count - start != count:
            raise RuntimeError("Newton changed the vertex count of the bag mesh")
        self.start = start
        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        model = self.model
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = self_ke, self_kd, p["friction"]

        masses = model.particle_mass.numpy()
        masses[start:start + count] = self.masses_kg
        model.particle_mass.assign(masses)
        inverse = model.particle_inv_mass.numpy()
        inverse[start:start + count] = 1.0 / self.masses_kg
        model.particle_inv_mass.assign(inverse)

        # Bending: map each unified hinge onto Newton's edge list.
        edges = model.edge_indices.numpy()
        lengths = model.edge_rest_length.numpy()
        lookup = {tuple(sorted((int(a) - start, int(b) - start))): e for e, (_, _, a, b) in enumerate(edges)}
        props = np.zeros((len(edges), 2), dtype=np.float32)
        self.edge_of_hinge = np.array([lookup[(int(a), int(b))] for a, b in self.hinges[:, :2]])
        props[self.edge_of_hinge, 0] = self.stiffness / lengths[self.edge_of_hinge]
        props[self.edge_of_hinge, 1] = self.hinge_damping / lengths[self.edge_of_hinge]
        model.edge_bending_properties.assign(props)
        self.edge_props = props
        self.edge_rest = model.edge_rest_angle.numpy().copy()
        local = edges[self.edge_of_hinge].astype(int) - start
        self.history = BendingHistory(p, self.stiffness, self.yield_angles, self.knee, local,
                                      self.edge_rest[self.edge_of_hinge], lengths[self.edge_of_hinge])

        radius = p["collision_radius_m"]
        self.self_contact = p["self_contact"] and p["newton_self_contact"]
        self.self_contact_margin = p["newton_self_contact_scale"] * radius
        self.solver = newton.solvers.SolverVBD(
            model, iterations=int(p["newton_iterations"]),
            # Seed body-particle penalties at the material stiffness instead of ramping from 100 N/m.
            rigid_contact_k_start=1e12,
            # A bag touches a rigid content or tool with hundreds of particles at once.
            rigid_body_particle_contact_buffer_size=4096, rigid_body_contact_buffer_size=256,
            particle_enable_self_contact=self.self_contact,
            particle_self_contact_margin=p["newton_self_contact_scale"] * radius,
            particle_self_contact_gap=0.5 * p["newton_self_contact_scale"] * radius)
        self.pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=radius,
                                                 enable_rigid_soft_full_surface_contact=p["newton_full_surface_contact"])
        self.s0, self.s1 = model.state(), model.state()
        self.control = model.control()
        self.contacts = self.pipeline.contacts()
        if initial is None:
            initial = self.rest.copy()
            if support:
                initial[:, 2] += max(0.0, radius + 0.00015 - initial[:, 2].min())
        q = self.s0.particle_q.numpy()
        q[start:start + count] = initial
        self.s0.particle_q.assign(q)
        self.s1.particle_q.assign(q)
        limit = max(min(p["newton_substep_s"], p["newton_substep_courant"] * self.wave_step), p["newton_min_substep_s"])
        # An even count lets a captured CUDA graph end on the same state buffers it started from.
        self.substeps = 2 * max(1, int(np.ceil(p["frame_dt_s"] / limit / 2)))
        self.dt = p["frame_dt_s"] / self.substeps
        if self.history.enabled:
            # Initial deformation (e.g. walls wrapped round a content) enters the history at once.
            self._write_history(initial, 0.0)
        self.graph = None
        self.use_graph = wp.get_device(model.device).is_cuda
        self.time = 0.0
        self.frames = 0

    @property
    def positions(self):
        return self.s0.particle_q.numpy()[self.start:self.start + len(self.rest)].astype(float)

    @property
    def velocities(self):
        return self.s0.particle_qd.numpy()[self.start:self.start + len(self.rest)].astype(float)

    def fix(self, vertex_ids):
        """Remove vertices from the simulation (clamps for fixtures and tests)."""
        flags = self.model.particle_flags.numpy()
        for v in np.asarray(vertex_ids) + self.start:
            flags[v] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)

    def _write_history(self, q, dt):
        rest, ke = self.history.update(q, dt)
        self.edge_rest[self.edge_of_hinge] = rest
        self.edge_props[self.edge_of_hinge, 0] = ke
        # Same device buffers, so a captured graph reads the new values on its next launch.
        self.model.edge_rest_angle.assign(self.edge_rest)
        self.model.edge_bending_properties.assign(self.edge_props)

    def _simulate_frame(self, ramp=None):
        rate = self.params["velocity_damping_per_s"]
        for k in range(self.substeps):
            if ramp is not None:
                # Quadratic start: pairs that begin almost touching meet the linear penalty first.
                fraction = min(1.0, (self.time + (k + 1) * self.dt) / ramp) ** 2
                self.solver.particle_self_contact_margin = self.self_contact_margin * fraction
            self.s0.clear_forces()
            if rate > 0:
                wp.launch(_world_drag, dim=self.model.particle_count,
                          inputs=[self.s0.particle_qd, self.model.particle_mass, rate, self.s0.particle_f])
            self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def step(self):
        """Advance one frame; on CUDA the frame's substeps replay as one captured graph."""
        ramp = self.params["newton_self_contact_ramp_s"]
        if self.self_contact and self.time < ramp:
            # Grow the self-contact distance; the graph would freeze it, so step directly.
            self._simulate_frame(ramp)
        elif self.use_graph:
            if self.graph is None:
                with wp.ScopedCapture() as capture:
                    self._simulate_frame()
                self.graph = capture.graph
            wp.capture_launch(self.graph)
        else:
            self._simulate_frame()
        self.time += self.substeps * self.dt
        self.frames += 1
        q = self.s0.particle_q.numpy()
        if not np.isfinite(q).all() or np.abs(q).max() > 20:
            raise FloatingPointError(f"Newton bag diverged at t={self.time:.3f} s")
        if self.history.enabled:
            self._write_history(q[self.start:self.start + len(self.rest)].astype(float), self.params["frame_dt_s"])

    def state_report(self):
        q = self.positions
        lengths = np.linalg.norm(q[self.edges[:, 1]] - q[self.edges[:, 0]], axis=1)
        strain = lengths / self.rest_lengths - 1
        return {"profile": self.name, "backend": f"Newton {newton.__version__} VBD", "time_s": round(self.time, 6),
                "mass_kg": float(self.masses_kg.sum()), "membrane_model": self.params["membrane_model"],
                "cushion": self.params["cushion"], "contact": self.contact, "finite": bool(np.isfinite(q).all()),
                "max_edge_stretch": float(strain.max()), "max_edge_compression": float(-strain.min()),
                "min_vertex_z_m": float(q[:, 2].min()), "max_vertex_z_m": float(q[:, 2].max()),
                "max_speed_m_s": float(np.linalg.norm(self.velocities, axis=1).max()),
                "soft_contacts": int(self.contacts.soft_contact_count.numpy()[0]),
                "substep_s": self.dt, "substeps_per_frame": self.substeps, "membrane_wave_step_s": self.wave_step,
                "membrane_scale": self.membrane_scale, "self_contact": self.self_contact,
                "cuda_graph": self.use_graph,
                "bending_history": self.history.report() if self.history.enabled else None,
                "physical_calibration": False}


def content_shape_config(contact, params, *, mu=0.6, density=None, primitive=True, voxel_m=0.0015):
    """ShapeConfig for a rigid body that touches the bag: its stiffness is chosen so that VBD's
    average with the particle value gives the bag's body-contact stiffness. With full-surface contact a primitive also gets the SDF that the edge
    and face passes sample; a mesh carries its own SDF instead (content_mesh)."""
    cfg = newton.ModelBuilder.ShapeConfig(ke=float(contact["shape_ke"]), kd=float(contact["shape_kd"]), mu=float(mu))
    if density is not None:
        cfg.density = float(density)
    if primitive and params["newton_full_surface_contact"]:
        cfg.configure_sdf(force_sdf=True, target_voxel_size=voxel_m)
    return cfg


def content_mesh(vertices, faces, params, *, voxel_m=0.0015):
    """newton.Mesh for a rigid content, with an SDF when full-surface contact is on."""
    mesh = newton.Mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.int32).ravel())
    if params["newton_full_surface_contact"]:
        mesh.build_sdf(target_voxel_size=voxel_m)
    return mesh


def load_profile_newton(name, *, params=None, **kwargs):
    """Load profiles/<name>.json into Newton; params override profile values for this instance only."""
    path = HERE / "profiles" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown profile {name!r}")
    profile = json.loads(path.read_text())
    return NewtonBag(HERE / profile["geometry"], {**profile["params"], **(params or {})}, name=name, **kwargs)
