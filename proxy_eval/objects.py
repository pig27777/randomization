"""Proxy-real test objects X1-X12 (test/proxy_eval/REPORT.md), built into a Newton model with two pads.

Never used for training. Sizes and masses were chosen without reference to the reduced-model anchors
(contents/segment/model.py ANCHORS). Reused code is imported, not copied:
  tetrahedral items   contents/newton/unified_soft.py (generator, layered presets)
  pads                test/gripper_grasp/grasp.py rounded_box (4 mm fillet)
  bags                bag_sim/newton_bag.py NewtonBag and bag_sim/wrap.py wrap_generic,
                      scenes as in the bag line's round 1 C1 (test/bag_model/REPORT.md)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import warp as wp
import newton

ROOT = Path(__file__).resolve().parents[1]
BAG = ROOT / 'bag_sim'
for p in (ROOT / 'test' / 'gripper_grasp', ROOT / 'test' / 'contact_transfer', ROOT / 'contents' / 'newton', BAG, BAG / 'legacy'):
    sys.path.insert(0, str(p))
import unified_soft as us  # noqa: E402
from grasp import rounded_box  # noqa: E402
from task import PAD_FILLET, PAD_HALF  # noqa: E402

FIRM = (6000., 2000., 1.)   # Lame mu, lambda, damping: the hard sponge of test/gripper_grasp
SOFTER = (3000., 6000., 1.)  # did not invert at 14.5 N in the pilot; 3000/1500 and 4000/2000 did (REPORT.md, round 1)

OBJECTS = {
    'X1': dict(kind='sponge', size=(.12, .08, .05), mass=.30, material=FIRM, mu=.8, category='compressible'),
    'X2': dict(kind='sponge', size=(.18, .12, .07), mass=.70, material=FIRM, mu=.8, category='compressible'),
    'X3': dict(kind='sponge', size=(.12, .08, .05), mass=.30, material=SOFTER, mu=.8, category='compressible'),
    'X4': dict(kind='sponge', size=(.18, .12, .07), mass=.70, material=SOFTER, mu=.8, category='compressible'),
    'X5': dict(kind='layered', preset='folded_clothing', mu=.6, category='bendable'),
    'X6': dict(kind='layered', preset='pillow', mu=.6, category='compressible'),
    # C1 contents: full 200 x 150 x 30 mm 0.5 kg, loose 120 x 80 x 30 mm 0.3 kg; long side along the bag
    'X7': dict(kind='bag', profile='kraft', content='box', size=(.15, .20, .03), mass=.5, category='hard'),
    'X8': dict(kind='bag', profile='kraft', content='box', size=(.08, .12, .03), mass=.3, category='hard'),
    'X9': dict(kind='bag', profile='bubble', content='box', size=(.15, .20, .03), mass=.5, category='hard'),
    'X10': dict(kind='bag', profile='bubble', content='box', size=(.08, .12, .03), mass=.3, category='hard'),
    'X11': dict(kind='bag', profile='kraft', content='sponge', size=(.15, .20, .03), mass=.5, material=SOFTER,
                category='compressible'),
    'X12': dict(kind='bag', profile='kraft', content='sponge', size=(.08, .12, .03), mass=.3, material=SOFTER,
                category='compressible'),
}

# Tetrahedral-only scenes (contact values of test/gripper_grasp)
TET_SIM = dict(dt=.0005, iterations=10, ke=3000., kd=3., friction_eps=1e-2, contact=.004)
PARK = (0., 0., 1.)  # pads wait here until the gripper is placed
SPONGE_CELL = .02    # sponge lattice cell edge (m); 10 mm cells inverted under oblique pads (REPORT.md, round 1)


def _pad_mesh():
    mesh = rounded_box(PAD_HALF, PAD_FILLET)
    mesh.build_sdf(target_voxel_size=.0015)
    return mesh


def _add_pads(builder, cfg):
    mesh = _pad_mesh()
    bodies, shapes = [], []
    for k in range(2):
        body = builder.add_body(xform=wp.transform(wp.vec3(*PARK), wp.quat_identity()), is_kinematic=True, label=f'pad{k}')
        bodies.append(body)
        shapes.append(builder.add_shape_mesh(body, mesh=mesh, cfg=cfg))
    return bodies, shapes


class Scene:
    """Model, solver and state of one test object with two kinematic pads.

    particles: slice of the package's particles; bodies: rigid package bodies; bag: the NewtonBag or None."""

    def __init__(self, name):
        spec = OBJECTS[name]
        self.name, self.spec, self.category = name, spec, spec['category']
        self.bag = None
        self.bodies = []
        self.drag = None
        if spec['kind'] == 'bag':
            self._build_bag(spec)
        else:
            self._build_tet(spec)
        m = self.model
        self.tets = m.tet_indices.numpy() if m.tet_count else np.zeros((0, 4), int)
        self.inv_rest = m.tet_poses.numpy().astype(float) if m.tet_count else None
        mass = m.particle_mass.numpy().astype(float)
        self.particle_mass = mass[self.particles]
        self.body_mass = m.body_mass.numpy().astype(float)[self.bodies] if self.bodies else np.zeros(0)
        self.mass = float(self.particle_mass.sum() + self.body_mass.sum())

    def _build_tet(self, spec):
        s = TET_SIM
        builder = newton.ModelBuilder(gravity=wp.vec3(0., 0., -9.81))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=s['ke'], kd=s['kd'], mu=spec['mu']))
        if spec['kind'] == 'sponge':
            cells = tuple(max(1, round(x / SPONGE_CELL)) for x in spec['size'])
            shape = us.SoftShape('slab', spec['size'], cells, spec['mass'], layers=('core',) * cells[2],
                                 materials={'core': spec['material']})
        else:
            shape = us.PRESETS[spec['preset']]
        end = us.add_item(builder, shape, base_z=.002)['end']
        for i in range(end):
            builder.particle_radius[i] = s['contact']
        self.particles = slice(0, end)
        self.pads, self.pad_shapes = _add_pads(builder, newton.ModelBuilder.ShapeConfig(ke=s['ke'], kd=s['kd'], mu=spec['mu'],
                                                                                         density=0.))
        builder.color()
        self.model = m = builder.finalize()
        m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = s['ke'], s['kd'], spec['mu']
        self.solver = newton.solvers.SolverVBD(m, iterations=s['iterations'], rigid_compliant_alm=True,
            friction_epsilon=s['friction_eps'], rigid_body_particle_contact_buffer_size=8192, rigid_body_contact_buffer_size=256)
        self.pipeline = newton.CollisionPipeline(m, broad_phase='nxn', soft_contact_gap=s['contact'],
                                                 enable_rigid_soft_full_surface_contact=True)
        self.contacts = self.pipeline.contacts()
        self.s0, self.s1, self.control = m.state(), m.state(), m.control()
        self.substeps = 2 * max(1, int(np.ceil(.01 / s['dt'] / 2)))
        self.dt = .01 / self.substeps
        self.settle_s = .3

    def _build_bag(self, spec):
        from bag_model import parameters
        from contents import normalize_contents, place_contents
        from wrap import wrap_generic
        from newton_bag import content_shape_config, load_profile_newton
        profile = json.loads((BAG / 'profiles' / f"{spec['profile']}.json").read_text())
        p = profile['params']
        radius = p['collision_radius_m']
        with np.load(BAG / profile['geometry']) as g:
            rest, faces, panel_ids = np.array(g['vertices'], float), np.array(g['faces']), np.array(g['panel_ids'])
            start = np.array(g['initial_vertices'], float) if 'initial_vertices' in g else rest.copy()
        item = normalize_contents([dict(shape='box', size_m=list(spec['size']), mass_kg=spec['mass'])])
        if spec['profile'] == 'bubble':
            # The validated legacy loader for this two-panel topology (as in demo_contents.py).
            from asset_mesh import tabletop_mesh
            from initialization import prepare_loaded_positions
            gap = float(rest[:, 2].max() - rest[:, 2].min())
            _, _, grid, _ = tabletop_mesh(14, 18, .18, .24, gap, .015)
            place_contents(item, 0.0, radius, (0., 0.))
            initial, report = prepare_loaded_positions(rest + [0, 0, radius + .0001], faces, grid,
                dict(width_mm=180., height_mm=240., radius_mm=radius * 1e3, rest_gap_mm=gap * 1e3), item, 0.0, (0., 0.))
        else:
            initial, report = wrap_generic(rest, start, faces, panel_ids, p['self_contact_front_panel_ids'],
                                           p['self_contact_back_panel_ids'], item[0], radius)
        if report['maximum_edge_strain'] > .05:
            raise ValueError(f"wrapped bag strains edges by {report['maximum_edge_strain']:.3f}")
        center = item[0]['position']
        params = parameters(p)
        built = {}

        def extra(builder, contact):
            if spec['content'] == 'box':
                cfg = content_shape_config(contact, params, density=spec['mass'] / float(np.prod(spec['size'])))
                body = builder.add_body(xform=wp.transform(wp.vec3(*map(float, center)), wp.quat_identity()), label='content')
                half = np.asarray(spec['size']) / 2
                built['content_shape'] = builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
                built['bodies'] = [body]
            else:
                cells = tuple(max(1, round(x / SPONGE_CELL)) for x in spec['size'])
                shape = us.SoftShape('slab', spec['size'], cells, spec['mass'], layers=('core',) * cells[2],
                                     materials={'core': spec['material']})
                end = us.add_item(builder, shape, base_z=float(center[2] - spec['size'][2] / 2), xy=tuple(center[:2]))['end']
                # Contact with the bag only: a film-thin radius, as in test/bagged_contact/bagged_grasp.py
                for i in range(end):
                    builder.particle_radius[i] = radius
                built['sponge'] = end
                built['bodies'] = []
            # Pads touch the bag with its body-contact stiffness; friction is the bag's (geometric mean in VBD).
            pad_cfg = newton.ModelBuilder.ShapeConfig(ke=float(contact['shape_ke']), kd=float(contact['shape_kd']),
                                                      mu=float(p['friction']), density=0.)
            built['pads'] = _add_pads(builder, pad_cfg)
            if 'content_shape' in built:
                for s in built['pads'][1]:
                    builder.add_shape_collision_filter_pair(s, built['content_shape'])

        self.bag = bag = load_profile_newton(spec['profile'], initial_vertices=initial, extra=extra)
        self.bodies = built['bodies']
        self.pads, self.pad_shapes = built['pads']
        self.particles = slice(0, bag.start + len(bag.rest))
        self.bag_particles = slice(bag.start, bag.start + len(bag.rest))
        self.model, self.solver, self.pipeline, self.contacts = bag.model, bag.solver, bag.pipeline, bag.contacts
        self.control = bag.control
        self.substeps, self.dt = bag.substeps, bag.dt
        self.drag = bag.params['velocity_damping_per_s']
        self.settle_s = .5

    @property
    def s0(self):
        return self.bag.s0 if self.bag else self._s0

    @s0.setter
    def s0(self, v):
        if self.bag:
            self.bag.s0 = v
        else:
            self._s0 = v

    @property
    def s1(self):
        return self.bag.s1 if self.bag else self._s1

    @s1.setter
    def s1(self, v):
        if self.bag:
            self.bag.s1 = v
        else:
            self._s1 = v

    def place(self, yaw, shift):
        """Rigidly turn the package about its centre and shift it in xy (seeded placement)."""
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        q = self.s0.particle_q.numpy()
        pts = q[self.particles].astype(float)
        center = np.r_[(pts[:, :2].min(0) + pts[:, :2].max(0)) / 2, 0.]
        q[self.particles] = (pts - center) @ rot.T + center + np.r_[shift, 0.]
        self.s0.particle_q.assign(q)
        self.s1.particle_q.assign(q)
        if self.bodies:
            bq = self.s0.body_q.numpy()
            half = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
            for b in self.bodies:
                bq[b, :3] = rot @ (bq[b, :3] - center) + center + np.r_[shift, 0.]
                bq[b, 3:] = _quat_mul(half, bq[b, 3:])
            self.s0.body_q.assign(bq)
            self.s1.body_q.assign(bq)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])
