"""Reusable empty-reference bag, with separately configured independent contents."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from appearance import apply_film_texture
from asset_mesh import tabletop_mesh
from contents import add_contents, normalize_contents, numbers, place_contents
from initialization import prepare_loaded_positions

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reusable_bag_material', ROOT / 'model.py')
material = importlib.util.module_from_spec(spec)
spec.loader.exec_module(material)
_ACTIVE_BAG = None


class BagAsset(material.MuJoCoBag):
    def __init__(self, contents=(), *, params=None, support_z=.045, origin_xy=(0., 0.)):
        global _ACTIVE_BAG
        if _ACTIVE_BAG is not None:
            raise RuntimeError('One live bag material model per process; close the previous bag first')
        self._closed = False
        self.contents = normalize_contents(contents)
        self.support_z = float(support_z)
        self.origin_xy = np.asarray(origin_xy, float)
        config = json.loads((ROOT / 'config.json').read_text())
        parameters = {**config['bag_params'], **(params or {})}
        if not np.isfinite(self.support_z) or self.origin_xy.shape != (2,) or not np.isfinite(self.origin_xy).all():
            raise ValueError('Support height and origin must be finite')
        place_contents(self.contents, self.support_z, parameters['radius_mm'] * .001, self.origin_xy)
        super().__init__(parameters, mesh_factory=tabletop_mesh)
        self.data.eq_active[:] = 0
        self._release_right()
        self.running = True
        self.content_body_ids = [self.model.body(item['body_name']).id for item in self.contents]
        self.content_geom_ids = [self.model.geom(item['geom_name']).id for item in self.contents]
        self.initial_content_positions = [self.data.xpos[i].copy() for i in self.content_body_ids]
        _ACTIVE_BAG = self

    def _xml(self):
        p = self.params
        self.material_reference = self.rest.astype(float).copy()
        self.reference_fingerprint = hashlib.sha256(self.material_reference.tobytes()).hexdigest()
        rest = self.material_reference
        radius, gap = p['radius_mm'] * .001, p['rest_gap_mm'] * .001
        self.rest = np.column_stack([
            rest[:, 0] + self.origin_xy[0], .065 + p['height_mm'] * .0005 - rest[:, 2] + self.origin_xy[1],
            self.support_z + radius + gap / 2 + .0001 + rest[:, 1]])
        root = ET.parse(ROOT / 'bag_tabletop.xml').getroot()
        root.set('model', 'reusable_empty_reference_bag')
        for node in list(root.findall('keyframe')):
            root.remove(node)
        root.find('option').attrib.update(timestep=str(p['timestep']), iterations=str(p['iterations']),
                                         tolerance=str(p['tolerance']), gravity='0 0 -9.81' if p['gravity'] else '0 0 0')
        world = root.find('worldbody')
        for node in list(world.findall('body')):
            if node.get('mocap') != 'true':
                world.remove(node)
        for node in list(world.findall('geom')):
            world.remove(node)
        ET.SubElement(world, 'geom', name='table', type='plane',
                      pos=numbers([0, 0, self.support_z]), size='.65 .4 .001',
                      rgba='.24 .29 .34 1', friction='.8 .005 .0001', condim='3',
                      priority='1', solref='.004 1', solimp='.99 .999 .0001')
        flex = world.find('flexcomp')
        flex.attrib.update(point=numbers(self.rest), element=' '.join(map(str, self.faces.ravel())),
                           radius=str(radius), mass=str(p['mass_g'] * .001))
        flex.find('elasticity').attrib.update(young=str(p['young_pa']), poisson=str(p['poisson']),
                                             thickness=str(p['thickness_mm'] * .001), damping=str(p['rayleigh']),
                                             elastic2d='stretch')
        for side, name in enumerate(('upper_contact', 'lower_contact')):
            contact = root.find(f"deformable/flex[@name='{name}']")
            vids = [self.grid[side, i, j] for j in range(p['ny'] + 1) for i in range(p['nx'] + 1)]
            n = (p['nx'] + 1) * (p['ny'] + 1)
            count = 2 * p['nx'] * p['ny']
            contact.attrib.update(radius=str(radius), body=' '.join(f'bag_{vid}' for vid in vids),
                                  vertex=' '.join(['0'] * (3 * len(vids))),
                                  element=' '.join(map(str, (self.faces[side * count:(side + 1) * count] - side * n).ravel())))
            contact.find('contact').set('margin', str(p['margin_mm'] * .001))
            contact.find('contact').attrib.update(solref='.004 1', solimp='.99 .999 .0001')
            # Keep full panels for self/inter-layer contact. Rigid contact is
            # partitioned below, avoiding MuJoCo's per-pair contact truncation.
            contact.find('contact').attrib.update(contype=str(8 if side == 0 else 16), conaffinity='24')
        deformable = root.find('deformable')
        for node in list(deformable.findall('flex')):
            if '_patch_' in node.get('name', ''):
                deformable.remove(node)
        panel_faces = 2 * p['nx'] * p['ny']
        for side, prefix in enumerate(('upper_contact', 'lower_contact')):
            for j0 in range(0, p['ny'], 3):
                for i0 in range(0, p['nx'], 2):
                    triangles = []
                    for j in range(j0, min(j0 + 3, p['ny'])):
                        for i in range(i0, min(i0 + 2, p['nx'])):
                            start = side * panel_faces + 2 * (j * p['nx'] + i)
                            triangles.extend(self.faces[start:start + 2])
                    triangles = np.asarray(triangles)
                    vids = np.unique(triangles)
                    indices = {int(vid): k for k, vid in enumerate(vids)}
                    patch = ET.SubElement(deformable, 'flex', name=f'{prefix}_patch_{i0}_{j0}',
                                          dim='2', group='5', radius=str(radius), rgba='0 0 0 0',
                                          body=' '.join(f'bag_{vid}' for vid in vids),
                                          vertex=' '.join(['0'] * (3 * len(vids))),
                                          element=' '.join(str(indices[int(v)]) for v in triangles.ravel()))
                    ET.SubElement(patch, 'contact', selfcollide='none', contype='0', conaffinity='3',
                                  margin=str(p['margin_mm'] * .001), condim='3', friction='.12 .005 .0001',
                                  solref='.004 1', solimp='.99 .999 .0001')
                    ET.SubElement(patch, 'edge', damping='1e-12')
        for i, vid in enumerate(self.grip_ids):
            world.find(f"body[@name='grip_{i}']").set('pos', numbers(self.rest[vid]))
            pin = root.find(f"equality/connect[@name='pin_{i}']")
            pin.set('body2', f'bag_{vid}')
            pin.set('active', 'false')
        assets = root.find('asset')
        if assets is None:
            assets = ET.SubElement(root, 'asset')
        # Assets inherited from an exported example are not part of the bag.
        for node in list(assets.findall('mesh')):
            if node.get('name', '').startswith('contents_mesh_'):
                assets.remove(node)
        add_contents(world, self.contents, assets)
        apply_film_texture(root, p['nx'], p['ny'], absolute_paths=True)
        ET.indent(root)
        return ET.tostring(root, encoding='unicode')

    def _initial_positions(self):
        initial, self.initialization_report = prepare_loaded_positions(
            self.rest, self.faces, self.grid, self.params, self.contents, self.support_z, self.origin_xy)
        if self.initialization_report['maximum_edge_strain'] > .05:
            raise ValueError('Loading preparation exceeds 5 percent edge strain; adjust contents or bag dimensions')
        return initial

    def report(self):
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_FLEX, i) for i in range(self.model.nflex)]
        contents = []
        for item, body_id, geom_id in zip(self.contents, self.content_body_ids, self.content_geom_ids):
            contacts = [c for c in self.data.contact[:self.data.ncon] if geom_id in c.geom and np.any(c.flex >= 0)]
            counts = {prefix: sum(sum(i in c.flex for c in contacts) for i, name in enumerate(names)
                                  if name.startswith(prefix)) for prefix in ('upper_contact', 'lower_contact')}
            contents.append(dict(name=item['name'], shape=item['shape'], mass_kg=item['mass_kg'],
                                 position_m=self.data.xpos[body_id].tolist(),
                                 bag_contacts=counts,
                                 minimum_contact_distance_m=min((float(c.dist) for c in contacts), default=None)))
        return dict(time_s=float(self.data.time), finite=bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()),
                    warnings=[int(w.number) for w in self.data.warning],
                    reference='empty two-panel bag; independent of contents', reference_fingerprint=self.reference_fingerprint,
                    thickness_mm=self.params['thickness_mm'], radius_mm=self.params['radius_mm'],
                    timestep_s=self.params['timestep'], material_params=dict(self.params),
                    initialization=self.initialization_report, contents=contents,
                    bag_bbox_min_m=self.q.min(axis=0).tolist(), bag_bbox_max_m=self.q.max(axis=0).tolist(),
                    bag_height_mm=float(np.ptp(self.q[:, 2]) * 1000),
                    maximum_vertex_speed_m_s=float(np.linalg.norm(self.data.qvel[self.dofs], axis=1).max()),
                    solver_iterations_last=int(self.data.solver_niter.max()),
                    material_callback_calls=self.callback_calls, insertion_simulated=False, calibrated=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        global _ACTIVE_BAG
        if not self._closed:
            super().close()
            if _ACTIVE_BAG is self:
                _ACTIVE_BAG = None
            self._closed = True
