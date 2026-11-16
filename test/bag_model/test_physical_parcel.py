"""Static and force-law regressions for the existing-envelope parcel."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import contents  # noqa: E402,F401  (package import must coexist with legacy helpers)
from bag_sim.physical_parcel import Config, Parcel, material_parameters, compression_force  # noqa: E402
from bag_sim.bag_model import dihedrals  # noqa: E402
from bag_sim.wrap import triangle_box_intersections, winding_number  # noqa: E402


class PhysicalParcelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parcel = Parcel()

    def test_existing_topology_and_positive_mass(self):
        p = self.parcel
        with np.load(ROOT / 'bag_sim/geometry/kraft_coarse_35.npz') as original:
            np.testing.assert_array_equal(p.envelope['faces'], original['faces'])
        self.assertEqual(p.m.nflexvert, 200)
        self.assertEqual(p.m.nflexelem, 396)
        self.assertEqual(p.m.neq, 0)
        self.assertTrue(np.all(p.envelope['masses'] > 0))

    def test_reference_geometry_is_independent(self):
        p = self.parcel
        self.assertGreater(np.max(abs(p.envelope['current']-p.envelope['reference'])), .001)
        rest = mujoco.MjData(p.m)
        p.forward(p.m, rest)
        self.assertLess(np.max(abs(rest.qfrc_passive)), 1e-7)

    def test_initial_triangles_enclose_payload(self):
        p = self.parcel
        e = p.envelope
        half = np.asarray(p.cfg.payload_size) / 2
        self.assertFalse(triangle_box_intersections(e['current'], e['faces'], half,
                                                   e['payload_origin']).any())
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
        winding = winding_number(e['current'], e['faces'], corners + e['payload_origin'])
        np.testing.assert_allclose(abs(winding), 1, atol=1e-10)
        self.assertLessEqual(e['metadata']['preparation']['maximum_edge_strain'], .05)

    def test_triangle_crossing_is_detected_with_exterior_vertices(self):
        vertices = np.array([[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]])
        faces = np.array([[0, 1, 2]])
        self.assertTrue(np.all(np.max(abs(vertices), axis=1) > 1))
        self.assertTrue(triangle_box_intersections(vertices, faces, np.ones(3), np.zeros(3))[0])
        self.assertFalse(triangle_box_intersections(vertices + [0, 0, 2], faces,
                                                    np.ones(3), np.zeros(3))[0])

    def test_forward_matches_step_reactions_without_advancing(self):
        p = Parcel()
        p.d.qpos[p.film_dofs[:, 2]] -= .0004
        before = p.d.qpos.copy()
        p.forward()
        support = p.contacts()['table_support_N']
        self.assertTrue(np.isfinite(support))
        self.assertGreater(support, 0)
        np.testing.assert_array_equal(p.d.qpos, before)
        self.assertEqual(p.d.time, 0)
        p.step()
        self.assertAlmostEqual(p.contacts()['table_support_N'], support, places=8)

    def test_payload_convex_shape_and_triangle_face_contact(self):
        p = self.parcel
        g = p.m.geom('payload_geom_0').id
        self.assertEqual(p.m.geom_type[g], mujoco.mjtGeom.mjGEOM_MESH)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        world = (signs * p.m.geom_size[g]) @ p.d.geom_xmat[g].reshape(3, 3).T + p.d.geom_xpos[g]
        np.testing.assert_allclose(world.max(0) - world.min(0), p.cfg.payload_size, atol=1e-8)
        root = ET.Element('mujoco')
        root.append(ET.fromstring(p.xml).find('asset'))
        body = ET.SubElement(root, 'worldbody')
        ET.SubElement(body, 'geom', type='mesh', mesh='payload_box_0', contype='8', conaffinity='2')
        triangle = ET.SubElement(body, 'flexcomp', name='crossing_triangle', type='direct', dim='2',
                                 point='-.08 -.06 .01 .08 -.06 .01 0 .08 .01',
                                 element='0 1 2', mass='.01', radius='.0005')
        ET.SubElement(triangle, 'contact', contype='2', conaffinity='8')
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertLess(min(contact.dist for contact in data.contact), -.005)

    def test_bending_internal_force_and_torque(self):
        p = self.parcel
        points = p.envelope['current']
        theta, gradients = dihedrals(points, p.hinges)
        delta = np.arctan2(np.sin(theta-p.rest_angles), np.cos(theta-p.rest_angles))
        forces = np.zeros_like(points)
        np.add.at(forces, p.hinges.ravel(), (-p.hinge_stiffness[:,None,None]*delta[:,None,None]*gradients).reshape(-1,3))
        self.assertLess(np.linalg.norm(forces.sum(0)), 1e-10)
        self.assertLess(np.linalg.norm(np.cross(points, forces).sum(0)), 1e-10)

    def test_unilateral_compression(self):
        cfg = Config()
        self.assertAlmostEqual(float(compression_force(.004, 0, cfg)), 3.52)
        self.assertEqual(float(compression_force(0, -1, cfg)), 0)
        self.assertGreater(float(compression_force(.005, 0, cfg)), float(compression_force(.004, 0, cfg)))

    def test_material_parameters_are_correlated(self):
        first = material_parameters(6e6, .00015, 920)
        second = material_parameters(6e6, .0003, 920)
        self.assertAlmostEqual(second['membrane']/first['membrane'], 2)
        self.assertAlmostEqual(second['areal_mass']/first['areal_mass'], 2)
        self.assertAlmostEqual(second['bending']/first['bending'], 8)

    def test_payload_mass_and_torque(self):
        p = Parcel(replace(Config(), mode='bendable'))
        self.assertAlmostEqual(float(p.m.body_mass[p.payload_bodies].sum()), .3)
        joint = p.m.joint('bend_1').id
        p.d.qpos[p.m.jnt_qposadr[joint]] = .1
        mujoco.mj_step1(p.m, p.d)
        self.assertAlmostEqual(float(p.d.qfrc_spring[p.m.jnt_dofadr[joint]]), -.008)

    def test_unresolved_bending_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not resolve'):
            Parcel(replace(Config(), bending=.00135, timestep=.0004))


if __name__ == '__main__':
    unittest.main()
