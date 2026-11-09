"""Guard against visually different targets, fake grasp attachments and thickened envelopes."""
import importlib.util
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'policy'))
import parcel_heap as heap
heap.install_backend(heap.RUN/'contact_fix')


class HeapContract(unittest.TestCase):
    def test_raised_heap_preserves_shell_and_content_geometry(self):
        low=heap.scene('book','gap')
        high=heap.scene('book','gap','side',settings=replace(heap.HeapSettings(),stack_lift=.17,
                                                          adaptive_close=True,every_step_safety=True))
        np.testing.assert_allclose(high.sim.shell_points()-low.sim.shell_points(),
                                   np.broadcast_to([0,0,.17],high.sim.shell_points().shape),atol=1e-10)
        np.testing.assert_allclose(high.sim.d.xipos[high.sim.content_bodies]-low.sim.d.xipos[low.sim.content_bodies],
                                   [[0,0,.17]],atol=1e-10)
        foundation=high.sim.m.body('heap_foundation').id
        self.assertEqual(high.sim.m.body_dofnum[foundation],6)
        self.assertAlmostEqual(high.sim.m.body_mass[foundation],3.)
        self.assertLessEqual(max((-c.dist for c in high.sim.d.contact),default=0.),1e-7)

    def test_per_step_guard_matches_native_contact_decoder(self):
        import json,mujoco
        path=heap.RUN/'contact_fix/contact.jsonl'
        row=next(json.loads(line) for line in path.read_text().splitlines()
                 if json.loads(line)['case']=='paper_envelope' and json.loads(line)['action']=='center')
        robot=heap.scene(row['case'],row['layout'])
        robot.sim.d.qpos[:]=row['keyframes'][-1]['qpos']
        mujoco.mj_forward(robot.sim.m,robot.sim.d)
        full=heap.contacts(robot);fast=heap.contact_safety(robot)
        self.assertGreater(fast['penetration_mm'],.15)
        for key,value in fast.items():self.assertAlmostEqual(value,full[key])

    def test_side_approaches_towards_package(self):
        np.testing.assert_array_equal(heap.SIDE_ROTATION[:,2],[0.,1.,0.])
        self.assertAlmostEqual(abs(heap.SIDE_ROTATION[2,1]),1.)
        self.assertAlmostEqual(np.linalg.det(heap.SIDE_ROTATION),1.)

    def test_same_shell_different_contents(self):
        fingerprints = {}
        for case in heap.CASES:
            robot = heap.scene(case,'top')
            sim = robot.sim
            family = 'envelope' if 'envelope' in case else 'bag'
            signature = (sim.info['shell_sha256'],sim.m.flex_radius[0],sim.m.flex_rgba[0].tolist())
            if family in fingerprints:
                self.assertEqual(signature,fingerprints[family])
            fingerprints[family] = signature
            self.assertEqual(heap.outside_count(sim.d.xipos[sim.content_bodies],
                             sim.shell_points(),sim.info['faces']),0)
            if family=='envelope':
                self.assertAlmostEqual(sim.m.flex_radius[0]*2,.00015)
            # The only equalities are content compaction welds and native YAM coupling.
            names = [sim.m.equality(i).name for i in range(sim.m.neq)]
            self.assertFalse(any('pinch' in name or 'bag' in name for name in names))

    def test_initial_heap_has_no_intersections(self):
        for layout in heap.LAYOUTS:
            robot = heap.scene('book',layout)
            self.assertLessEqual(max((-c.dist for c in robot.sim.d.contact),default=0.),1e-7)
            sim = robot.sim
            self.assertTrue(all(sim.m.body_dofnum[b]==6 for b in sim.neighbor_bodies))

    def test_old_yam_path_unchanged(self):
        from model import ANCHORS
        from yam_probe_robot import Yam
        path = ROOT/'policy/results/yam_probe_feedback_20261108/driver_source/test/yam_probe_robot.py'
        spec = importlib.util.spec_from_file_location('old_yam',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        old,new = module.Yam(ANCHORS['hard'],0),Yam(ANCHORS['hard'],0)
        self.assertEqual(old.sim.built.xml,new.sim.built.xml)
        np.testing.assert_array_equal(old.sim.d.qpos,new.sim.d.qpos)
        for robot in (old,new):
            robot.command(robot.initial_target+[0.,0.,.001],.09)
            robot.sim.step()
        np.testing.assert_array_equal(old.sim.d.qpos,new.sim.d.qpos)


if __name__=='__main__':
    unittest.main()
