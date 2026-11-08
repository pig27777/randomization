"""Guard against visually different targets, fake grasp attachments and thickened envelopes."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'policy'))
import parcel_heap as heap
heap.install_backend(heap.RUN/'contact_fix')


class HeapContract(unittest.TestCase):
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
