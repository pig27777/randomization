"""A0 conservative bounds and explicit handling of shared MuJoCo model fields."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mjwarp_v13 import stacked_model, union_aabbs


class BatchModelTest(unittest.TestCase):
    def test_union_covers_shifted_boxes(self):
        boxes=np.array([[[2,0,0,1,2,3]], [[-2,1,0,2,1,4]]],dtype=float)
        merged=union_aabbs(boxes)
        np.testing.assert_array_equal(merged, [[-.5,0,0,3.5,2,4]])
        self.assertTrue(np.all(merged[:,:3]-merged[:,3:]<=boxes[...,:3]-boxes[...,3:]))
        self.assertTrue(np.all(merged[:,:3]+merged[:,3:]>=boxes[...,:3]+boxes[...,3:]))

    def test_union_copy_preserves_cpu_models(self):
        def make(radius):
            xml=f'<mujoco><worldbody><body><freejoint/><geom size="{radius}"/>' \
                f'<geom pos=".1 0 0" size="{radius}"/></body></worldbody></mujoco>'
            return SimpleNamespace(m=mujoco.MjModel.from_xml_string(xml))
        sims=[make(.02),make(.03)]
        originals=[s.m.bvh_aabb.copy() for s in sims]
        class Inspected(Exception):pass
        def inspect(model,**unused):
            self.assertIsNot(model,sims[0].m)
            np.testing.assert_array_equal(model.bvh_aabb,union_aabbs(np.stack(originals)))
            for s,old in zip(sims,originals):np.testing.assert_array_equal(s.m.bvh_aabb,old)
            raise Inspected
        with patch('mjwarp_v13.mjw.put_model',side_effect=inspect):
            with self.assertRaises(Inspected):stacked_model(sims)

    def test_unknown_shared_difference_is_rejected(self):
        def make(value):
            return SimpleNamespace(m=SimpleNamespace(bvh_aabb=np.zeros((1,6)),dof_M0=np.ones(1),
                dof_length=np.ones(1),stat=SimpleNamespace(meaninertia=1.),unknown=np.array([value])))
        with self.assertRaisesRegex(ValueError,'non-batchable.*unknown'):
            stacked_model([make(1),make(2)])
