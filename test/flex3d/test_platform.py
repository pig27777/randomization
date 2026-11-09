"""Verify I-37 against actual MuJoCo connect reaction signs."""
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from native_flex import FlexPinch


class LoadProjectionTest(unittest.TestCase):
    def attachment(self, gravity):
        xml=f'''<mujoco><option gravity="0 0 {gravity}"/>
          <worldbody><body name="palm"/><body name="node"><freejoint/>
            <geom type="sphere" size=".01" mass="1" contype="0"/></body></worldbody>
          <equality><connect body1="palm" body2="node" anchor="0 0 0" solref=".005 1"/></equality></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        for _ in range(100):mujoco.mj_step(m,d)
        pinch=FlexPinch.__new__(FlexPinch)
        pinch.sim=SimpleNamespace(m=m,d=d);pinch.active=[0]
        pinch.segment=0;pinch.segments=[0];pinch.load=0.;pinch.peak_force=0.
        pinch.coefficient=1.;pinch.events=[];pinch.broken=False
        return pinch

    def test_actual_connect_pull_releases_and_press_does_not(self):
        pull=self.attachment(-9.81);press=self.attachment(9.81)
        np.testing.assert_allclose(pull.force_vector(),[0,0,-9.81],atol=1e-6)
        np.testing.assert_allclose(press.force_vector(),[0,0,9.81],atol=1e-6)
        for _ in range(50):self.assertFalse(press.check_load(.001,1.))
        self.assertEqual(press.load,0.)
        self.assertTrue(pull.check_load(.001,1.))
        self.assertFalse(pull.sim.d.eq_active[0])

    def test_horizontal_component_counts_toward_capacity(self):
        pinch=self.attachment(0.)
        self.assertTrue(pinch.check_load(.001,1.,np.array([10.,0.,100.])))
        self.assertAlmostEqual(pinch.load,10/6)


if __name__=='__main__':unittest.main()
