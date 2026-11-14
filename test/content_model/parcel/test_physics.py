"""Behavioral checks for device sampling, contact capture and force transfer."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel import kernels as K
from contents.parcel.model import film_proxy


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=Batch(8,prototype=3)

    def setUp(self):
        self.b.reset()

    def solve(self, close=0):
        b=self.b
        b.data.xfrc_applied.zero_()
        b.data.qfrc_applied.zero_()
        wp.launch(K.dynamics,b.nworld,[b.dt,0,2,b.cart,1,close,0,0,.04,b.params,b.states,b.ids,
            b.normal,b.gap,b.data.xpos,b.data.xmat,b.data.xipos,b.data.qpos,b.data.qfrc_applied,b.data.xfrc_applied])

    def active(self, captured=.05):
        b=self.b
        state=b.states.numpy()
        state['mode']=1
        state['captured']=captured
        state['rest']=.01
        b.states.assign(state)
        p=b.params.numpy()
        p['strength']=1000
        b.params.assign(p)

    def test_native_mass_inertia_reset(self):
        b=self.b
        p=b.params.numpy()
        masses=b.model.body_mass.numpy()
        ids=b.ids.numpy()
        np.testing.assert_allclose(masses[:,ids[:,0]].sum(1)+masses[:,ids[:,2]].sum(1),p['mass'],rtol=1e-6)
        self.assertTrue((b.model.body_inertia.numpy()[:,ids[:,0]]>0).all())
        np.testing.assert_allclose(b.model.geom_size.numpy()[:,ids[0,1],0]*2/.98,p['size'][:,0]/3,rtol=1e-6)
        self.assertTrue((p['mu_d']<=p['mu_s']).all())
        self.assertTrue(np.ptp(p['mass'])>.01)
        b.step()
        b.reset()
        np.testing.assert_array_equal(b.params.numpy(),p)
        b.reset(1)
        self.assertFalse(np.array_equal(b.params.numpy(),p))

    def test_force_and_torque_conservation(self):
        b=self.b
        self.active()
        b.normal.fill_(10000.)
        self.solve()
        f=b.data.xfrc_applied.numpy()
        com=b.data.xipos.numpy()
        np.testing.assert_allclose(f[:,:,:3].sum(1),0,atol=1e-5)
        global_torque=(f[:,:,3:]+np.cross(com,f[:,:,:3])).sum(1)
        np.testing.assert_allclose(global_torque,0,atol=1e-5)
        self.assertTrue((b.states.numpy()['tension']>0).any())
        state=b.states.numpy()
        state['mode']=0
        b.states.assign(state)
        self.solve()
        np.testing.assert_array_equal(b.data.xfrc_applied.numpy(),0)

    def test_slip_detach_and_rupture(self):
        b=self.b
        self.active(.001)
        b.normal.zero_()
        for _ in range(30):
            self.solve()
        self.assertTrue((b.states.numpy()['mode']==K.DETACHMENT).all())
        np.testing.assert_array_equal(b.states.numpy()['tension'],0)
        self.active()
        b.normal.fill_(10000.)
        p=b.params.numpy()
        p['strength']=.001
        b.params.assign(p)
        self.solve()
        self.assertTrue((b.states.numpy()['mode']==K.RUPTURE).all())
        np.testing.assert_array_equal(b.data.xfrc_applied.numpy(),0)

    def test_engagement_event_and_material(self):
        b=self.b
        b.normal.fill_(10.)
        state=b.states.numpy()
        state['gathered']=.003
        b.states.assign(state)
        self.solve(close=1)
        first=b.states.numpy()
        self.solve(close=1)
        second=b.states.numpy()
        np.testing.assert_array_equal(first['rng'],second['rng'])
        self.assertTrue((second['gathered']>=0).all())
        self.assertTrue((second['gathered']<=b.params.numpy()['budget'][:,1]).all())
        self.assertTrue((second['captured']<=.02).all())
        # Same closure sample with same seed and state is reproducible.
        b.reset()
        b.normal.fill_(10.)
        state=b.states.numpy()
        state['gathered']=.003
        b.states.assign(state)
        self.solve(close=1)
        np.testing.assert_array_equal(first,b.states.numpy())

    def test_friction_slack_and_environment_isolation(self):
        b=self.b
        self.active(.05)
        b.normal.fill_(.1)
        self.solve()
        low=b.states.numpy()['slip'].copy()
        state=b.states.numpy();state['slip']=0;state['mode']=1
        b.states.assign(state);b.normal.fill_(10000.)
        self.solve()
        self.assertTrue((b.states.numpy()['slip']<=low).all())
        initial=b.states.numpy()['tension'].copy()
        state=b.states.numpy();state['rest']+=.01
        b.states.assign(state)
        self.solve()
        self.assertTrue((b.states.numpy()['tension']<=initial+1e-5).all())
        reference=b.data.xfrc_applied.numpy().copy()
        state=b.states.numpy();state['mode'][0]=0;b.states.assign(state)
        self.solve()
        np.testing.assert_array_equal(b.data.xfrc_applied.numpy()[1:],reference[1:])
        np.testing.assert_array_equal(b.data.xfrc_applied.numpy()[0],0)

    def test_film_normal_requires_gathered_material(self):
        b=self.b
        q=b.data.qpos.numpy()
        addresses=b.addresses.numpy()
        q[:,addresses[3,0]]=.0015
        q[:,addresses[4,0]]=.0015
        b.data.qpos.assign(q)
        b.data.qfrc_applied.zero_()
        b.normal.zero_()
        args=[b.states,b.params,b.data.qpos,b.data.qvel,b.addresses,b.normal,b.data.qfrc_applied]
        wp.launch(film_proxy,b.nworld,args)
        np.testing.assert_array_equal(b.normal.numpy(),0)
        state=b.states.numpy();state['gathered']=.003;b.states.assign(state)
        wp.launch(film_proxy,b.nworld,args)
        self.assertTrue((b.normal.numpy()>0).all())

    def test_response_mapping(self):
        p=self.b.params.numpy()
        depth=10/p['compression']
        force=p['compression']*depth+p['quadratic']*depth**2
        np.testing.assert_allclose(force,20,rtol=1e-6)
        self.assertTrue((p['budget'].sum(1)<=2*(p['slack']*p['size'][:,0]+p['slack_y']*p['size'][:,1])+1e-7).all())


if __name__=='__main__':
    unittest.main()
