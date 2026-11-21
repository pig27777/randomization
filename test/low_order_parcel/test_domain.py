import unittest

import torch
from policy.rss2027.domain import DomainRandomizer, HardwareGapFilter


class RandomizationTest(unittest.TestCase):
    def test_distribution_and_local_reset(self):
        dr = DomainRandomizer("cpu", 7)
        e = dr.sample(8192)
        e.proxy.validate()
        e.x.validate()
        self.assertTrue((e.proxy.S_bag <= e.x.fill_slack_capacity()).all())
        self.assertTrue(((e.proxy.mu_eff[:, 1]/e.proxy.mu_eff[:, 0]).sub(1).abs() <= .35001).all())
        self.assertAlmostEqual(e.x.total_mass.log().mean().exp().item(), (.03*2)**.5, delta=.006)
        for i in range(4):
            rows = e.material == i
            corr = torch.corrcoef(torch.cat([e.proxy.alpha[rows],e.proxy.k[rows]], dim=1).T)[0,1]
            self.assertLess(corr, -.32)
        self.assertLess(e.proxy.mu_eff.min().item(), .04)
        self.assertLess(e.proxy.k.min()*2, 5.)
        self.assertGreater(e.proxy.k.max()*2, 2100.)
        ids = torch.arange(0,8192,2)
        old = e.x.total_mass.clone()
        old_mu = e.proxy.mu_eff.clone()
        e.reset_rows(ids, dr.sample(len(ids)))
        torch.testing.assert_close(e.x.total_mass[1::2], old[1::2])
        torch.testing.assert_close(e.proxy.mu_eff[1::2], old_mu[1::2])
        self.assertFalse(torch.equal(e.x.total_mass[::2],old[::2]))

    def test_fixed_corners_and_delay(self):
        dr = DomainRandomizer("cpu", 19)
        e = dr.sample(8, corners=torch.arange(8))
        e.proxy.validate()
        e.x.validate()
        cmd = torch.zeros(8, 4)
        f = HardwareGapFilter(e,cmd)
        e.noise.zero_()
        e.bias.zero_()
        for t in range(26):
            signal = f.actuate(cmd+1)
            obs = f.observe(torch.ones(8,6))
            self.assertEqual(float(signal[4,0]),float(t >= 25))
            self.assertEqual(float(obs[4,0]) > 0, t >= 25)
        self.assertTrue((f.filtered[4] < .7).all())
        f.reset(torch.arange(8)==4,cmd)
        self.assertTrue((f.signal[:,4]==0).all())
        self.assertTrue((f.signal[:,0]>0).all())
        self.assertAlmostEqual((e.proxy.mu_eff[2,1]/e.proxy.mu_eff[2,0]).item(),.65,places=5)
