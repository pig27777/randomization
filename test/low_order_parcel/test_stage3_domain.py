"""Distribution and fixed-corner checks for the Stage 3 dual-arm randomizer."""

from dataclasses import fields
import math
import os
import unittest

import torch

from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer


DEVICE = os.environ.get("STAGE3_TEST_DEVICE", "cpu")


class Stage3DomainTest(unittest.TestCase):
    def test_wide_physics_and_workcell_distributions(self):
        dr = DomainRandomizer(DEVICE, seed=347)
        e = dr.sample(16384)
        e.payload.validate()
        e.proxy.validate()
        k = 2. * e.proxy.k
        self.assertGreaterEqual(k.min().item(), 4.)
        self.assertLess(k.min().item(), 4.2)
        self.assertLessEqual(k.max().item(), 5000.)
        self.assertGreater(k.max().item(), 4700.)
        self.assertAlmostEqual(k.log().mean().exp().item() / math.sqrt(4. * 5000.), 1., delta=.055)
        low, high = dr.stiffness_bounds[e.material].split(1, dim=1)
        self.assertTrue(((k >= low) & (k <= high)).all())
        self.assertAlmostEqual(e.payload.total_mass.log().mean().exp().item(), math.sqrt(.03 * 2.), delta=.006)
        self.assertAlmostEqual((e.payload.max_indent_ratio == 0.).float().mean().item(), .35, delta=.012)
        self.assertAlmostEqual((e.clearance_m < .02).any(dim=1).float().mean().item(), .30, delta=.012)
        self.assertTrue((e.proxy.S_bag <= e.payload.fill_slack_capacity()).all())
        self.assertTrue((e.payload.slide_slack_max <= e.payload.cavity_slack(
            e.payload.bag_half_extents, e.payload.fill_ratio)).all())
        ratio = e.proxy.mu_eff[:, 1] / e.proxy.mu_eff[:, 0]
        self.assertTrue(((ratio >= .65) & (ratio <= 1.35)).all())
        self.assertLess(e.proxy.mu_eff[:, 0].min().item(), .021)
        self.assertTrue(((e.F_half >= .05) & (e.F_half <= 2.5)).all())
        self.assertTrue(((e.T_tear >= 20.) & (e.T_tear <= 60.)).all())
        print(f"Wide DR: whole-bag k={k.min().item():.3f}–{k.max().item():.3f} N/m, "
              f"rigid={(e.content == 0).float().mean().item():.3f}, "
              f"occluded={(e.clearance_m < .02).any(dim=1).float().mean().item():.3f}", flush=True)

    def test_curriculum_and_local_replacement(self):
        dr = DomainRandomizer(DEVICE, seed=8)
        e = dr.sample(128, curriculum_alpha=0.)
        self.assertTrue((e.yam.sensor_gain == 1.).all())
        self.assertTrue((e.yam.cutoff_hz == 2500.).all())
        for value in (e.yam.coulomb_N, e.yam.noise_N, e.yam.delay_steps, e.rgb.delay_steps,
                      e.rgb.rotation_rad, e.rgb.mask_jitter, e.rgb.drop_probability):
            self.assertEqual(value.count_nonzero().item(), 0)
        self.assertFalse((e.clearance_m < .02).any())
        mask = torch.arange(128, device=DEVICE) % 3 == 0
        new = dr.resample(e, mask, curriculum_alpha=1.)
        for old_group, new_group in ((e, new), (e.proxy, new.proxy), (e.payload, new.payload)):
            for field in fields(old_group):
                old, new_value = getattr(old_group, field.name), getattr(new_group, field.name)
                if isinstance(old, torch.Tensor):
                    torch.testing.assert_close(old[~mask], new_value[~mask], atol=0, rtol=0)
        for old, new_value in zip((*e.yam, *e.rgb), (*new.yam, *new.rgb), strict=True):
            torch.testing.assert_close(old[~mask], new_value[~mask], atol=0, rtol=0)
        self.assertTrue((new.yam.delay_steps[mask] >= 5).all())
        self.assertTrue((new.rgb.delay_steps[mask] >= 30).all())
        self.assertFalse(torch.equal(new.yam.sensor_gain[mask, 0], new.yam.sensor_gain[mask, 1]))
        self.assertFalse(torch.equal(new.rgb.delay_steps[mask, 0], new.rgb.delay_steps[mask, 1]))
        self.assertFalse(torch.equal(new.payload.total_mass[mask], e.payload.total_mass[mask]))

    def test_fixed_corners_and_initial_pose(self):
        dr = DomainRandomizer(DEVICE, seed=42)
        suite = CornerSuite(dr)
        ids = torch.tensor([9, 2, 7, 8, 5, 3, 4, 1, 6, 0, 9], device=DEVICE)
        e = suite.select(ids)
        e.proxy.validate()
        e.payload.validate()
        self.assertEqual(suite.names, tuple(f"C{i}" for i in range(1, 11)))
        torch.testing.assert_close(e.payload.total_mass[[0, 1, 4, -1], 0], dr.tensor([.065, 1.5, .065, .065]))
        self.assertAlmostEqual(e.payload.mass_weights[1].max().item(), .65)
        self.assertAlmostEqual((e.proxy.mu_eff[1, 1] / e.proxy.mu_eff[1, 0]).item(), .65)
        self.assertAlmostEqual(e.payload.table_friction[4, 0].item(), .06)
        self.assertAlmostEqual(e.payload.fill_ratio[7, 0].item(), .90)
        self.assertEqual(e.payload.max_indent_ratio[7, 0].item(), 0.)
        torch.testing.assert_close(e.clearance_m[2], dr.tensor([.006, .006]))
        self.assertAlmostEqual(e.fold_rad[3, 0].item(), .59)
        self.assertAlmostEqual(e.swing_rad[3, 0].item(), math.radians(15.))
        self.assertAlmostEqual(e.rgb.wrist_min_range_m[3, 0].item(), .18)
        self.assertTrue((e.yam.coulomb_N[[0, 6, -1]] == 2.).all())
        self.assertTrue((e.yam.delay_steps[[0, 6, -1]] == 25).all())
        self.assertTrue((e.rgb.delay_steps[[0, 6, -1]] == 60).all())
        self.assertAlmostEqual(e.F_half[5, 0].item(), 2.)
        another = CornerSuite(DomainRandomizer(DEVICE, seed=523)).select(ids)
        torch.testing.assert_close(e.proxy.k, another.proxy.k, atol=0, rtol=0)
        torch.testing.assert_close(e.yam.noise_N, another.yam.noise_N, atol=0, rtol=0)
        sampled = dr.sample(4)
        sim = sampled.make_sim()
        direction = sim.payload.frame(sampled.table_state(sim).payload)[:, :, 0]
        torch.testing.assert_close(direction[:, 0], sampled.yaw_rad[:, 0].cos())
        torch.testing.assert_close(direction[:, 1], sampled.yaw_rad[:, 0].sin())
