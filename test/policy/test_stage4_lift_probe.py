"""Regression for the Phase B actor inputs and the visual measurement window."""
import unittest

import torch

from policy.rss2027.domain import DomainRandomizer
from policy.rss2027.grasp_env import SoftParcelGraspEnv
from policy.rss2027.train import Policy
from src.low_order_parcel.lift_probe import probe_height


class Stage4ProbeTest(unittest.TestCase):
    def test_scaled_stroke_and_actor_entry(self):
        torch.manual_seed(18237)
        dr = DomainRandomizer("cpu", 18237)
        env = SoftParcelGraspEnv(dr.sample(3), open_loop=True)
        h = env.probe_height
        torch.testing.assert_close(probe_height(torch.tensor([.01, .10, .20])), torch.tensor([.04,.09,.10]))
        self.assertTrue((h > .025).all())
        touch = env.touch()
        micro = env.grasp(torch.tensor([0, 1, 1]), torch.full((3,4), .5))
        self.assertEqual(micro.shape, (3,32,18))
        self.assertTrue(torch.isfinite(micro).all())
        torch.testing.assert_close(env.request[:,1:2] - env.start[:,1:2], h)
        self.assertTrue((micro[:,:,-1] == 1).any())
        policy = Policy()
        weights = policy.state_dict()
        weights["encoder.weight_ih_l0"] = weights["encoder.weight_ih_l0"][:, :15]
        policy.load_state_dict(weights)
        self.assertTrue((policy.encoder.weight_ih_l0[:, 15:] == 0).all())
        for phase, obs in enumerate((touch, micro)):
            policy.decide(obs, env.privileged(), phase, greedy=True)
        self.assertGreater(micro[:,:,15].abs().max().item(), 0.)

    def test_image_ratio_keeps_updating_beyond_25mm(self):
        torch.manual_seed(151)
        e = DomainRandomizer("cpu", 151).sample(2)
        env = SoftParcelGraspEnv(e, "ablation_no_hw_dr", training=True)
        cam = env.rgb
        fk = env._pad(env.real).mean(1)
        lift = torch.ones(2,1,dtype=torch.bool)
        origin = env.state.payload
        cam.step(origin, cam.top_camera, fk, lift, exterior_height=env.top)
        # Both masks stay on the table through 25 mm. Only row 0 then follows.
        dz = torch.tensor([[0.,0.,.025]]).expand(2,-1)
        cam.step(origin, cam.top_camera, fk+dz, lift, exterior_height=env.top)
        moved = torch.tensor([[[0.,0.,.07]],[[0.,0.,0.]]])
        last = cam.step(origin._replace(pos=origin.pos+moved), cam.top_camera, fk+3.6*dz, lift, exterior_height=env.top)
        self.assertTrue(last.top_valid.all())
        self.assertGreater(last.vis_lift_ratio[0].item(), .45)
        self.assertEqual(last.vis_lift_ratio[1].item(), 0.)


if __name__ == "__main__":
    unittest.main()
