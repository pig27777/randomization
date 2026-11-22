"""CAN delays and subset reset for the motor-only observation."""

import unittest

import torch

from src.isaac_yam_parcel.joint_current_filter import JointCurrentEstimatorFilter, JointCurrentParams


class CurrentFilterTest(unittest.TestCase):
    def test_channel_delay_tare_and_partial_reset(self):
        ones = torch.ones((2, 1, 7))
        zeros = torch.zeros_like(ones)
        opening = torch.full((2, 1, 1), .04)
        delay = torch.tensor([5, 25]).reshape(-1, 1, 1).expand_as(ones)
        params = JointCurrentParams(ones, zeros, zeros, zeros, 15. * ones, zeros, delay, zeros[..., :1])
        model = JointCurrentEstimatorFilter(params)
        drive = 6. * ones
        first_response = torch.full((2,), -1)
        for tick in range(180):
            effort = zeros if tick < 80 else drive
            obs = model.step(effort, zeros, zeros, ones, opening)
            detected = (obs.delta_effort[:, 0, 0] > 0.) & (first_response < 0)
            first_response[detected] = tick
            torch.testing.assert_close(obs.tracking_error, -ones)
            torch.testing.assert_close(obs.opening_m, opening)
        self.assertEqual(first_response.tolist(), [85, 105])
        torch.testing.assert_close(obs.delta_effort, drive, atol=.001, rtol=0.)
        model.reset(torch.tensor([True, False]))
        obs = model.step(drive, zeros, zeros, zeros, opening)
        self.assertTrue((obs.delta_effort[0] == 0.).all())
        self.assertTrue((obs.delta_effort[1] > 5.99).all())


if __name__ == "__main__":
    unittest.main()
