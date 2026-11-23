"""Pad-centred YAM inverse kinematics for simulated demonstrations."""

import math

import torch
from isaaclab.controllers import DifferentialIKController


def transition(time, start, end):
    phase = min(1., max(0., (time - start) / (end - start)))
    return .5 - .5 * math.cos(math.pi * phase)


class PadIK(DifferentialIKController):
    """Track a pad center in metres and give orientation lower task-space weight."""

    angular_weight = None

    def _compute_pose_task(self, ee_pos, ee_quat, jacobian):
        jac, error = super()._compute_pose_task(ee_pos, ee_quat, jacobian)
        scale = error.new_tensor([6., 6., 6., 1., 1., 1.])
        if self.angular_weight is not None:
            scale = torch.cat((scale[:3].expand(len(error), -1), self.angular_weight[:, None].expand(-1, 3)), dim=-1)
        return jac * scale.unsqueeze(-1), error * scale
