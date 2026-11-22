"""A wrist wrench and independent prismatic-joint load for the fixed-base ABC YAM."""

from typing import NamedTuple

import torch
import warp as wp
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_physx.assets.articulation.articulation import Articulation

from src.low_order_parcel.virtual_rgb import CameraPose
from .yam_scene import RAIL_JOINTS, YamSceneFrames


class RailLoadedYam(Articulation):
    """Submit external rail loads after computing and recording pure motor effort."""

    def _initialize_impl(self):
        super()._initialize_impl()
        self.rail_ids = [self.joint_names.index(name) for name in RAIL_JOINTS]
        self.rail_indices = torch.tensor(self.rail_ids, device=self.device)
        self.rail_load = torch.zeros_like(self.data.default_joint_pos.torch[:, :2])
        self._total_effort = torch.empty_like(self.data.default_joint_pos.torch)
        self._backend_effort = torch.empty_like(self._total_effort)
        self._backend_effort_view = wp.from_torch(self._backend_effort, dtype=wp.float32)
        self._joint_order = (wp.to_torch(self.data.joint_ordering.backend_to_user).long()
                             if self.data.has_joint_ordering else None)

    def write_data_to_sim(self):
        super().write_data_to_sim()
        # PhysX exposes one generalized-force buffer. Compose it only at the
        # backend boundary, downstream of target processing, actuator limits,
        # and the applied_effort view used by the current estimator.
        self._total_effort.copy_(self.actuators.applied_effort.torch)
        self._total_effort.index_add_(1, self.rail_indices, self.rail_load)
        if self._joint_order is None:
            self._backend_effort.copy_(self._total_effort)
        else:
            torch.index_select(self._total_effort, 1, self._joint_order, out=self._backend_effort)
        self.root_view.set_dof_actuation_forces(self._backend_effort_view, self._ALL_INDICES)


class YamPortState(NamedTuple):
    pads: torch.Tensor
    pad_velocity: torch.Tensor
    wrist_quat_xyzw: torch.Tensor
    wrist_omega: torch.Tensor
    wrist_com: torch.Tensor
    wrist_velocity: torch.Tensor
    rail_axis: torch.Tensor
    pad_jacobian: torch.Tensor
    angular_jacobian: torch.Tensor
    wrist_camera: CameraPose


class YamLoadPort:
    def __init__(self, robot, scene, env_origins):
        self.robot = robot
        self.origins = env_origins
        self.frames = YamSceneFrames(scene, robot.body_names, env_origins)
        self.wrist_id = robot.body_names.index(scene["wrist_camera"]["parent"])
        self.wrist_indices = wp.array([self.wrist_id], dtype=wp.int32, device=robot.device)
        self.pad_ids = self.frames.body_ids[1:]
        if not robot.is_fixed_base or robot.body_names[0] != "base":
            raise ValueError("The ABC port expects the I2RT fixed base at index zero")
        self.pad_jacobian_ids = self.pad_ids - 1
        self.pair_ids = torch.arange(2, device=env_origins.device)

    def read(self):
        robot = self.robot
        pose = robot.data.body_link_pose_w.torch
        velocity = robot.data.body_link_vel_w.torch
        camera, pads = self.frames.wrist_camera_and_pads(pose, self.origins)
        pad_links = pose.index_select(1, self.pad_ids)[..., :3]
        lever = pads + self.origins[:, None] - pad_links
        pad_vel, pad_omega = velocity.index_select(1, self.pad_ids).split(3, dim=-1)
        pad_velocity = pad_vel + torch.linalg.cross(pad_omega, lever)

        # Jacobian rows at the actual pad face, rather than the finger link origin.
        link_j = robot.data.body_link_jacobian_w.torch.index_select(1, self.pad_jacobian_ids)
        linear, angular = link_j.split(3, dim=-2)
        pad_j = linear + torch.linalg.cross(angular, lever.unsqueeze(-1), dim=-2)
        axes = pad_j.mT[:, self.pair_ids, robot.rail_indices]
        rate = robot.data.body_com_vel_w.torch[:, self.wrist_id]
        return YamPortState(pads, pad_velocity, pose[:, self.wrist_id, 3:], rate[:, 3:],
                            robot.data.body_com_pos_w.torch[:, self.wrist_id] - self.origins, rate[:, :3],
                            axes, pad_j, angular, camera)

    def mobility(self, state, anchor_from_pad, direction):
        """nᵢᵀJᵢ M⁻¹ Jⱼᵀnⱼ, including free rails and the rotation term at the anchor."""
        jac = state.pad_jacobian + torch.linalg.cross(
            state.angular_jacobian, anchor_from_pad.unsqueeze(-1), dim=-2)
        projected = (jac * direction.unsqueeze(-1)).sum(dim=-2)
        inverse = torch.linalg.solve_ex(self.robot.data.mass_matrix.torch, projected.mT).result
        return projected @ inverse

    def write(self, state, wrist_wrench, pad_reaction):
        """PhysX wrench origin is the link COM. Pad couple is already in wrist_wrench."""
        force, couple = wrist_wrench.split(3, dim=-1)
        midpoint = state.pads.mean(dim=1)
        torque = couple + torch.linalg.cross(midpoint - state.wrist_com, force)
        q = state.wrist_quat_xyzw
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=quat_apply_inverse(q, force).unsqueeze(1),
            torques=quat_apply_inverse(q, torque).unsqueeze(1),
            body_ids=self.wrist_indices,
        )
        self.robot.rail_load.copy_((state.rail_axis * pad_reaction).sum(dim=-1))
