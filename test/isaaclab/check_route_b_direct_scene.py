"""Check the shared ABC scene through the Isaac Lab DirectRLEnv lifecycle."""

import argparse
import json

import torch
from isaaclab import sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab_physx.physics import PhysxCfg

from src.isaac_yam_parcel.yam_scene import YamSceneFrames, abc_scene_cfg, load_scene


class SceneCheckEnv(DirectRLEnv):
    def _pre_physics_step(self, actions):
        self.substeps = 0

    def _apply_action(self):
        self.substeps += 1
        for robot in self.scene.articulations.values():
            robot.actuators.target_command.set_position_index(
                value=robot.data.default_joint_pos.torch, full_data=True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        for robot in self.scene.articulations.values():
            q = robot.data.default_joint_pos.torch[env_ids]
            robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)

    def _get_observations(self):
        return {"policy": self.scene["arm_a"].data.joint_pos.torch}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return done, done


def check(env, dual_arm):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab.utils.math import quat_apply

    scene = load_scene()
    env.reset()
    assert not env.sim.has_gui and not env.sim.has_offscreen_render
    assert not env.scene.sensors
    expected_arms = ("arm_a", "arm_b") if dual_arm else ("arm_a",)
    assert tuple(env.scene.articulations) == expected_arms
    env.step(torch.zeros((env.num_envs, env.cfg.action_space), device=env.device))
    assert env.substeps == 20 and env.physics_dt == .002 and env.step_dt == .040
    origins = env.scene.env_origins
    assert len(origins.unique(dim=0)) == env.num_envs
    positions = origins.cpu().tolist()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for env_path, origin in zip(env.scene.env_prim_paths, positions, strict=True):
        table = env.sim.stage.GetPrimAtPath(f"{env_path}/Table")
        top = bbox.ComputeWorldBound(table).ComputeAlignedRange().GetMax()[2]
        assert abs(top - origin[2]) < 1e-7
        assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(table))
        assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(table))

    minimum_height = []
    for index, arm in enumerate(expected_arms):
        robot = env.scene[arm]
        poses = robot.data.body_link_pose_w.torch
        frames = YamSceneFrames(scene, robot.body_names, poses)
        camera, pads = frames.wrist_camera_and_pads(poses, origins)
        torch.testing.assert_close(robot.data.root_link_pos_w.torch - origins,
                                   origins.new_tensor(scene["arm_bases_m"][index]).expand_as(origins),
                                   atol=1e-6, rtol=0.)
        wrist = poses[:, robot.body_names.index(scene["wrist_camera"]["parent"])]
        mount = origins.new_tensor(scene["wrist_camera"]["position_m"]).expand_as(origins)
        torch.testing.assert_close(camera.position, wrist[:, :3] - origins + quat_apply(wrist[:, 3:], mount),
                                   atol=1e-6, rtol=0.)
        torch.testing.assert_close(pads, pads[:1].expand_as(pads), atol=1e-5, rtol=0.)
        torch.testing.assert_close(camera.world_from_camera,
                                   camera.world_from_camera[:1].expand_as(camera.world_from_camera),
                                   atol=1e-5, rtol=0.)
        assert pads.isfinite().all() and pads[..., 2].min() > 0.
        minimum_height.append(pads[..., 2].min().item())
    return dict(envs=env.num_envs, arms=len(expected_arms), table_top_m=0.,
                physics_dt_s=env.physics_dt, policy_dt_s=env.step_dt, substeps=env.substeps,
                sensors=len(env.scene.sensors), minimum_pad_height_m=minimum_height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dual-arm", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream or args.num_envs < 1:
        parser.error("Use --headless and a positive batch size with visualization and livestream disabled.")
    cfg = DirectRLEnvCfg(
        sim=sim_utils.SimulationCfg(device="cuda:0", dt=.002, render_interval=20, physics=PhysxCfg()),
        scene=abc_scene_cfg(args.num_envs, dual_arm=args.dual_arm),
        seed=17, decimation=20, episode_length_s=1., action_space=14 if args.dual_arm else 7,
        observation_space=8,
    )
    with launch_simulation(cfg.sim, args):
        env = SceneCheckEnv(cfg)
        try:
            print("ABC_DIRECT_SCENE_PASS " + json.dumps(check(env, args.dual_arm)), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
