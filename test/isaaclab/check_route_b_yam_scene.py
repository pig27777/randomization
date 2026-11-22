"""Import both ABC YAMs into GPU PhysX and check the table-relative frames."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab import sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_physx.physics import PhysxCfg

from src.isaac_yam_parcel.yam_scene import (
    ARM_JOINTS, RAIL_JOINTS, YamSceneFrames, arm_cfg, load_scene, table_cfg,
)


def read_frames(robot, scene, index):
    from scipy.spatial.transform import Rotation

    np.testing.assert_allclose(robot.data.root_link_pos_w.torch[0].cpu(), scene["arm_bases_m"][index], atol=1e-6)
    root_xyzw = robot.data.root_link_quat_w.torch[0].cpu().numpy()
    expected_base = Rotation.from_quat(scene["arm_base_quat_wxyz"], scalar_first=True)
    np.testing.assert_allclose(Rotation.from_quat(root_xyzw).as_matrix(), expected_base.as_matrix(), atol=1e-6)
    assert set(robot.joint_names) == set(ARM_JOINTS + RAIL_JOINTS)
    pose_tensor = robot.data.body_link_pose_w.torch
    frames = YamSceneFrames(scene, robot.body_names, pose_tensor)
    origins = pose_tensor.new_tensor([[.25, -.45, .75]])
    camera, pad_faces = frames.wrist_camera_and_pads(pose_tensor, origins)
    poses = pose_tensor[0].cpu().numpy()

    def world_from_link(name):
        pose = poses[robot.body_names.index(name)]
        return pose[:3], Rotation.from_quat(pose[3:])

    wrist = scene["wrist_camera"]
    wrist_position, wrist_rotation = world_from_link(wrist["parent"])
    camera_position = wrist_position + wrist_rotation.apply(wrist["position_m"])
    camera_rotation = wrist_rotation * Rotation.from_quat(wrist["quat_wxyz"], scalar_first=True)
    faces = []
    for pad in scene["pads"]:
        position, rotation = world_from_link(pad["parent"])
        box_to_link = Rotation.from_euler("z", pad["box_yaw_rad"])
        inside = box_to_link.apply([0., .5 * scene["pad_box_size_m"][1], 0.])
        faces.append(position + rotation.apply(np.asarray(pad["box_center_m"]) + inside))
    faces = np.array(faces)
    assert np.isfinite(faces).all() and faces[:, 2].min() > 0., (index, faces, wrist_position,
                                                                robot.data.joint_pos.torch.tolist())
    assert np.isfinite(camera_position).all()
    np.testing.assert_allclose((pad_faces + origins[:, None])[0].cpu(), faces, atol=1e-6)
    np.testing.assert_allclose((camera.position + origins)[0].cpu(), camera_position, atol=1e-6)
    np.testing.assert_allclose(camera.world_from_camera[0].cpu(), camera_rotation.as_matrix(), atol=1e-6)
    top = frames.top_camera(1)
    np.testing.assert_allclose(top.position[0].cpu(), scene["top_camera"]["position_m"], atol=1e-6)
    expected_top = Rotation.from_quat(scene["top_camera"]["quat_wxyz"], scalar_first=True)
    np.testing.assert_allclose(top.world_from_camera[0].cpu(), expected_top.as_matrix(), atol=1e-6)
    return dict(joints=robot.joint_names, links=robot.body_names,
                pad_face_z_m=faces[:, 2].tolist(), opening_m=float(np.linalg.norm(faces[0] - faces[1])),
                wrist_camera_m=camera_position.tolist(), wrist_optical_z=camera_rotation.as_matrix()[:, 2].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream or not args.usd.is_file():
        parser.error("Supply --headless and --usd with visualization and livestream disabled.")

    scene = load_scene()
    table = scene["table"]
    top = scene["top_camera"]
    assert table["center_m"][2] + .5 * table["size_m"][2] == 0.

    cfg = sim_utils.SimulationCfg(device="cuda:0", dt=.002, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        sim = sim_utils.SimulationContext(cfg)
        assert not sim.has_gui and not sim.has_offscreen_render
        from scipy.spatial.transform import Rotation
        from pxr import Usd, UsdGeom, UsdPhysics

        camera_to_world = Rotation.from_quat(top["quat_wxyz"], scalar_first=True)
        center_in_camera = camera_to_world.inv().apply(-np.array(top["position_m"]))
        assert center_in_camera[2] > 0.
        normalized_center = center_in_camera[:2] / center_in_camera[2]
        assert abs(normalized_center[1]) < np.tan(np.deg2rad(top["vertical_fov_deg"] / 2))
        slab = table_cfg(scene, "/World/Table")
        slab.spawn.func(slab.prim_path, slab.spawn, translation=slab.init_state.pos)
        robots = [Articulation(arm_cfg(i, scene, args.usd.resolve(), f"/World/Yam_{i}")) for i in range(2)]
        sim.reset()
        table_prim = sim.stage.GetPrimAtPath("/World/Table")
        table_bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        table_top = table_bound.ComputeWorldBound(table_prim).ComputeAlignedRange().GetMax()[2]
        assert abs(table_top) < 1e-7
        assert any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(table_prim))
        assert not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(table_prim))
        positions = [robot.data.default_joint_pos.torch for robot in robots]
        for robot, position in zip(robots, positions, strict=True):
            robot.write_joint_position_to_sim_index(position=position)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position))
        sim.forward()
        for index, (robot, position) in enumerate(zip(robots, positions, strict=True)):
            current = robot.data.joint_pos.torch
            torch.testing.assert_close(current, position, atol=1e-5, rtol=0.)
            gripper = robot.data.body_link_pose_w.torch[0, robot.body_names.index("gripper")].cpu().numpy()
            wrist = Rotation.from_quat(gripper[3:])
            top_grasp = Rotation.from_euler("z", 270., degrees=True)
            assert (wrist * top_grasp.inv()).magnitude() < 1e-4, (index, gripper)
            print("ABC_YAM_RESET", index, gripper.tolist(), flush=True)
        for _ in range(20):
            for robot, position in zip(robots, positions, strict=True):
                robot.actuators.target_command.set_position_index(value=position, full_data=True)
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in robots:
                robot.update(cfg.dt)
        rows = []
        for index, (robot, target) in enumerate(zip(robots, positions, strict=True)):
            current = robot.data.joint_pos.torch
            assert current.device.type == "cuda" and current.isfinite().all()
            row = read_frames(robot, scene, index)
            row["maximum_joint_deviation"] = (current - target).abs().max().item()
            rows.append(row)
        print("ABC_YAM_SCENE_PASS " + json.dumps(dict(configuration=scene["abc_source"],
              dt_s=cfg.dt, steps=20, table_top_m=table_top, table_static=True,
              table_center_in_top_frame=normalized_center.tolist(), arms=rows)), flush=True)


if __name__ == "__main__":
    main()
