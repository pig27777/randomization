"""Check an actual Isaac Lab PhysX CUDA buffer against the frozen parcel kernel."""
import argparse
import importlib.metadata as metadata
import json
import math
import os
import sys

import torch
import warp as wp
import isaaclab
import src.low_order_parcel
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab import sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab_physx.physics import PhysxCfg
from src.low_order_parcel.domain_randomizer import CornerSuite, DomainRandomizer
from src.low_order_parcel.dual_arm_parcel_sim import DualArmAction
from src.low_order_parcel.macro_parcel_sim import ParcelAction

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Require the no-GUI smoke test.")
add_launcher_args(parser)
parser.set_defaults(visualizer=[], livestream=0)
args = parser.parse_args()
if not args.headless or args.visualizer or args.livestream:
    parser.error("Use --headless with visualization and livestream disabled.")

device = "cuda:0"
cfg = sim_utils.SimulationCfg(device=device, dt=.002, physics=PhysxCfg())
print("ROUTE_B_IMPORT " + json.dumps(dict(
    pid=os.getpid(), python=sys.version.split()[0], executable=sys.executable, torch=torch.__version__,
    cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
    isaac_lab=metadata.version("isaaclab"), isaac_sim=metadata.version("isaacsim"), warp=wp.__version__,
    isaac_lab_source=list(isaaclab.__path__), parcel_source=list(src.low_order_parcel.__path__))), flush=True)

with launch_simulation(cfg, args):
    sim = sim_utils.SimulationContext(cfg)
    assert not sim.has_gui and not sim.has_offscreen_render
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/InteropProbe",
        spawn=sim_utils.CuboidCfg(size=(.1, .1, .1), rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                                   mass_props=sim_utils.MassCfg(mass=1.),
                                   collision_props=sim_utils.UsdPhysicsCollisionCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., .30))))
    sim.reset()
    raw_pose = cube.root_view.get_transforms()
    pose = wp.to_torch(raw_pose)
    raw_vel = cube.root_view.get_velocities()
    vel = wp.to_torch(raw_vel)
    assert pose.data_ptr() == raw_pose.ptr and vel.data_ptr() == raw_vel.ptr
    assert pose.device.type == vel.device.type == "cuda" and pose.dtype == vel.dtype == torch.float32
    z0 = pose[0, 2].item()
    vz0 = vel[0, 2].item()

    episode = CornerSuite(DomainRandomizer(device, 347)).select(torch.tensor([5], device=device))
    parcel = episode.make_sim()
    state = episode.table_state(parcel)
    ones = pose.new_ones((1, 2))
    pins = pose[:, None, :3].expand(-1, 2, -1)
    speeds = vel[:, None, :3].expand_as(pins)
    assert pins.data_ptr() == raw_pose.ptr and speeds.data_ptr() == raw_vel.ptr
    action = ParcelAction(pins, speeds, ones * 0., ones * 0., ones * 0., ones,
                          pose.new_zeros((1,3)), pose.new_zeros((1,1)))
    enabled = torch.tensor([[True, False]], device=device)
    out = parcel.step(state, DualArmAction(action, action, enabled))
    assert out.wrist_wrench.device == pose.device and out.wrist_wrench.isfinite().all()

    # A known external test force isolates write-back from capture physics.
    force = pose.new_tensor([[0., 0., 14.]])
    torque = torch.zeros_like(force)
    idx = torch.zeros(1, dtype=torch.int32, device=device)
    force_w, torque_w = wp.from_torch(force.flatten()), wp.from_torch(torque.flatten())
    assert force_w.ptr == force.data_ptr() and torque_w.ptr == torque.data_ptr()
    cube.root_view.apply_forces_and_torques_at_position(force_data=force_w, torque_data=torque_w,
        position_data=None, indices=wp.from_torch(idx), is_global=True)
    sim.step(render=False)
    cube.update(cfg.dt)
    z1 = wp.to_torch(cube.root_view.get_transforms())[0, 2].item()
    vz1 = wp.to_torch(cube.root_view.get_velocities())[0, 2].item()
    expected_dvz = (14. + cfg.gravity[2]) * cfg.dt
    assert math.isclose(vz1 - vz0, expected_dvz, abs_tol=1e-4), (vz0, vz1, expected_dvz)
    print("ROUTE_B_INTEROP_PASS " + json.dumps(dict(
        pid=os.getpid(), dt_s=cfg.dt, physics=type(cfg.physics).__name__, device=str(pose.device),
        pose_dtype=str(pose.dtype), shape=list(pose.shape), pose_zero_copy=True, velocity_zero_copy=True,
        torch_to_physx_force_zero_copy=True, z_start_m=z0, z_after_m=z1, vz_before_m_s=vz0, vz_after_m_s=vz1,
        expected_dvz_m_s=expected_dvz, actual_dvz_m_s=vz1 - vz0,
        parcel_wrench_shape=list(out.wrist_wrench.shape), parcel_com_m=parcel.payload.com(out.state.payload).tolist(),
        kit_gui=sim.has_gui, offscreen=sim.has_offscreen_render)), flush=True)
