"""Check the work of Fix 4.1 when ABC pad reactions are reduced to one wrist wrench."""

import argparse
import json

import torch
from isaaclab import sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import Articulation
from isaaclab_physx.physics import PhysxCfg

from src.isaac_yam_parcel.yam_scene import RAIL_JOINTS, YAM_USD, YamSceneFrames, arm_cfg, load_scene
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.neck_damping import NeckDamping, NeckDampingParams


def measure(sim, dt):
    scene = load_scene()
    robot = Articulation(arm_cfg(0, scene, YAM_USD, "/World/Yam"))
    sim.reset()
    q = robot.data.default_joint_pos.torch
    dq = torch.zeros_like(q)
    rails = [robot.joint_names.index(name) for name in RAIL_JOINTS]
    dq[:, rails] = .4
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=dq)
    sim.forward()
    robot.update(dt)

    link = robot.data.body_link_pose_w.torch
    link_rate = robot.data.body_link_vel_w.torch
    frames = YamSceneFrames(scene, robot.body_names, link)
    _, pads = frames.wrist_camera_and_pads(link, link.new_zeros((1, 3)))
    pad_ids = frames.body_ids[1:]
    pad_link_rate = link_rate.index_select(1, pad_ids)
    pad_rate = pad_link_rate[..., :3] + torch.linalg.cross(
        pad_link_rate[..., 3:], pads - link.index_select(1, pad_ids)[..., :3])
    wrist = robot.body_names.index(scene["wrist_camera"]["parent"])
    base = robot.data.body_com_pos_w.torch[:, wrist:wrist + 1]
    base_rate = robot.data.body_com_vel_w.torch[:, wrist:wrist + 1]
    v_base, omega = base_rate.split(3, dim=-1)
    omega_pad = omega.expand_as(pads)
    torch.testing.assert_close(pad_link_rate[..., 3:], omega_pad, atol=1e-6, rtol=0.)
    v_base_at_pad = v_base + torch.linalg.cross(omega_pad, pads - base)
    slide = pad_rate - v_base_at_pad

    # An already-captured 65 g fixture. The first anchor co-moves at half the rail
    # speed; the second neck is inactive. No motor or control input uses the anchor.
    anchors = pads + pads.new_tensor([0., 0., -.1])
    anchor_rate = v_base + torch.linalg.cross(omega_pad, anchors - base) + .5 * slide
    mass = pads.new_full((1, 1), .065)
    contacts = pads.new_zeros((1, 2, 6))
    contacts[:, 0, ContactState.CAPTURE] = 1.
    normal = pads.new_full((1, 2), 6.)
    neck = NeckDamping(mass, pads.new_full((1, 1, 2), .5), dt, NeckDampingParams(12., .05))
    identity = pads.new_tensor([[1., 0., 0., 0.]])
    inv_inertia = torch.diag_embed(pads.new_full((1, 3), 3000.))
    result = neck.rigid(pads, anchors, pad_rate, anchor_rate, contacts, contacts,
                        normal, mass, anchors - anchors.mean(dim=1, keepdim=True),
                        identity, inv_inertia, omega_pad)
    force = result.force
    wrist_force = -force.sum(dim=1, keepdim=True)
    wrist_torque = -torch.linalg.cross(anchors - base, force).sum(dim=1, keepdim=True)
    # Positive work means net energy injection. Use the measured pre-step
    # velocities for the instantaneous discrete-port power test.
    paired_work = dt * (result.anchor_power - result.pad_force_power - result.pad_couple_power).sum()
    wrist_only_work = dt * (
        result.anchor_power.sum() + (wrist_force * v_base + wrist_torque * omega).sum()
    )
    missing_rail_work = -dt * (force * slide).sum()
    torch.testing.assert_close(paired_work, wrist_only_work + missing_rail_work, atol=1e-9, rtol=0.)
    assert 0.39 < slide.norm(dim=-1).min().item() < .41
    assert force[:, 0].norm().item() > .4 and force[:, 1].count_nonzero().item() == 0

    # An equal/opposite central pinch has a zero wrist wrench. The two
    # actuated rails nevertheless see nonzero generalized forces.
    axis = torch.nn.functional.normalize(pads[:, 1] - pads[:, 0], dim=-1)
    squeeze = 6. * torch.stack((-axis, axis), dim=1)
    rail_load = (squeeze * torch.nn.functional.normalize(slide, dim=-1)).sum(dim=-1)
    squeeze_force = squeeze.sum(dim=1)
    squeeze_torque = torch.linalg.cross(pads - base, squeeze).sum(dim=1)
    assert squeeze_force.norm() < 1e-7 and squeeze_torque.norm() < 1e-6
    assert rail_load.abs().min() > 5.99

    return dict(dt_s=dt, rail_speed_m_s=slide.norm(dim=-1)[0].tolist(),
                F_neck_N=force[:, 0].norm().item(), paired_work_mJ=1000 * paired_work.item(),
                wrist_only_work_mJ=1000 * wrist_only_work.item(),
                omitted_rail_work_mJ=1000 * missing_rail_work.item(),
                symmetric_wrist_force_N=squeeze_force.norm().item(),
                symmetric_wrist_torque_Nm=squeeze_torque.norm().item(),
                symmetric_rail_effort_N=rail_load[0].tolist(),
                wrist_port_pass=bool(wrist_only_work * 1000 <= .1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream:
        parser.error("Use --headless with visualization and livestream disabled.")
    cfg = sim_utils.SimulationCfg(device="cuda:0", dt=.002, physics=PhysxCfg())
    with launch_simulation(cfg, args):
        sim = sim_utils.SimulationContext(cfg)
        assert not sim.has_gui and not sim.has_offscreen_render
        result = measure(sim, cfg.dt)
        print("ROUTE_B_WRIST_POWER " + json.dumps(result), flush=True)
        if not result["wrist_port_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
