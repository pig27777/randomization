"""Verify Fix 4.1 work closure and current isolation on the ABC articulation."""

import argparse
import json

import torch
from isaaclab import sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_physx.physics import PhysxCfg

from src.isaac_yam_parcel.joint_current_filter import JointCurrentEstimatorFilter, JointCurrentParams
from src.isaac_yam_parcel.yam_scene import YAM_USD, arm_cfg, load_scene
from src.low_order_parcel.lumped_contact_proxy import ContactState
from src.low_order_parcel.neck_damping import NeckDamping, NeckDampingParams


def measure(robot, sim, dt):
    from src.isaac_yam_parcel.yam_load_port import YamLoadPort

    scene = load_scene()
    q = robot.data.default_joint_pos.torch
    dq = torch.zeros_like(q)
    rails = robot.rail_ids
    dq[:, rails] = .4
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=dq)
    sim.forward()
    robot.update(dt)

    port = YamLoadPort(robot, scene, q.new_zeros((1, 3)))
    state = port.read()
    pads, pad_rate = state.pads, state.pad_velocity
    base, v_base, omega = (axis[:, None] for axis in (state.wrist_com, state.wrist_velocity, state.wrist_omega))
    omega_pad = omega.expand_as(pads)
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
    inputs = (pads, anchors, pad_rate, anchor_rate, contacts, contacts, normal, mass,
              anchors - anchors.mean(dim=1, keepdim=True), identity, inv_inertia, omega_pad)
    frozen = neck.rigid(*inputs)
    result = neck.rigid(*inputs, port_mobility=lambda r, n: port.mobility(state, r, n))
    torch.testing.assert_close(result.force, frozen.force, atol=1e-7, rtol=0.)
    force = result.force
    wrist_force = -force.sum(dim=1, keepdim=True)
    wrist_torque = -torch.linalg.cross(anchors - base, force).sum(dim=1, keepdim=True)
    # Positive work means net energy injection. Use the measured pre-step
    # velocities for the instantaneous discrete-port power test.
    paired_work = dt * (result.anchor_power - result.pad_force_power - result.pad_couple_power).sum()
    wrist_only_work = dt * (
        result.anchor_power.sum() + (wrist_force * v_base + wrist_torque * omega).sum()
    )
    couple_mid = (result.pad_couple - torch.linalg.cross(pads - pads.mean(dim=1, keepdim=True), force)).sum(dim=1)
    port.write(state, torch.cat((wrist_force[:, 0], couple_mid), dim=-1), -force)
    rail_work = dt * (robot.rail_load * robot.data.joint_vel.torch[:, rails]).sum()
    closed_work = wrist_only_work + rail_work
    torch.testing.assert_close(paired_work, closed_work, atol=1e-9, rtol=0.)
    torch.testing.assert_close(slide.norm(dim=-1), pads.new_full((1, 2), .4), atol=1e-6, rtol=0.)
    assert force[:, 0].norm().item() > .4 and force[:, 1].count_nonzero().item() == 0

    # Overdrive the damper so that the combined mass, rather than coefficient,
    # determines the impulse. The reconstructed relative speed must not reverse.
    stiff = NeckDamping(mass, pads.new_full((1, 1, 2), .5), dt, NeckDampingParams(1e5, .05))
    stiff_inputs = (*inputs[:6], torch.full_like(normal, 100.), *inputs[7:])
    saturated = stiff.rigid(*stiff_inputs, port_mobility=lambda r, n: port.mobility(state, r, n))
    arm, _, direction, speed = stiff.relative(pads, anchors, pad_rate, anchor_rate, omega_pad)
    rigid_lever = torch.linalg.cross(anchors - anchors.mean(dim=1, keepdim=True), direction)
    parcel_mobility = direction @ direction.mT / mass[:, :, None] + rigid_lever @ inv_inertia @ rigid_lever.mT
    projected = port.mobility(state, arm, direction)
    post = speed - dt * ((projected + parcel_mobility) @ saturated.force.norm(dim=-1, keepdim=True)).squeeze(-1)
    assert projected[:, 0, 0].item() > 10.
    assert -.000001 < post[:, 0].item() < .000001

    return dict(dt_s=dt, rail_speed_m_s=slide.norm(dim=-1)[0].tolist(),
                F_neck_N=force[:, 0].norm().item(), paired_work_mJ=1000 * paired_work.item(),
                wrist_only_work_mJ=1000 * wrist_only_work.item(),
                rail_work_mJ=1000 * rail_work.item(), closed_work_mJ=1000 * closed_work.item(),
                robot_mobility_inverse_kg=projected[0, 0, 0].item(),
                impulse_limited_speed_m_s=post[0, 0].item(),
                wrist_port_pass=bool(closed_work * 1000 <= .1))


def check_current_isolation(robot, sim, dt):
    from src.isaac_yam_parcel.yam_load_port import YamLoadPort

    scene = load_scene()
    rails = robot.rail_ids
    port = YamLoadPort(robot, scene, robot.rail_load.new_zeros((1, 3)))
    q_ref = robot.data.default_joint_pos.torch.clone()
    q_ref[:, rails] = -.022
    robot.reset()
    robot.write_joint_position_to_sim_index(position=q_ref)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_ref))
    sim.forward()
    robot.update(dt)

    ones = q_ref.new_ones((1, 1, 7))
    zero = torch.zeros_like(ones)
    current = JointCurrentEstimatorFilter(JointCurrentParams(
        ones, zero, zero, zero, 15. * ones, zero, torch.zeros_like(ones, dtype=torch.long), zero[..., :1]))
    q_des = q_ref.clone()
    samples = []

    def compress_rails(values):
        return torch.cat((values[:, :6], values[:, rails].mean(dim=-1, keepdim=True)), dim=-1)[:, None]

    for tick in range(350):
        state = port.read()
        # A prescribed static-pinch fixture, applied as external load with zero net wrist wrench.
        axis = torch.nn.functional.normalize(state.pads[:, 1] - state.pads[:, 0], dim=-1)
        squeeze = float(tick >= 75) * 6. * torch.stack((-axis, axis), dim=1)
        wrench = torch.cat((squeeze.sum(dim=1), torch.linalg.cross(
            state.pads - state.pads.mean(dim=1, keepdim=True), squeeze).sum(dim=1)), dim=-1)
        port.write(state, wrench, squeeze)
        robot.actuators.target_command.set_position_index(value=q_des, full_data=True)
        robot.write_data_to_sim()
        q, dq = robot.data.joint_pos.torch, robot.data.joint_vel.torch
        motor = robot.actuators.applied_effort.torch
        opening = (state.pads[:, 1:] - state.pads[:, :1]).norm(dim=-1, keepdim=True)
        obs = current.step(compress_rails(motor), compress_rails(q), compress_rails(dq),
                           compress_rails(q_des), opening)
        if tick >= 250:
            samples.append(torch.stack((motor[:, rails], robot._total_effort[:, rails], q[:, rails], dq[:, rails])))
        sim.step(render=False)
        robot.update(dt)
    history = torch.stack(samples)
    motor_mean, net_mean, q_mean, dq_mean = history.mean(dim=0)
    row = dict(rail_load_N=robot.rail_load[0].tolist(), motor_effort_N=motor_mean[0].tolist(),
                net_effort_N=net_mean[0].tolist(),
                tared_current_equivalent_N=obs.delta_effort[0, 0, -1].item(),
                q_m=q_mean[0].tolist(), dq_m_s=dq_mean[0].tolist(),
                rail_position_span_um=(1e6 * (history[:, 2].amax(dim=0) - history[:, 2].amin(dim=0)))[0].tolist(),
                rail_speed_peak_m_s=history[:, 3].abs().amax(dim=0)[0].tolist())
    print("ROUTE_B_STATIC_PINCH " + json.dumps(row), flush=True)
    torch.testing.assert_close(robot.rail_load, torch.full_like(robot.rail_load, -6.), atol=.001, rtol=0.)
    torch.testing.assert_close(motor_mean, torch.full_like(robot.rail_load, 6.), atol=.03, rtol=0.)
    assert max(row["rail_position_span_um"]) < 30., row
    torch.testing.assert_close(obs.delta_effort[..., -1], ones[..., 0] * 6., atol=.03, rtol=0.)
    return dict(row, pass_isolation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[], livestream=0)
    args = parser.parse_args()
    if not args.headless or args.visualizer or args.livestream:
        parser.error("Use --headless with visualization and livestream disabled.")
    cfg = sim_utils.SimulationCfg(device="cuda:0", dt=.002, gravity=(0., 0., 0.), physics=PhysxCfg(),
                                  use_newton_actuators=False)
    with launch_simulation(cfg, args):
        from src.isaac_yam_parcel.yam_load_port import RailLoadedYam

        sim = sim_utils.SimulationContext(cfg)
        assert not sim.has_gui and not sim.has_offscreen_render
        robot = RailLoadedYam(arm_cfg(0, load_scene(), YAM_USD, "/World/Yam"))
        sim.reset()
        result = measure(robot, sim, cfg.dt)
        print("ROUTE_B_WRIST_POWER " + json.dumps(result), flush=True)
        if not result["wrist_port_pass"]:
            raise SystemExit(1)
        print("ROUTE_B_CURRENT_ISOLATION " + json.dumps(check_current_isolation(robot, sim, cfg.dt)), flush=True)


if __name__ == "__main__":
    main()
