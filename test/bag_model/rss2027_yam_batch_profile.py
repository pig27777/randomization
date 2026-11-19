"""Measure fixed-command device throughput at archived whole-YAM contact poses."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import physical_parcel_yam as cpu
from physical_parcel_yam_warp import WarpParcelSim
from bag_sim.physical_parcel_gpu import bending, contact_health, state_health


def profile(trial, phase, worlds):
    config = cpu.Config(**json.loads((trial / "config.json").read_text()))
    trace = json.loads((trial / "trace.json").read_text())
    indices = [i for i, row in enumerate(trace) if row["phase"] == phase]
    index = indices[len(indices) // 2] if phase == "lift_hold" else indices[-1]
    with np.load(trial / "trajectory.npz") as archive:
        if not np.array_equal(archive["time"], [row["t"] for row in trace]):
            raise ValueError("Archived configurations and feedback samples are misaligned")
        pose = archive["qpos"][index]
    robot = cpu.scene(config, sim_factory=WarpParcelSim)
    sim = robot.sim
    data, parcel = sim.d, sim.parcel
    data.qpos[:] = pose
    data.qvel[:] = 0.
    mujoco.mj_forward(sim.m, data)
    robot.qtarget = data.qpos[robot.qadr].copy()
    robot.command(data.site_xpos[robot.site].copy(), robot.gap())
    sim.device = wp.get_device("cuda:0")
    sim.overflow = 0
    steps = 64
    with wp.ScopedDevice(sim.device):
        model = mjw.put_model(sim.m)
        gpu = mjw.put_data(sim.m, data, nworld=worlds, nconmax=2048, naconmax=2048 * worlds, njmax=8192)
        film_inputs = [gpu.flexvert_xpos, gpu.qvel, wp.array(sim.film_dofs[:, 0], dtype=int),
                       wp.array(parcel.hinges, dtype=wp.vec4i), wp.array(parcel.rest_angles, dtype=float),
                       wp.array(parcel.hinge_stiffness, dtype=float), wp.array(parcel.hinge_damping, dtype=float),
                       gpu.qfrc_passive]
        bits, invalid, iterations, contacts = (wp.zeros(worlds, dtype=int) for _ in range(4))
        penetration = wp.zeros(worlds, dtype=float)

        def step():
            mjw.step1(model, gpu)
            wp.launch(bending, (worlds, len(parcel.hinges)), film_inputs)
            mjw.step2(model, gpu)
            wp.launch(state_health, worlds, [gpu.qpos, gpu.qvel, gpu.overflow,
                                            gpu.solver_niter, bits, invalid, iterations])
            wp.launch(contact_health, gpu.naconmax, [gpu.nacon, gpu.contact.worldid, gpu.contact.dist,
                                                    contacts, penetration])

        step()
        with wp.ScopedCapture(device=sim.device) as capture:
            step()
        for metric in (bits, invalid, iterations, contacts, penetration):
            metric.zero_()
        wp.synchronize_device(sim.device)
        started = time.perf_counter()
        for _ in range(steps):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(sim.device)
        elapsed = time.perf_counter() - started
        sim.overflow = int(np.bitwise_or.reduce(bits.numpy()))
        invalid_count = int(np.count_nonzero(invalid.numpy()))
        peak = float(penetration.numpy().max())
        return dict(phase=phase, archived_time_s=trace[index]["t"], worlds=worlds, steps_per_world=steps,
                    dt_s=config.dt, wall_s=elapsed, env_steps_per_s=worlds * steps / elapsed,
                    mean_contacts_per_world_step=float(contacts.numpy().sum() / (worlds * steps)),
                    peak_solver_iterations=int(iterations.numpy().max()), peak_penetration_m=peak,
                    invalid_worlds=invalid_count, device=sim.telemetry(),
                    valid=not invalid_count and not sim.overflow and peak <= .001)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", required=True, choices=("settle", "close", "lift_hold"))
    parser.add_argument("--worlds", required=True, type=int, choices=(1, 16, 64))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = profile(args.trial, args.phase, args.worlds)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
