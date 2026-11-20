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


SCALING_WORLDS = (1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 12000)
SCALING_CAPACITY = dict(nconmax=384, njmax=1200, njmax_nnz=24576)
COMPACT_CAPACITY = dict(nconmax=256, njmax=768, njmax_nnz=16384)
EPISODE_SECONDS = 4.77


def profile(trial, phase, worlds, *, scaling_capacity=None, eager=False):
    scaling = scaling_capacity is not None
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
    gap_command = trace[index]["command_gap"] if scaling else robot.gap()
    robot.command(data.site_xpos[robot.site].copy(), gap_command)
    if scaling and (sim.m.nflexvert != 142 or config.dt != .0002):
        raise ValueError("Scaling scan requires the 142-node, 200 µs reference")
    sim.device = wp.get_device("cuda:0")
    sim.overflow = 0
    steps = 64
    with wp.ScopedDevice(sim.device):
        model = mjw.put_model(sim.m)
        capacity = scaling_capacity if scaling else dict(nconmax=2048, njmax=8192)
        print(f"Allocating {worlds} worlds with {capacity}", flush=True)
        gpu = mjw.put_data(sim.m, data, nworld=worlds, naconmax=capacity["nconmax"] * worlds, **capacity)
        print("World buffers allocated", flush=True)
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

        if scaling:
            memory_samples = [sim.telemetry()["process_memory_mib"]]
            print(f"Process memory after allocation: {memory_samples[-1]} MiB", flush=True)
        step()
        advance = step
        if not eager:
            with wp.ScopedCapture(device=sim.device) as capture:
                step()

            def advance():
                wp.capture_launch(capture.graph)

        def measure():
            for metric in (bits, invalid, iterations, contacts, penetration):
                metric.zero_()
            wp.synchronize_device(sim.device)
            started = time.perf_counter()
            for _ in range(steps):
                advance()
            wp.synchronize_device(sim.device)
            elapsed = time.perf_counter() - started
            sim.overflow = int(np.bitwise_or.reduce(bits.numpy()))
            invalid_count = int(np.count_nonzero(invalid.numpy()))
            peak = float(penetration.numpy().max())
            counts = contacts.numpy()
            return dict(phase=phase, archived_time_s=trace[index]["t"], worlds=worlds, steps_per_world=steps,
                        dt_s=config.dt, wall_s=elapsed, env_steps_per_s=worlds * steps / elapsed,
                        mean_contacts_per_world_step=float(counts.sum() / (worlds * steps)),
                        peak_solver_iterations=int(iterations.numpy().max()), peak_penetration_m=peak,
                        invalid_worlds=invalid_count, device=sim.telemetry(),
                        valid=not invalid_count and not sim.overflow and peak <= .001,
                        contact_count_range_per_world=[int(counts.min()), int(counts.max())])

        if not scaling:
            return measure()

        for _ in range(8):
            advance()
        wp.synchronize_device(sim.device)
        memory_samples.append(sim.telemetry()["process_memory_mib"])
        warmup_overflow = int(np.bitwise_or.reduce(bits.numpy()))
        payload = int(sim.m.joint("payload_free").qposadr[0])
        repeats = []
        for seed in (20261120, 20261121, 20261122):
            print(f"Timing {phase} with seed {seed}", flush=True)
            jitter = np.random.default_rng(seed).uniform(-1., 1., (worlds, 3))
            qpos = np.broadcast_to(pose, gpu.qpos.shape).astype(np.float32)
            ctrl = np.broadcast_to(data.ctrl, gpu.ctrl.shape).astype(np.float32)
            qpos[:, payload:payload + 2] += 2.5e-5 * jitter[:, :2]
            ctrl[:, robot.gripper] = np.clip(ctrl[:, robot.gripper] + 2.5e-5 * jitter[:, 2], -.005, .0475)
            mjw.reset_data(model, gpu)
            gpu.qpos.assign(qpos)
            gpu.ctrl.assign(ctrl)
            result = measure()
            clocks = gpu.time.numpy()
            if not np.allclose(clocks, steps * config.dt, rtol=0., atol=1e-7):
                raise RuntimeError(f"Expected every world to advance {steps * config.dt:.6f} s")
            rate = result["env_steps_per_s"] * config.dt
            result.update(seed=seed, sim_sec_per_wall_sec=rate,
                          simulated_time_range_s=[float(clocks.min()), float(clocks.max())],
                          episode_equivalents_per_hour=3600 * rate / EPISODE_SECONDS)
            repeats.append(result)
            memory_samples.append(result["device"]["process_memory_mib"])
            if sim.overflow:
                break

        valid = not warmup_overflow and len(repeats) == 3 and all(row["valid"] for row in repeats)
        median = float(np.median([row["env_steps_per_s"] for row in repeats])) if valid else None
        return dict(phase=phase, worlds=worlds, dt_s=config.dt, capacity=capacity, repeats=repeats,
                    execution="eager" if eager else "cuda_graph",
                    valid=valid, median_env_steps_per_s=median,
                    sim_sec_per_wall_sec=median * config.dt if valid else None,
                    episode_equivalents_per_hour=3600 * median * config.dt / EPISODE_SECONDS if valid else None,
                    peak_process_memory_mib=max((v for v in memory_samples if v is not None), default=None),
                    peak_penetration_m=max(row["peak_penetration_m"] for row in repeats),
                    warmup_overflow_bits=warmup_overflow,
                    overflow_bits=warmup_overflow | int(np.bitwise_or.reduce(
                        [row["device"]["overflow_bits"] for row in repeats])))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", required=True, choices=("settle", "close", "lift_hold"))
    parser.add_argument("--worlds", required=True, type=int, choices=SCALING_WORLDS)
    parser.add_argument("--scaling", action="store_true", help="Run three perturbed windows for the scaling scan")
    parser.add_argument("--compact-capacity", action="store_true",
                        help="Use the contact and constraint capacities of the 8,000-world probe")
    parser.add_argument("--ccd-capacity", type=int, help="Set the CCD scratch slots per world")
    parser.add_argument("--eager", action="store_true", help="Launch GPU steps directly without capturing a graph")
    args = parser.parse_args()
    if not args.scaling and args.worlds not in (1, 16, 64):
        parser.error("world counts above 64 require --scaling")
    if args.compact_capacity and not args.scaling:
        parser.error("--compact-capacity requires --scaling")
    if args.output.exists():
        raise FileExistsError(args.output)
    capacity = None
    if args.scaling:
        capacity = (COMPACT_CAPACITY if args.compact_capacity else SCALING_CAPACITY).copy()
    if args.ccd_capacity is not None:
        if capacity is None or not 1 <= args.ccd_capacity <= capacity["nconmax"]:
            parser.error("--ccd-capacity requires --scaling and must be within the contact capacity")
        capacity["nccdmax"] = args.ccd_capacity
    result = profile(args.trial, args.phase, args.worlds, scaling_capacity=capacity, eager=args.eager)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
