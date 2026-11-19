"""Separate device-step and host-feedback cost for archived YAM poses."""

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


def profile(cfg, qpos, steps, split):
    robot = cpu.scene(cfg, sim_factory=WarpParcelSim)
    sim = robot.sim
    sim.d.qpos[:] = qpos
    sim.d.qvel[:] = 0.
    sim.d.qacc_warmstart[:] = 0.
    mujoco.mj_forward(sim.m, sim.d)
    robot.qtarget = sim.d.qpos[robot.qadr].copy()
    robot.command(sim.d.site_xpos[robot.site].copy(), robot.gap())
    sim.upload()
    sim.step()
    timings = []
    with wp.ScopedDevice(sim.device):
        stages = [("step_ms", sim.graph)]
        if split:
            with wp.ScopedCapture(device=sim.device) as before:
                sim.prepare_step()
            with wp.ScopedCapture(device=sim.device) as after:
                mjw.step2(sim.gpu_model, sim.gpu)
            stages = [("step1_and_bending_ms", before.graph), ("step2_ms", after.graph)]
        for _ in range(steps):
            row = []
            for _, graph in stages:
                started = time.perf_counter()
                wp.capture_launch(graph)
                wp.synchronize_device(sim.device)
                row.append(time.perf_counter() - started)
            stepped = time.perf_counter()
            sim.read_feedback()
            timings.append([*row, time.perf_counter() - stepped])
            if sim.overflow:
                raise RuntimeError(f"Warp capacity overflow bits {sim.overflow}")
            if not np.isfinite(sim.d.qpos).all() or not np.isfinite(sim.d.qvel).all():
                raise RuntimeError("nonfinite GPU state during the timing window")
    sample_ms = np.asarray(timings) * 1000.
    averages = {name: float(sample_ms[:, i].mean()) for i, (name, _) in enumerate(stages)}
    return dict(device=sim.telemetry(), steps=steps, dt_s=cfg.dt,
                feedback_ms=float(sample_ms[:, -1].mean()), feedback_fraction=float(sample_ms[:, -1].sum() / sample_ms.sum()),
                p95_total_ms=float(np.quantile(sample_ms.sum(axis=1), .95)), final_contacts=sim.d.ncon, **averages)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-stages", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = cpu.Config(**json.loads((args.trial / "config.json").read_text()))
    trace = json.loads((args.trial / "trace.json").read_text())
    with np.load(args.trial / "trajectory.npz") as archive:
        if not np.array_equal(archive["time"], [row["t"] for row in trace]):
            raise ValueError("trajectory and force trace time samples do not match")
        data = archive["qpos"]
    samples = {
        "settle": max(i for i, row in enumerate(trace) if row["phase"] == "settle"),
        "closure_entry": next(i for i, row in enumerate(trace) if row["phase"] == "close"),
    }
    result = dict(trial=str(args.trial), mode="fixed command, zero initial velocity",
                  graph_split=args.split_stages, frames={})
    for phase, index in samples.items():
        result["frames"][phase] = dict(archived_time_s=trace[index]["t"],
                                      **profile(cfg, data[index], 64, args.split_stages))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
