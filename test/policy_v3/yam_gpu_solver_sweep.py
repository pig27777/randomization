"""Measure solver time and capacity use on real Yam_B control frames."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from subprocess import check_output
from time import perf_counter
from unittest.mock import patch

import mujoco
from mujoco_warp._src import solver
from mujoco_warp._src.types import OverflowType
import numpy as np
import warp as wp

from policy_v3.yam_backend import YamBackend


@wp.kernel
def count_contacts(nacon: wp.array(dtype=int), world_id: wp.array(dtype=int),
                   address: wp.array2d(dtype=int), counts: wp.array2d(dtype=int)):
    i = wp.tid()
    if i < nacon[0]:
        world = world_id[i]
        wp.atomic_add(counts, world, 0, 1)
        if address[i, 0] >= 0:
            wp.atomic_add(counts, world, 1, 1)


@wp.kernel
def update_peaks(counts: wp.array2d(dtype=int), nacon: wp.array(dtype=int),
                 nefc: wp.array(dtype=int), peak: wp.array2d(dtype=int)):
    world = wp.tid()
    wp.atomic_max(peak, world, 0, counts[world, 0])
    wp.atomic_max(peak, world, 1, counts[world, 1])
    wp.atomic_max(peak, world, 2, nefc[world])
    if world == 0:
        wp.atomic_max(peak, 0, 3, nacon[0])


@wp.kernel
def update_solver_diagnostics(niter: wp.array(dtype=int), overflow: wp.array(dtype=int),
                              peak: wp.array2d(dtype=int)):
    world = wp.tid()
    wp.atomic_max(peak, world, 4, niter[world])
    wp.atomic_or(peak, world, 5, overflow[world])


def process_memory_mib():
    rows = check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                         "--format=csv,noheader,nounits"], text=True)
    return max((int(mib) for line in rows.splitlines() for pid, mib in [line.split(", ")]
                if pid == str(os.getpid()) and mib.isdigit()), default=None)


def run(args):
    solver_options = {key: getattr(args, key) for key in ("timestep", "iterations", "ls_iterations",
                                                         "tolerance", "solver")
                      if getattr(args, key) is not None}
    backend = YamBackend(7, scene="flex", device="cuda:0", worlds=args.worlds, flex_solver=args.solver_profile,
                         solver_options=solver_options, contact_capacity=args.contacts,
                         constraint_capacity=args.constraints)
    drive = backend.drive
    sim = drive.sim
    d = sim.data
    model_options = drive.model.opt
    config = dict(worlds=args.worlds, profile=args.solver_profile, dt_s=model_options.timestep,
                  iterations=model_options.iterations, ls_iterations=model_options.ls_iterations,
                  xml_tolerance=model_options.tolerance,
                  device_tolerance=float(sim.gpu_model.opt.tolerance.numpy()[0]),
                  solver=mujoco.mjtSolver(model_options.solver).name,
                  integrator=mujoco.mjtIntegrator(model_options.integrator).name,
                  graph_conditional=bool(sim.gpu_model.opt.graph_conditional),
                  substeps=sim.substeps, nconmax_per_world=d.naconmax // args.worlds,
                  naconmax_total=d.naconmax, njmax_per_world=d.njmax, njmax_nnz_per_world=d.njmax_nnz)
    print(json.dumps(config), flush=True)
    counts = wp.zeros((args.worlds, 2), dtype=int, device=sim.device)
    peaks = wp.zeros((args.worlds, 6), dtype=int, device=sim.device)
    # Log the flags once per frame. The device still latches every hard stop.
    sim.gpu_model.opt.warn_overflow &= ~(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
    contacts = d.contact
    before_solve = sim.before_solve
    native_solve = solver.solve
    timing = []

    def measure_before(model, data):
        before_solve(model, data)
        counts.zero_()
        wp.launch(count_contacts, d.naconmax, [d.nacon, contacts.worldid, contacts.efc_address, counts],
                  device=sim.device)
        wp.launch(update_peaks, args.worlds, [counts, d.nacon, d.nefc, peaks], device=sim.device)

    def measure_solver(model, data):
        start = wp.Event(sim.device, enable_timing=True)
        stop = wp.Event(sim.device, enable_timing=True)
        wp.record_event(start, external=True)
        native_solve(model, data)
        wp.record_event(stop, external=True)
        timing.append((start, stop))
        wp.launch(update_solver_diagnostics, args.worlds, [d.solver_niter, d.overflow, peaks], device=sim.device)

    sim.before_solve = measure_before
    positions = np.repeat(drive.initial[None], args.worlds, axis=0)
    commands = np.repeat(np.r_[backend.hover_joints, backend.hover_gap][None], args.worlds, axis=0)
    efforts = np.full(args.worlds, backend.probe_effort_Nm)
    limits = np.tile([3.5, 0., 0.], (args.worlds, 1))
    vision_host = mujoco.MjData(drive.model)
    previous_data = drive.data
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "profile.json"
    try:
        drive.reset_batch(positions)
        t = perf_counter()
        with patch.object(solver, "solve", measure_solver):
            drive.step_batch(commands, efforts, limits)
        wp.synchronize_device(sim.device)
        warmup_s = perf_counter() - t
        if len(timing) != sim.substeps:
            raise RuntimeError("Unexpected number of captured native solver calls")
        warmup_flags = int(np.bitwise_or.reduce(peaks.numpy()[:, 5]))
        for frame in range(args.frames):
            peaks.zero_()
            t = perf_counter()
            drive.step_batch(commands, efforts, limits)
            wp.synchronize_device(sim.device)
            graph_wall_ms = (perf_counter() - t) * 1000
            snapshot = drive.read_batch()
            count = peaks.numpy()
            reasons = snapshot["frame_reason"]
            inset, exposed = 0, 0
            if np.isfinite(snapshot["qpos"]).all() and np.isfinite(snapshot["qvel"]).all():
                drive.data = vision_host
                for world in range(args.worlds):
                    drive.populate_frame(drive, snapshot, world)
                    i, e = backend.flex_core_counts()
                    inset, exposed = max(inset, i), max(exposed, e)
                drive.data = previous_data
            wall_ms = (perf_counter() - t) * 1000
            minimum = snapshot["frame_static_film_min_m"]
            row = dict(frame=frame + 1, wall_ms=wall_ms, graph_wall_ms=graph_wall_ms,
                       solver_ms=sum(wp.get_event_elapsed_time(a, b, synchronize=False) for a, b in timing),
                       max_detected_contacts=int(count[:, 0].max()), max_active_contacts=int(count[:, 1].max()),
                       max_active_constraints=int(count[:, 2].max()), max_total_detected_contacts=int(count[0, 3]),
                       overflow_flags=int(np.bitwise_or.reduce(count[:, 5])),
                       max_solver_iterations=int(count[:, 4].max()), max_edge_ratio=float(snapshot["frame_edge_peak"].max()),
                       min_static_film_m=float(minimum.min()) if np.isfinite(minimum.min()) else None,
                       max_pad_N=float(snapshot["frame_peak"][:, 1:3].max()),
                       max_direct_N=float(snapshot["frame_peak"][:, 4].max()),
                       inset=inset, exposed=exposed, reason_counts=np.bincount(reasons).tolist(),
                       memory_mib=process_memory_mib())
            rows.append(row)
            print(json.dumps(row), flush=True)
            summary = dict(config=config, warmup_s=warmup_s, frames=rows,
                           warmup_overflow_flags=warmup_flags,
                           mean_frame_ms=mean(r["wall_ms"] for r in rows),
                           mean_solver_ms=mean(r["solver_ms"] for r in rows),
                           env_steps_per_s=args.worlds * len(rows) * 1000 / sum(r["wall_ms"] for r in rows))
            output.write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        drive.data = previous_data
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--solver-profile", choices=("nominal", "a2"), default="a2")
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--ls-iterations", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--solver", choices=("Newton", "CG"))
    parser.add_argument("--contacts", type=int)
    parser.add_argument("--constraints", type=int)
    run(parser.parse_args())
