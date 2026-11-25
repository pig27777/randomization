"""Controlled Flex contact diagnostics from previously saved physical states."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from pe_physics import MAX_PENETRATION, PAIRS, FlexBagBackend, save_json
from pe_physics_protocol import MeshMetrics, action, phase_one, sample


def run(label, original, path, first, last, *, dt=None, self_time=.001, collision=True):
    trial = replace(original, dt=dt or original.dt, self_contact_time=self_time, self_collision=collision)
    with np.load(path) as trace:
        times = trace[f"{trial.case}/time"]
        start = int(np.argmin(abs(times - first)))
        t0 = float(times[start])
        q, v = (trace[f"{trial.case}/{k}"][start] for k in ("q", "v"))
    backend = FlexBagBackend.reset(trial)
    backend.d.qpos[backend.film_qpos] = q - backend.envelope["reference"]
    backend.d.qvel[backend.film_dofs] = v
    backend.d.time = t0
    mujoco.mj_forward(backend.m, backend.d)
    metric = MeshMetrics(backend.envelope, not trial.coupon)
    timeline = []
    for step in range(1, round((last - t0) / trial.dt) + 1):
        frame = action(trial, t0 + (step - 1) * trial.dt)
        backend.step(frame)
        if step % round(.001 / trial.dt) == 0:
            timeline.append(sample(backend, trial, metric, t0 + step * trial.dt, frame, 1.))
    arrays = {k: np.asarray([row[k] for row in timeline]) for k in timeline[0]}
    cross = arrays["self_crossings"]
    late = arrays["time"] >= last - (.10 if trial.case == "B01" else .024) - .1 * trial.dt
    z_ratio = (np.sum(arrays["v"][late, :, 2]**2 * metric.masses) /
               np.sum(arrays["v"][late]**2 * metric.masses[None, :, None]))
    row = dict(label=label, dt_s=trial.dt, self_time_s=self_time, self_enabled=collision,
               case=trial.case, start_s=t0, last_s=float(arrays["time"][-1]),
               tail_peak_rms_mm_s=float(arrays["rms_speed"][late].max() * 1000),
               vertical_kinetic_fraction=float(z_ratio), crossing_pairs=int(cross.max()),
               first_crossing_s=float(arrays["time"][cross > 0][0]) if cross.any() else None,
               peak_table_depth_mm=float(arrays["table_depth"].max() * 1000),
               peak_solver_depth_mm=float(arrays["pair_depth"].max() * 1000),
               peak_self_force_N=float(arrays["pair_normal"][:, PAIRS.index("film_self")].max()),
               penetration_gate=bool(max(arrays["table_depth"].max(), arrays["pair_depth"].max()) <= MAX_PENETRATION),
               solver_warnings=int(arrays["warnings"].max()))
    return row, arrays


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--case", choices=("B01", "B04-U"), required=True)
    ap.add_argument("--variant", choices=("baseline", "half_step", "softened", "softened_3ms", "softened_4ms", "self_off"), required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    branch = dict(baseline={}, half_step={"dt": .0000125}, softened={"self_time": .002},
                  softened_3ms={"self_time": .003}, softened_4ms={"self_time": .004}, self_off={"collision": False})[args.variant]
    trial = next(t for t in phase_one(None) if t.case == args.case)
    first, last = (1.08, 1.20) if args.case == "B01" else (.200, .270)
    result, arrays = run(args.variant, trial, args.source, first, last, **branch)
    result["source"] = str(args.source)
    np.savez_compressed(args.output / "raw_trajectories.npz", **arrays)
    save_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
