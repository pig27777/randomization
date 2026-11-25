"""Replay the same PE manipulation fixture on MJWarp with device contact readback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from bag_mvp import CORNERS, FIXTURES, PAD, PAD_HALF, PAIRS, PROBE_D, PROBE_K, SmokeFlex
from bag_mvp import (PEAK_DEPTH, geometry, hold_window, jaw_core_overlap,
                     make_trial, metrics, probe_command, save_json)
from pe_physics import pair_name
from bag_sim.physical_parcel_gpu import Batch

BLOCK = 20


@wp.kernel
def drive(tick: wp.array[int], command: wp.array3d[float], q_adr: wp.array[int], v_adr: wp.array[int],
          probe: bool, kp: float, kd: float, qpos: wp.array2d[float], qvel: wp.array2d[float],
          applied: wp.array2d[float]):
    world = wp.tid()
    row = wp.min(command.shape[0] - 1, tick[world])
    if probe:
        q = q_adr[0]
        v = v_adr[0]
        qpos[world, q] = command[row, 0, 0]
        qpos[world, q + 1] = 0.
        qvel[world, v] = command[row, 0, 1]
        qvel[world, v + 1] = 0.
        applied[world, v + 2] = kp * (command[row, 0, 2] - qpos[world, q + 2]) - kd * qvel[world, v + 2]
    else:
        for side in range(2):
            if wp.isfinite(command[row, side, 0]):
                q = q_adr[side + 2]
                v = v_adr[side + 2]
                for i in range(3):
                    qpos[world, q + i] = command[row, side, i]
                    qvel[world, v + i] = command[row, side, i + 3]
                    qvel[world, v + 3 + i] = 0.
                for i in range(4):
                    qpos[world, q + 3 + i] = command[row, side, 6 + i]
    tick[world] += 1


def memory_mib():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits"], text=True)
    return max((int(memory) for line in raw.splitlines() for pid, memory in [line.split(",")]
                if pid.strip() == str(os.getpid())), default=None)


def reactions(batch, ids, local, world):
    d = batch.d
    count = min(int(d.nacon.numpy()[0]), d.naconmax)
    forces = np.zeros((batch.nworld, len(FIXTURES) + 2, 6))
    pairs = np.zeros((batch.nworld, len(PAIRS)), dtype=int)
    depth = np.zeros((batch.nworld, len(PAIRS)))
    if not count:
        return forces, pairs, depth
    mjw.contact_force(batch.m, d, ids[:count], False, local[:count])
    mjw.contact_force(batch.m, d, ids[:count], True, world[:count])
    active = np.flatnonzero(local[:count].numpy()[:, 0] > 0.)
    ws = d.contact.worldid[:count].numpy()
    dist = d.contact.dist[:count].numpy()
    geom = d.contact.geom[:count].numpy()
    vectors = world[:count].numpy()
    m = batch.parcel.m
    labels = (*FIXTURES, "table", "payload")
    for i in active:
        names = [m.geom(int(g)).name if g >= 0 else "film" for g in geom[i]]
        tag = pair_name(names)
        w = int(ws[i])
        if tag is not None:
            j = PAIRS.index(tag)
            pairs[w, j] += 1
            depth[w, j] = max(depth[w, j], -dist[i])
        for side, name in enumerate(names):
            if name.startswith("payload"):
                name = "payload"
            if name in labels:
                forces[w, labels.index(name), :3] += (2 * side - 1) * vectors[i, :3]
    return forces, pairs, depth


def snapshot(batch, t, ids, local, world):
    b = batch.parcel
    d = batch.d
    whole_q = d.qpos.numpy()
    positions = b.envelope["reference"][None] + whole_q[:, b.film_qpos]
    piv = b.m.jnt_qposadr[b.m.joint("payload_free").id]
    payloads = whole_q[:, piv:piv + 7]
    fixtures = whole_q[:, b.tool_q[:, None] + np.arange(7)]
    force, pair, depths = reactions(batch, ids, local, world)
    warnings = np.logical_or(d.overflow.numpy() != 0, batch.invalid.numpy() != 0)
    rows = []
    for i in range(batch.nworld):
        q, payload, tools = positions[i], payloads[i], fixtures[i]
        floor, outside, pdepth, escaped, corners = geometry(b, q, payload)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, tools[2, 3:])
        axis = mat.reshape(3, 3)[:, 0]
        jaw = np.maximum(0., [-force[i, 2, :3] @ axis, force[i, 3, :3] @ axis])
        intrusion = 0.
        if b.shelf:
            for tool in tools[2:4]:
                mujoco.mju_quat2Mat(mat, tool[3:])
                points = (CORNERS * [PAD_HALF, *PAD / 2]) @ mat.reshape(3, 3).T + tool[:3]
                if points[:, 2].min() < 0 < points[:, 2].max() + .018:
                    intrusion = max(intrusion, b.edge - float(points[:, 0].min()))
        rows.append(dict(time=t, q=q, payload=payload, fixtures=tools, wrench=force[i],
                         pairs=pair[i], pair_depth=depths[i], floor=floor, outside=outside,
                         core_penetration=pdepth, escaped_center=escaped, outside_corners=corners,
                         table_penetration=max(0., -float(q[:, 2].min())) if not b.shelf else 0.,
                         shelf_intrusion=intrusion, solved_warning=int(warnings[i]), jaw_normal=jaw))
    return rows


def hold_metrics(case, data, size, floor0):
    raised = data["floor"] - floor0 >= .010
    active = np.all(data["jaw_normal"] > 0, axis=1) & np.all(data["pairs"][:, [5, 6]] > 0, axis=1)
    geom = not np.any(data["escaped_center"]) and max(data["outside"].max(),
                                                     data["core_penetration"].max()) <= PEAK_DEPTH
    bypass = bool(jaw_core_overlap(data["payload"], data["fixtures"], size)
                  or data["pairs"][:, PAIRS.index("payload_fixture")].max()
                  or data["shelf_intrusion"].max() > .001)
    passed = bool(np.all(raised) and active.mean() >= .95 and geom and not bypass)
    return dict(case=case, min_lift_mm=float(1000 * (data["floor"].min() - floor0)),
                bilateral_fraction=float(active.mean()), pass_geometry=geom, bypass=bypass,
                status="PASS" if passed else "FAIL")


def run(case, nworld, output, cpu, dt, seconds, hold_from):
    output.mkdir(parents=True, exist_ok=False)
    probe = case.startswith("try")
    variant = "baseline"
    command_dt = dt
    if probe:
        spec = make_trial(case, dt)
        commands = np.zeros((round(spec.duration / dt), 2, 10), dtype=np.float32)
        commands[:, 0, :3] = [probe_command(i * dt) for i in range(len(commands))]
        mesh_pads = True
        edge = .095
    else:
        if cpu is None:
            raise ValueError("GPU grasp replay requires the CPU trajectory directory")
        config = json.loads((cpu / "config.json").read_text())
        if config["case"] != case:
            raise ValueError("GPU replay case must match the recorded CPU trajectory")
        command_dt = config["dt_s"]
        ratio = dt / command_dt
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("GPU time step must be a positive integer multiple of the CPU command step")
        with np.load(cpu / "trace.npz") as data:
            trajectory = data["pad_commands"]
            if len(trajectory) % round(ratio):
                raise ValueError("GPU time step must divide the replay duration")
            commands = trajectory[::round(ratio)].astype(np.float32)
        mesh_pads = config["mesh_pads"]
        edge = config["support_edge"]
        variant = config.get("variant", "baseline")
        spec = make_trial(case, dt, variant)
    full_steps = len(commands)
    checkpoint = None
    if hold_from is not None:
        accepted = json.loads((hold_from / "summary.json").read_text())
        if accepted["status"] != "PASS" or accepted["case"] != case or accepted["dt_s"] != dt:
            raise ValueError("Batch hold needs a passing single-world run of the same fixture and step")
        with np.load(hold_from / "checkpoint.npz") as data:
            checkpoint = dict(data)
        i = round(float(checkpoint["time_s"]) / dt)
        commands = commands[i:i + round(.12 / dt)]
    if seconds is not None:
        commands = commands[:round(seconds / dt)]
    diagnostic = hold_from is None and len(commands) < full_steps
    init_start = perf_counter()
    backend = SmokeFlex(spec, mesh_pads, edge, variant)
    with wp.ScopedDevice("cuda:0"):
        batch = Batch(nworld, parcel=backend, randomize=False, contact_capacity=1024, constraint_capacity=4096)
        tick = wp.zeros(nworld, dtype=int)
        q = wp.array(backend.tool_q, dtype=int)
        v = wp.array(backend.tool_v, dtype=int)
        move = wp.array(commands, dtype=float)
        ids = wp.array(np.arange(batch.d.naconmax), dtype=int)
        local = wp.zeros(batch.d.naconmax, dtype=wp.spatial_vector)
        world = wp.zeros_like(local)

        def step():
            wp.launch(drive, dim=nworld, inputs=[tick, move, q, v, probe, PROBE_K, PROBE_D,
                                                 batch.d.qpos, batch.d.qvel, batch.d.qfrc_applied])
            batch.step()

        def reset():
            if checkpoint is None:
                batch.reset()
            else:
                batch.reset(initial_qpos=checkpoint["qpos"])
                velocities = np.broadcast_to(checkpoint["qvel"], tuple(batch.d.qvel.shape)).copy()
                batch.d.qvel.assign(velocities)
            tick.zero_()

        reset()
        wp.synchronize_device()
        init_s = perf_counter() - init_start
        warm_start = perf_counter()
        step()
        wp.synchronize_device()
        warm_s = perf_counter() - warm_start
        reset()
        capture_start = perf_counter()
        with wp.ScopedCapture() as cap:
            for _ in range(BLOCK):
                step()
        wp.synchronize_device()
        capture_s = perf_counter() - capture_start
        traces = [[row] for row in snapshot(batch, 0., ids, local, world)]
        memory = [memory_mib()]
        total_sim, physics_wall, readback_wall = 0, 0., 0.
        interval = max(BLOCK, round(.010 / dt / BLOCK) * BLOCK)
        save_at = float(np.mean(hold_window(variant)))
        saved = False
        while total_sim < len(commands):
            advance = min(interval, len(commands) - total_sim)
            start = perf_counter()
            for _ in range(advance // BLOCK):
                wp.capture_launch(cap.graph)
            for _ in range(advance % BLOCK):
                step()
            wp.synchronize_device()
            physics_wall += perf_counter() - start
            total_sim += advance
            start = perf_counter()
            for trace, row in zip(traces, snapshot(batch, total_sim * dt, ids, local, world)):
                trace.append(row)
            if not probe and not saved and hold_from is None and total_sim * dt >= save_at:
                np.savez_compressed(output / "checkpoint.npz", qpos=batch.d.qpos.numpy()[0],
                                    qvel=batch.d.qvel.numpy()[0], time_s=total_sim * dt, floor0=traces[0][0]["floor"])
                saved = True
            readback_wall += perf_counter() - start
            if total_sim % (10 * interval) == 0:
                memory.append(memory_mib())
            if total_sim % (30 * interval) == 0:
                print(f"simulated {total_sim * dt:.3f} s in {physics_wall:.2f} s of physics", flush=True)
            if any(trace[-1]["solved_warning"] for trace in traces):
                break
        per_world = []
        for i, trace in enumerate(traces):
            d = {key: np.array([row[key] for row in trace]) for key in trace[0]}
            if i == 0:
                d.update({k: backend.envelope[k] for k in ("reference", "faces", "edges")})
                np.savez_compressed(output / "trace.npz", **d)
            if total_sim != len(commands):
                per_world.append(dict(case=case, status="FAIL", reason="nonfinite state or capacity warning"))
            elif checkpoint is not None:
                per_world.append(hold_metrics(case, d, spec.size, float(checkpoint["floor0"])))
            elif diagnostic:
                per_world.append(dict(case=case, status="NOT TESTED", reason="partial contact replay",
                                      peak_jaw_N=float(d["jaw_normal"].max()),
                                      peak_film_pad_contacts=int(d["pairs"][:, [5, 6]].max()),
                                      peak_payload_lift_mm=1000 * float(d["floor"].max() - d["floor"][0]),
                                      direct_payload_fixture_contacts=int(d["pairs"][:, -1].max())))
            else:
                per_world.append(metrics(case, d, variant, spec.size))
        health = batch.diagnostics()
        success = sum(row["status"] == "PASS" for row in per_world)
        status = "NOT TESTED" if diagnostic and total_sim == len(commands) else "PASS" if success == nworld else "FAIL"
        result = dict(case=case, variant=variant, worlds=nworld, diagnostic_only=diagnostic, completed_steps=total_sim, expected_steps=len(commands),
                      dt_s=dt, source_command_dt_s=command_dt,
                      scope="retained-contact hold" if hold_from else "full action" if not diagnostic else "partial replay",
                      hold_from=str(hold_from) if hold_from else None, device=str(batch.d.qpos.device), mujoco=mujoco.__version__, warp=wp.__version__,
                      init_s=init_s, warmup_jit_s=warm_s, capture_s=capture_s, physics_wall_s=physics_wall,
                      readback_wall_s=readback_wall, physics_steps_s=total_sim / physics_wall,
                      aggregate_env_steps_s=total_sim * nworld / physics_wall,
                      physics_and_readback_env_steps_s=total_sim * nworld / (physics_wall + readback_wall),
                      simulated_seconds_per_wall_second=total_sim * dt / physics_wall,
                      sampled_peak_process_mib=max((m for m in memory if m is not None), default=None),
                      physics_pass_worlds=success, overflow=health["overflow_bits"], nonfinite=health["nonfinite"],
                      control="same replayed CPU jaw commands" if cpu else "same compliant probe drive",
                      valid_aggregate_env_steps_s=total_sim * success / physics_wall,
                      status=status, per_world=per_world)
        save_json(output / "summary.json", result)
        print({key: value for key, value in result.items() if key != "per_world"}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("case", choices=("try-center", "film", "through"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds", type=int, choices=(1, 16, 64, 256), default=1)
    parser.add_argument("--dt", type=float, default=.00005)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--seconds", type=float, help="Limit a single-world contact diagnostic")
    parser.add_argument("--hold-from", type=Path, help="Benchmark a valid GPU grasp's 120 ms hold window")
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0.:
        parser.error("Time step must be positive and finite")
    if args.hold_from is not None and (args.case.startswith("try") or args.seconds is not None or args.cpu is None):
        parser.error("A hold window requires a matching recorded CPU grasp and a passing GPU checkpoint")
    if args.seconds is not None and (args.worlds != 1 or not np.isfinite(args.seconds) or args.seconds < args.dt):
        parser.error("A positive replay limit requires one world")
    run(args.case, args.worlds, args.output, args.cpu, args.dt, args.seconds, args.hold_from)


if __name__ == "__main__":
    main()
