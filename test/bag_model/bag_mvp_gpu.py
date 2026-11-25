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
from bag_mvp import (PEAK_DEPTH, SAFE_JAW_FORCE, geometry, hold_window, longest,
                     make_trial, metrics, probe_command, save_json)
from pe_physics import pair_name
from bag_sim.physical_parcel_gpu import Batch

BLOCK = 20


@wp.kernel
def drive(tick: wp.array[int], command: wp.array3d[float], q_adr: wp.array[int], v_adr: wp.array[int],
          probe: bool, kp: float, kd: float, feedback: bool, dt: float,
          release: float, minimum: float, jaw: wp.array2d[float], qpos: wp.array2d[float], qvel: wp.array2d[float],
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
        time = float(tick[world]) * dt
        use_gap = feedback and time >= .47
        rate = jaw[world, 1] if time < release else .14
        gap = wp.clamp(jaw[world, 0] + rate * dt, minimum, .046) if use_gap else jaw[world, 0]
        if use_gap:
            rate = (gap - jaw[world, 0]) / dt
            jaw[world, 0] = gap
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
                if use_gap:
                    sign = 2.0 * float(side) - 1.0
                    mid = .5 * (command[row, 0, 2] + command[row, 1, 2])
                    speed = .5 * (command[row, 0, 5] + command[row, 1, 5])
                    qpos[world, q + 2] = mid + sign * (.003 + gap / 2.0)
                    qvel[world, v + 2] = speed + sign * rate / 2.0
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


def contact_checkpoint(source, backend, dt):
    accepted = json.loads((source / "summary.json").read_text())
    if accepted["status"] != "PASS" or accepted["case"] != backend.spec.case or accepted["dt_s"] != dt:
        raise ValueError("Contact window requires a passing device trace of the same scene and step")
    with np.load(source / "trace.npz") as trace:
        time = trace["time"]
        index = int(np.argmin(np.abs(time - 1.0)))
        j = backend.m.jnt_qposadr[backend.m.joint("payload_free").id]

        def qpos(i):
            state = backend.d.qpos.copy()
            state[backend.film_qpos] = trace["q"][i] - backend.envelope["reference"]
            state[j:j + 7] = trace["payload"][i]
            state[backend.tool_q[:, None] + np.arange(7)] = trace["fixtures"][i]
            return state

        velocity = np.empty(backend.m.nv)
        mujoco.mj_differentiatePos(backend.m, velocity, time[index + 1] - time[index - 1],
                                  qpos(index - 1), qpos(index + 1))
        return dict(qpos=qpos(index), qvel=velocity, time_s=time[index])


def contact_window_metrics(case, data):
    force = data["wrench"][:, 0, 2]
    geom = not np.any(data["escaped_center"]) and max(data["outside"].max(), data["table_penetration"].max(),
                                                     data["core_penetration"].max()) <= PEAK_DEPTH
    bypass = bool(data["pairs"][:, PAIRS.index("payload_fixture")].max())
    duration = longest(force >= .08, data["time"])
    return dict(case=case, mean_N=float(force.mean()), peak_N=float(force.max()), sustained_s=duration,
                pass_geometry=bool(geom), bypass=bypass,
                status="PASS" if geom and not bypass and duration >= .02 else "FAIL")


def run(case, nworld, output, cpu, dt, contact_from, feedback):
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
    checkpoint = None
    init_start = perf_counter()
    backend = SmokeFlex(spec, mesh_pads, edge, variant)
    if contact_from is not None:
        checkpoint = contact_checkpoint(contact_from, backend, dt)
        i = round(float(checkpoint["time_s"]) / dt)
        commands = commands[i:i + round(.12 / dt)]
    if feedback and variant != "exposed-underface":
        raise ValueError("Closed-loop gap applies only to vertically opposed underface jaws")
    with wp.ScopedDevice("cuda:0"):
        batch = Batch(nworld, parcel=backend, randomize=False, contact_capacity=1024, constraint_capacity=4096)
        tick = wp.zeros(nworld, dtype=int)
        q = wp.array(backend.tool_q, dtype=int)
        v = wp.array(backend.tool_v, dtype=int)
        move = wp.array(commands, dtype=float)
        ids = wp.array(np.arange(batch.d.naconmax), dtype=int)
        local = wp.zeros(batch.d.naconmax, dtype=wp.spatial_vector)
        world = wp.zeros_like(local)
        initial_jaw = np.zeros((nworld, 2), dtype=np.float32)
        initial_jaw[:, 0] = .046
        jaw = wp.array(initial_jaw, dtype=float)

        def step():
            wp.launch(drive, dim=nworld, inputs=[tick, move, q, v, probe, PROBE_K, PROBE_D, feedback, dt,
                                                 hold_window(variant)[1], spec.size[2] - .00075, jaw, batch.d.qpos, batch.d.qvel, batch.d.qfrc_applied])
            batch.step()

        def reset():
            if checkpoint is None:
                batch.reset()
            else:
                batch.reset(initial_qpos=checkpoint["qpos"])
                velocities = np.broadcast_to(checkpoint["qvel"], tuple(batch.d.qvel.shape)).copy()
                batch.d.qvel.assign(velocities)
            tick.zero_()
            jaw.assign(initial_jaw)

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
        interval = BLOCK if feedback else max(BLOCK, round(.010 / dt / BLOCK) * BLOCK)
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
            snapshots = snapshot(batch, total_sim * dt, ids, local, world)
            if feedback and .47 <= total_sim * dt < hold_window(variant)[1]:
                state = jaw.numpy()
                measured = np.array([row["jaw_normal"] for row in snapshots])
                state[:, 1] = np.where(measured.max(axis=1) > SAFE_JAW_FORCE, .16,
                                       np.clip(.012 * (measured.mean(axis=1) - 8.), -.055, .16))
                jaw.assign(state)
            for trace, row in zip(traces, snapshots):
                trace.append(row)
            if not probe and not saved and total_sim * dt >= save_at:
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
            elif contact_from is not None:
                per_world.append(contact_window_metrics(case, d))
            else:
                per_world.append(metrics(case, d, variant, spec.size))
        health = batch.diagnostics()
        success = sum(row["status"] == "PASS" for row in per_world)
        status = "PASS" if success == nworld else "FAIL"
        result = dict(case=case, variant=variant, worlds=nworld, completed_steps=total_sim, expected_steps=len(commands),
                      dt_s=dt, source_command_dt_s=command_dt,
                      scope="moving probe contact" if contact_from else "full action",
                      contact_from=str(contact_from) if contact_from else None, device=str(batch.d.qpos.device), mujoco=mujoco.__version__, warp=wp.__version__,
                      init_s=init_s, warmup_jit_s=warm_s, capture_s=capture_s, physics_wall_s=physics_wall,
                      readback_wall_s=readback_wall, physics_steps_s=total_sim / physics_wall,
                      aggregate_env_steps_s=total_sim * nworld / physics_wall,
                      physics_and_readback_env_steps_s=total_sim * nworld / (physics_wall + readback_wall),
                      simulated_seconds_per_wall_second=total_sim * dt / physics_wall,
                      sampled_peak_process_mib=max((m for m in memory if m is not None), default=None),
                      physics_pass_worlds=success, overflow=health["overflow_bits"], nonfinite=health["nonfinite"],
                      control=("device jaw-force feedback" if feedback else
                               "same replayed CPU jaw commands" if cpu else "same compliant probe drive"),
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
    parser.add_argument("--contact-from", type=Path, help="Benchmark a valid probe's 120 ms moving contact")
    parser.add_argument("--jaw-feedback", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0.:
        parser.error("Time step must be positive and finite")
    if args.contact_from is not None and (args.case != "try-center" or args.jaw_feedback):
        parser.error("Contact-window input is for the compliant probe only")
    if args.jaw_feedback and args.case != "through":
        parser.error("Jaw feedback requires a full through-film replay")
    run(args.case, args.worlds, args.output, args.cpu, args.dt, args.contact_from, args.jaw_feedback)



if __name__ == "__main__":
    main()
