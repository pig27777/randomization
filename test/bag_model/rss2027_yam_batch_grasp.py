"""Benchmark replicated full YAM grasps with a live reference controller."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import physical_parcel_yam as cpu
import physical_parcel_yam_warp as backend
from bag_sim.physical_parcel_gpu import bending, state_health
from yam_platform.protocol import run


@wp.kernel
def film_penetration(nacon: wp.array(dtype=int), world: wp.array(dtype=int),
                     flex: wp.array(dtype=wp.vec2i), address: wp.array2d(dtype=int),
                     distance: wp.array(dtype=float), peak: wp.array(dtype=float)):
    i = wp.tid()
    if i < wp.min(nacon[0], world.shape[0]):
        if address[i, 0] >= 0 and (flex[i][0] == 0 or flex[i][1] == 0):
            wp.atomic_max(peak, world[i], wp.max(-distance[i], 0.0))


@wp.kernel
def hold_reactions(nacon: wp.array(dtype=int), world: wp.array(dtype=int), flex: wp.array(dtype=wp.vec2i),
                   geom: wp.array(dtype=wp.vec2i), address: wp.array2d(dtype=int), dim: wp.array(dtype=int),
                   frame: wp.array(dtype=wp.mat33), force: wp.array2d(dtype=float),
                   positions: wp.array2d(dtype=wp.vec3), roles: wp.array(dtype=int), left: int, right: int,
                   reactions: wp.array2d(dtype=float)):
    i = wp.tid()
    if i >= wp.min(nacon[0], world.shape[0]) or address[i, 0] < 0 or (flex[i][0] != 0 and flex[i][1] != 0):
        return
    w = world[i]
    local = wp.vec3(0.0)
    for axis in range(wp.min(dim[i], 3)):
        local[axis] = force[w, address[i, axis]]
    total = wp.transpose(frame[i]) * local
    closing = wp.normalize(positions[w, right] - positions[w, left])
    for side in range(2):
        g = geom[i][side]
        if g >= 0:
            role = roles[g]
            sign = 2.0 * float(side) - 1.0
            if role == 1:
                wp.atomic_add(reactions, w, 0, -sign * total[2])
            elif role == 2:
                wp.atomic_add(reactions, w, 1, -sign * wp.dot(total, closing))
            elif role == 3:
                wp.atomic_add(reactions, w, 2, sign * wp.dot(total, closing))


@wp.kernel
def box_corners_inside(vertices: wp.array2d(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i),
                       positions: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.mat33),
                       geom: int, half: wp.vec3, outside: wp.array(dtype=int)):
    w, corner = wp.tid()
    sign = wp.vec3(float(2 * (corner & 1) - 1), float(2 * ((corner >> 1) & 1) - 1),
                   float(2 * ((corner >> 2) & 1) - 1))
    p = positions[w, geom] + rotations[w, geom] * wp.cw_mul(half, sign)
    angle = float(0.0)
    for i in range(faces.shape[0]):
        f = faces[i]
        a, b, c = vertices[w, f[0]] - p, vertices[w, f[1]] - p, vertices[w, f[2]] - p
        la, lb, lc = wp.length(a), wp.length(b), wp.length(c)
        angle += wp.atan2(wp.dot(a, wp.cross(b, c)),
                          la * lb * lc + wp.dot(a, b) * lc + wp.dot(b, c) * la + wp.dot(c, a) * lb)
    if wp.abs(angle) < 3.14159265:
        wp.atomic_max(outside, w, 1)


@wp.func
def separates(axis: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, half: wp.vec3) -> bool:
    length = wp.length(axis)
    if length <= 1.0e-14:
        return False
    direction = axis / length
    radius = wp.dot(wp.vec3(wp.abs(direction[0]), wp.abs(direction[1]), wp.abs(direction[2])), half)
    pa, pb, pc = wp.dot(a, direction), wp.dot(b, direction), wp.dot(c, direction)
    return wp.min(pa, wp.min(pb, pc)) >= radius or wp.max(pa, wp.max(pb, pc)) <= -radius


@wp.kernel
def reject_inset_intersection(vertices: wp.array2d(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i),
                              positions: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.mat33),
                              geom: int, half: wp.vec3, intersects: wp.array(dtype=int)):
    w, face = wp.tid()
    f, center = faces[face], positions[w, geom]
    rotation = wp.transpose(rotations[w, geom])
    a, b, c = rotation * (vertices[w, f[0]] - center), rotation * (vertices[w, f[1]] - center), rotation * (vertices[w, f[2]] - center)
    e0, e1, e2 = b - a, c - b, a - c
    ex, ey, ez = wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)
    if (separates(ex, a, b, c, half) or separates(ey, a, b, c, half) or separates(ez, a, b, c, half)
        or separates(wp.cross(e0, e1), a, b, c, half)):
        return
    for axis in range(3):
        unit = wp.vec3(0.0)
        unit[axis] = 1.0
        if separates(wp.cross(e0, unit), a, b, c, half) or separates(wp.cross(e1, unit), a, b, c, half) or separates(wp.cross(e2, unit), a, b, c, half):
            return
    wp.atomic_max(intersects, w, 1)


class Replicas:
    def __init__(self, sim, worlds, contacts, rows, nnz):
        self.sim, self.worlds = sim, worlds
        self.device = sim.device
        self.steps = 0
        self.graph = None
        with wp.ScopedDevice(self.device):
            self.gpu = mjw.put_data(sim.m, sim.d, nworld=worlds, nconmax=contacts,
                                    naconmax=contacts * worlds, njmax=rows, njmax_nnz=nnz)
            film = sim.parcel
            self.film_inputs = [self.gpu.flexvert_xpos, self.gpu.qvel, wp.array(sim.film_dofs[:, 0], dtype=int),
                                wp.array(film.hinges, dtype=wp.vec4i), wp.array(film.rest_angles, dtype=float),
                                wp.array(film.hinge_stiffness, dtype=float), wp.array(film.hinge_damping, dtype=float),
                                self.gpu.qfrc_passive]
            self.bits, self.invalid, self.iterations = (wp.zeros(worlds, dtype=int) for _ in range(3))
            self.penetration = wp.zeros(worlds, dtype=float)

    def prepare(self):
        sim, gpu = self.sim, self.gpu
        mjw.step1(sim.gpu_model, gpu)
        wp.launch(bending, (self.worlds, len(sim.parcel.hinges)), self.film_inputs)

    def advance(self):
        sim, gpu = self.sim, self.gpu
        self.prepare()
        mjw.step2(sim.gpu_model, gpu)
        wp.launch(state_health, self.worlds, [gpu.qpos, gpu.qvel, gpu.overflow, gpu.solver_niter,
                                            self.bits, self.invalid, self.iterations])
        wp.launch(film_penetration, gpu.naconmax, [gpu.nacon, gpu.contact.worldid, gpu.contact.flex,
                                                 gpu.contact.efc_address, gpu.contact.dist, self.penetration])

    def step(self, ctrl):
        with wp.ScopedDevice(self.device):
            self.gpu.ctrl.assign(np.broadcast_to(ctrl, (self.worlds, len(ctrl))).astype(np.float32))
            if self.graph is None:
                self.advance()
                with wp.ScopedCapture(device=self.device) as capture:
                    self.advance()
                self.graph = capture.graph
            else:
                wp.capture_launch(self.graph)
        self.steps += 1

    def health(self):
        return dict(overflow_bits=int(np.bitwise_or.reduce(self.bits.numpy())),
                    invalid_worlds=int(np.count_nonzero(self.invalid.numpy())),
                    peak_penetration_mm=1000. * float(self.penetration.numpy().max()),
                    peak_solver_iterations=int(self.iterations.numpy().max()))


class GraspChecks:
    def __init__(self, robot, batch):
        sim = robot.sim
        self.batch = batch
        self.faces = wp.array(sim.parcel.envelope["faces"], dtype=wp.vec3i, device=batch.device)
        self.left, self.right = map(int, robot.pad_geom)
        self.payload = sim.m.geom("payload_geom_0").id
        self.body = int(sim.payload_bodies[0])
        self.boxes = [self.payload, sim.table_geom]
        roles = np.zeros(sim.m.ngeom, dtype=np.int32)
        roles[sim.table_geom] = 1
        if sim.floor_geom >= 0:
            roles[sim.floor_geom] = 1
        for geom_ids, role in zip(robot.all_finger_sets, (2, 3), strict=True):
            roles[list(geom_ids)] = role
        self.roles = wp.array(roles, dtype=int, device=batch.device)
        self.reactions = wp.zeros((batch.worlds, 3), dtype=float, device=batch.device)
        self.outside = wp.zeros(batch.worlds, dtype=int, device=batch.device)
        self.intersects = wp.zeros(batch.worlds, dtype=int, device=batch.device)
        self.baseline_z = float(sim.com()[2])

    def sample(self, hold):
        batch = self.batch
        gpu, sim = batch.gpu, batch.sim
        with wp.ScopedDevice(batch.device):
            batch.prepare()
            wp.launch(box_corners_inside, (batch.worlds, 8), [gpu.flexvert_xpos, self.faces, gpu.geom_xpos, gpu.geom_xmat,
                      self.payload, wp.vec3(*sim.m.geom_size[self.payload]), self.outside])
            for geom in self.boxes:
                half = wp.vec3(*(sim.m.geom_size[geom] - .001))
                wp.launch(reject_inset_intersection, (batch.worlds, len(self.faces)),
                          [gpu.flexvert_xpos, self.faces, gpu.geom_xpos, gpu.geom_xmat, geom, half, self.intersects])
            if hold:
                mjw.fwd_actuation(sim.gpu_model, gpu)
                mjw.fwd_acceleration(sim.gpu_model, gpu)
                mjw.solve(sim.gpu_model, gpu)
                self.reactions.zero_()
                wp.launch(hold_reactions, gpu.naconmax,
                          [gpu.nacon, gpu.contact.worldid, gpu.contact.flex, gpu.contact.geom, gpu.contact.efc_address,
                           gpu.contact.dim, gpu.contact.frame, gpu.efc.force, gpu.geom_xpos, self.roles,
                           self.left, self.right, self.reactions])
        geometry_ok = (self.outside.numpy() == 0) & (self.intersects.numpy() == 0)
        if not hold:
            return geometry_ok, None, {}
        forces = self.reactions.numpy()
        rise = gpu.xipos.numpy()[:, self.body, 2] - self.baseline_z
        valid = (forces[:, 0] <= .05) & (forces[:, 1:].min(axis=1) > .001) & (rise > 0.)
        return geometry_ok, valid, dict(hold_min_rise_m=float(rise.min()), hold_max_support_N=float(forces[:, 0].max()),
                                        hold_min_inner_N=float(forces[:, 1:].min()))


def create_batch(sim, args):
    return Replicas(sim, args.worlds, args.contact_capacity, args.constraint_capacity, args.sparse_capacity)


def capacity_probe(cfg, args):
    robot = cpu.scene(replace(cfg, initial_state=None), sim_factory=backend.WarpParcelSim)
    trace = json.loads((args.trial / "trace.json").read_bytes())
    frames = [i for i, row in enumerate(trace) if row["phase"] == "lift_hold"]
    with np.load(args.trial / "trajectory.npz") as archive:
        robot.sim.d.qpos[:] = archive["qpos"][frames[len(frames) // 2]]
    robot.sim.d.qvel[:] = 0.
    mujoco.mj_forward(robot.sim.m, robot.sim.d)
    robot.qtarget = robot.sim.d.qpos[robot.qadr].copy()
    robot.command(robot.sim.d.site_xpos[robot.site].copy(), robot.gap())
    robot.sim.upload()
    print("allocating", args.worlds, "replicas", flush=True)
    batch = create_batch(robot.sim, args)
    print("allocated", robot.sim.telemetry(), flush=True)
    ctrl = robot.sim.d.ctrl.copy()
    batch.step(ctrl)
    wp.synchronize_device(batch.device)
    start = time.perf_counter()
    for _ in range(64):
        batch.step(ctrl)
    wp.synchronize_device(batch.device)
    elapsed = time.perf_counter() - start
    metrics = batch.health()
    rate = args.worlds * 64 / elapsed
    return dict(steps_per_world=64, simulated_s_per_world=64 * cfg.dt, wall_s=elapsed,
                env_steps_per_s=rate, sim_seconds_per_wall_second=rate * cfg.dt, device=robot.sim.telemetry(), **metrics)


def full_grasp(cfg, args):
    batch = None
    checks = None
    samples = []
    last_phase = None
    peak_device = 0
    max_error = 0.
    started = time.perf_counter()
    agrees = np.ones(args.worlds, dtype=bool)
    enclosed = np.ones(args.worlds, dtype=bool)
    held = np.ones(args.worlds, dtype=bool)

    class BatchedSim(backend.WarpParcelSim):
        def step(self):
            batch.step(self.d.ctrl)
            super().step()

    def scene(config):
        nonlocal batch, checks
        robot = cpu.scene(config, sim_factory=BatchedSim)
        robot.sim.upload()
        batch = create_batch(robot.sim, args)
        checks = GraspChecks(robot, batch)
        return robot

    def sample(robot, row):
        nonlocal last_phase, peak_device, max_error
        error = np.max(np.abs(batch.gpu.qpos.numpy() - robot.sim.d.qpos), axis=1)
        agrees[:] &= error <= 1e-5
        geometry, hold, hold_metrics = checks.sample(row["phase"] == "lift_hold")
        enclosed[:] &= geometry
        if hold is not None:
            held[:] &= hold
        max_error = max(max_error, float(error.max()))
        samples.append(dict(time_s=row["t"], phase=row["phase"], largest_generalized_coordinate_difference=float(error.max()),
                            replicas_within_1e5=int(np.count_nonzero(error <= 1e-5))))
        metrics = batch.health()
        status = dict(**samples[-1], steps_per_world=batch.steps, wall_s=time.perf_counter() - started,
                      reference_active_film_penetration_mm=row["penetration_mm"], enclosed_worlds=int(enclosed.sum()),
                      holding_worlds=int(held.sum()) if hold is not None else None, **hold_metrics, **metrics)
        if last_phase != row["phase"] or len(samples) % 10 == 1:
            tele = robot.sim.telemetry()
            peak_device = max(peak_device, tele["process_memory_mib"] or 0)
            last_phase = row["phase"]
            print(json.dumps({**tele, **status}), flush=True)
        temporary = args.output / "last-sample.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output / "last-sample.json")
        if (metrics["overflow_bits"] or metrics["invalid_worlds"]
                or (args.stop_on_penetration and metrics["peak_penetration_mm"] > 1.)):
            raise ValueError(f"Invalid batched dynamics at {row['t']:.6f} s in {row['phase']}: {metrics}")

    with patch.object(backend, "scene", scene):
        result = run(backend, cfg, args.output / "trial", on_sample=sample)
    elapsed = result["wall_s"]
    metrics = batch.health()
    base = json.loads((args.trial.parent.parent / "benchmark.json").read_bytes())
    trace = json.loads((args.output / "trial/trace.json").read_bytes())
    hold = [row for row in trace if row["phase"] == "lift_hold"]
    duration = sum(item["end_s"] - item["start_s"] for item in result["phases"] if item["phase"] == "lift_hold")
    leader_passed = bool(result["outcome"] == "pickup_ok" and hold and not result["warnings"]
                         and duration >= .20 - cfg.dt / 2.
                         and all(row["payload_geometry"]["outside_corners"] == 0
                                 and row["payload_geometry"]["intersecting_triangles"] == 0 for row in trace)
                         and max(row["support_N"] for row in hold) <= .05
                         and min(min(row["grip_each_N"]) for row in hold) > 0.)
    within_penetration = batch.penetration.numpy() <= .001
    verified = leader_passed and not metrics["overflow_bits"] and not metrics["invalid_worlds"]
    valid_count = int((enclosed & held & within_penetration).sum()) if verified else 0
    rate = args.worlds * batch.steps / elapsed
    return dict(steps_per_world=batch.steps, simulated_s_per_world=result["sim_time_s"], wall_s=elapsed,
                outcome=result["outcome"], reference_grasp_verified=leader_passed, replicas_verified=valid_count == args.worlds,
                verified_pickups=valid_count, largest_generalized_coordinate_difference=max_error,
                replicas_matching_all_samples=int(agrees.sum()),
                env_steps_per_s=rate, sim_seconds_per_wall_second=rate * cfg.dt,
                pickups_per_hour=3600. * valid_count / elapsed,
                aggregate_speedup_vs_archive=valid_count * base["wall_s"] / elapsed,
                sampled_peak_gpu_memory_mib=peak_device, state_samples=samples, **metrics)


def transplant_prepared_state(cfg, output):
    parcel, _ = cpu.parcel_world(cfg)
    signature = cpu.prepared_signature(parcel, cfg)
    with np.load(cfg.initial_state) as original:
        previous = json.loads(original["config_json"].item())
        current = json.loads(signature)
        before_dt = previous["model"].pop("timestep")
        after_dt = current["model"].pop("timestep")
        if previous != current or before_dt == after_dt:
            raise ValueError("Only a different timestep may reuse this exact prepared state")
        path = output / "same_prepared_state.npz"
        np.savez_compressed(path, qpos=original["qpos"], qvel=original["qvel"], config_json=np.asarray(signature))
    return replace(cfg, initial_state=str(path.resolve()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--step-us", type=int, choices=(200, 400, 5000, 10000), default=200)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-on-penetration", action="store_true", help="stop an unstable timestep study at its first sampled excess")
    parser.add_argument("--contact-capacity", type=int, default=256)
    parser.add_argument("--constraint-capacity", type=int, default=768)
    parser.add_argument("--sparse-capacity", type=int, default=16384)
    args = parser.parse_args()
    if min(args.worlds, args.contact_capacity, args.constraint_capacity, args.sparse_capacity) < 1:
        parser.error("World count and capacities must be positive")
    args.output.mkdir(exist_ok=False)
    config = cpu.Config(**json.loads((args.trial / "config.json").read_bytes()))
    cfg = replace(config, dt=args.step_us / 1e6, smoke_seconds=.020 if args.smoke else None, wall_limit=21600.)
    result = dict(worlds=args.worlds, dt_s=cfg.dt, capacity=[args.contact_capacity, args.constraint_capacity, args.sparse_capacity],
                  mode="held_pose_probe" if args.probe else "shared_live_yam_controller")
    try:
        if not args.probe and cfg.dt != config.dt and cfg.initial_state:
            cfg = transplant_prepared_state(cfg, args.output)
        result.update(capacity_probe(cfg, args) if args.probe else full_grasp(cfg, args))
    except (RuntimeError, ValueError) as error:
        result.update(outcome="failed", failure=str(error))
    with (args.output / "benchmark.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "state_samples"}), flush=True)
    if result.get("outcome") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
