"""Execute independent Policy V3 macro motions in one MJWarp world batch."""

from copy import copy
from queue import Queue
from threading import Event, Thread
from time import perf_counter

import mujoco
import numpy as np

from .yam_backend import YamBackend
from .yam_kinematics import WorkcellKinematics


class WorldDrive:
    """Host side of one world. The coordinator owns every CUDA operation."""

    def __init__(self, batch, world):
        self.batch, self.world = batch, world
        self.data = mujoco.MjData(batch.drive.model)
        self.initial = batch.drive.initial
        self.stopped = False

    @property
    def name(self):
        return self.batch.drive.name

    def reset(self, q, payload=None):
        self.stopped = False
        self.current_probe_cutoff = 0.
        self.probe_trigger_event = None
        self.clear_probe = False
        if payload is not None:
            self.payload_mass = payload[0]
        frame = self.batch.exchange(self.world, "reset", (q, payload))
        self.batch.drive.populate_frame(self, frame, self.world)

    def set_probe_cutoff(self, threshold):
        self.current_probe_cutoff = threshold
        self.clear_probe = True
        if threshold:
            self.probe_trigger_event = None

    def probe_unload_target(self):
        return self.unload_target

    def penetration_event(self):
        return self.batch.drive.contact_stop_event(self)

    def stop(self):
        self.stopped = True

    def tick(self, command, effort, _check, phase):
        if self.stopped:
            raise RuntimeError("Drive is stopped until reset")
        limits = self.batch.drive.limits
        total = 3.5 if phase == "probe" else (limits.probe_contact_N if phase == "reset" else limits.grasp_contact_N)
        force_limits = [total, 25. if phase == "grasp" else 0., self.current_probe_cutoff]
        frame = self.batch.exchange(self.world, "step", (command, effort, force_limits, self.clear_probe))
        self.clear_probe = False
        self.batch.drive.populate_frame(self, frame, self.world)
        self.batch.drive.check_stop(self.frame_reason, phase)


class BatchedYam:
    """Run normal macro methods lockstep, including per-world branching and stops."""

    def __init__(self, seed, worlds, device="cuda:0", *, randomized_payload=False, flex_solver="nominal",
                 solver_options=None, contact_capacity=None, constraint_capacity=None):
        self.worlds = worlds
        self.prototype = YamBackend(seed, scene="flex", device=device, worlds=worlds,
                                    randomized_payload=randomized_payload, flex_solver=flex_solver,
                                    solver_options=solver_options, contact_capacity=contact_capacity,
                                    constraint_capacity=constraint_capacity)
        self.drive = self.prototype.drive
        self.jobs = Queue()
        self.replies = [Queue() for _ in range(worlds)]
        self.aborted = Event()
        self.backends = []
        for world in range(worlds):
            backend = copy(self.prototype)
            backend.scene = copy(self.prototype.scene)
            backend.rng = np.random.default_rng(np.random.SeedSequence([seed, world]))
            backend.drive = WorldDrive(self, world)
            backend.kinematics = WorkcellKinematics(self.prototype.kinematics.model,
                                                   backend.workspace_lower, backend.workspace_upper)
            backend.optics = None
            backend.observe = lambda i=world: self.exchange(i, "camera", self.backends[i].drive.data)
            self.backends.append(backend)

    def exchange(self, world, kind, data):
        if self.aborted.is_set():
            raise RuntimeError("Batch aborted")
        self.jobs.put((world, kind, data))
        value = self.replies[world].get()
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        self.prototype.close()

    def run(self, episodes, *, progress=None):
        from .yam_gpu import SOLVER_LIMIT_FLAGS

        if not 0 < len(episodes) <= self.worlds:
            raise ValueError("Episode wave must fit the allocated MJWarp batch")
        start = perf_counter()
        results = [None] * len(episodes)
        latencies = np.zeros(len(episodes))

        def episode(world, fn):
            try:
                result = fn(self.backends[world])
            except Exception as error:
                result = error
            self.jobs.put((world, "done", result))

        threads = [Thread(target=episode, args=(i, fn), name=f"yam-world-{i}") for i, fn in enumerate(episodes)]
        active = set(range(len(threads)))
        pending = {}
        count = 0
        frames = 0
        active_step_wall_s = 0.

        def collect():
            while len(pending) < len(active):
                world, kind, value = self.jobs.get()
                if kind in ("camera", "rgb"):
                    try:
                        optics = self.prototype.optics
                        if kind == "camera":
                            value = optics.observe(value)
                        else:
                            optics.renderer.update_scene(value, camera=optics.camera)
                            value = optics.renderer.render().copy()
                    except Exception as error:
                        value = error
                    self.replies[world].put(value)
                elif kind == "done":
                    active.remove(world)
                    if isinstance(value, BaseException):
                        raise value
                    results[world] = value
                    latencies[world] = perf_counter() - start
                else:
                    pending[world] = kind, value

        try:
            for thread in threads:
                thread.start()
            collect()
            if any(kind != "reset" for kind, _ in pending.values()):
                raise RuntimeError("Every episode must start with one batched reset")
            initial = np.repeat(self.drive.initial[None, :], self.worlds, axis=0)
            payload = None
            if self.prototype.randomized_payload:
                payload = np.repeat([[self.drive.payload_mass, self.prototype.scene.parcel_config.friction]],
                                    self.worlds, axis=0)
            for world, (_, (q, properties)) in pending.items():
                initial[world] = q
                if payload is not None:
                    payload[world] = properties
            self.drive.reset_batch(initial, payload)
            frame = self.drive.read_batch()
            for world in active:
                self.replies[world].put(frame)
            pending.clear()
            pos = initial[:, self.prototype.qadr[:6]]
            commands = np.c_[pos, initial[:, self.prototype.qadr[6:]].sum(axis=1)]
            idle_effort = self.prototype.probe_effort_Nm
            idle_command = np.r_[self.prototype.hover_joints, self.prototype.hover_gap]
            idle_force_limits = [self.drive.limits.probe_contact_N, 0., 0.]
            efforts = np.full(self.worlds, idle_effort)
            limits = np.zeros((self.worlds, 3))
            clear = np.zeros(self.worlds, dtype=np.int32)
            while active:
                before = active.copy()
                collect()
                if not active:
                    break
                finished = before - active
                if finished:
                    self.drive.park_worlds(finished)
                    commands[list(finished)] = idle_command
                    efforts[list(finished)] = idle_effort
                limits[:] = idle_force_limits
                clear.fill(0)
                for world, (kind, value) in pending.items():
                    if kind != "step":
                        raise RuntimeError("Each batched episode may reset only once")
                    commands[world], efforts[world], limits[world], clear[world] = value
                start_frame = perf_counter()
                self.drive.step_batch(commands, efforts, limits, clear_probe=clear if clear.any() else None)
                frame = self.drive.read_batch()
                overflow = self.drive.sim.data.overflow.numpy()
                if np.any(frame["frame_reason"] == 1) or np.any(overflow & ~SOLVER_LIMIT_FLAGS):
                    raise RuntimeError("MJWarp capacity overflow during batched stepping")
                if self.drive.sim.device.free_memory < 4 * 1024**3:
                    raise RuntimeError("GPU memory headroom fell below 4 GiB")
                active_step_wall_s += perf_counter() - start_frame
                count += len(active)
                frames += 1
                if progress is not None and (frames == 1 or frames % 10 == 0):
                    progress(dict(control_frames=frames, env_steps=count, wall_s=perf_counter() - start,
                                  completed_episodes=len(results) - len(active)))
                for world in active:
                    self.replies[world].put(frame)
                pending.clear()
        except BaseException:
            self.aborted.set()
            for queue in self.replies:
                queue.put(RuntimeError("Batch coordinator interrupted"))
            raise
        finally:
            for thread in threads:
                if thread.ident is not None:
                    thread.join()
        wall_s = perf_counter() - start
        return results, dict(wall_s=wall_s, env_steps=count, control_frames=frames, completed_episodes=len(results),
                             env_steps_s=count / wall_s, mean_episode_wall_s=float(latencies.mean()),
                             amortized_wall_s_per_ep=wall_s / len(results),
                             device_phase_env_steps_s=count / active_step_wall_s if active_step_wall_s else 0.)
