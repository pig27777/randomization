"""The macro state machine and the single bounded 13D Try."""

from dataclasses import dataclass, field

import numpy as np

from .geometry import InfoMode, PublicContext, SE3GraspCandidateSampler, VisualUnavailable, exp_rotation, vector
from .history import RAW_DIM, encoder_frames


class MotionStop(RuntimeError):
    def __init__(self, reason, *, physics_valid=True, unsafe=False):
        super().__init__(reason)
        self.physics_valid = physics_valid
        self.unsafe = unsafe


@dataclass
class ProbeTrace:
    frames: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    tcp_world: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    motion_m: float = 0.


class ProbeActionExecutor:
    def __init__(self, backend):
        self.backend = backend

    def execute(self, action, v0):
        action = vector(action, 13)
        if np.any(np.abs(action) > 1):
            raise ValueError("Try action must be in [-1, 1]")
        b = self.backend
        region = v0.half_extent
        start = v0.center + v0.rotation @ (action[:3] * region + [0., 0., region[2] + .012])
        displacement = b.probe_displacement_m
        second = start + v0.rotation @ (action[6:9] * displacement)
        third = second + v0.rotation @ (action[9:12] * displacement)
        rv = action[3:6] * np.pi / max(1., np.linalg.norm(action[3:6]))
        gap = (action[12] + 1.) * .5 * b.max_gap
        trace = ProbeTrace()
        q0 = b.hover_joints
        points = [np.clip(point, b.workspace_lower, b.workspace_upper) for point in (start, second, third)]
        current_p, _ = b.kinematics.pose(q0)
        last_p, _ = b.kinematics.pose(b.joint_pos)

        def record(stage, progress):
            nonlocal last_p
            if (sample := b.history_sample(v0, progress)) is None:
                return
            frame, timestamp, pos = sample
            trace.motion_m += float(np.linalg.norm(pos - last_p))
            last_p = pos
            trace.frames.append(frame)
            trace.timestamps.append(timestamp)
            trace.tcp_world.append(pos.copy())
            trace.stages.append(stage)

        # Backtrack rotations toward hover only when the real workcell rejects the
        # requested trajectory. No inclination angle is used as a rejection rule.
        for scale in (1., .5, .25, 0.):
            rotation = exp_rotation(v0.rotation @ (scale * rv)) @ b.hover_rotation
            exit_point = np.maximum(points[-1], [*b.workspace_lower[:2], current_p[2]])
            motion = b.kinematics.plan(q0, b.hover_gap, [
                (points[0], rotation, b.hover_gap, "probe_approach"),
                (points[1], rotation, gap, "probe_first"),
                (points[2], rotation, gap, "probe_second"),
                (points[2], rotation, b.hover_gap, "probe_release"),
                (exit_point, rotation, b.hover_gap, "probe_retract"),
                (current_p, b.hover_rotation, b.hover_gap, "probe_retract"),
            ])
            if motion is not None and b.kinematics.connect(motion[-1][0], q0, b.hover_gap, b.hover_gap):
                break
        else:
            return trace, dict(stop_reason="probe_path_rejected", unsafe=True, physics_validity=True,
                               retracted=b.at_hover(), rotation_scale=None)

        stop = None
        cutoff_N = getattr(b, "probe_cutoff_N", None)
        trigger = None
        try:
            if cutoff_N is not None:
                approach = [step for step in motion if step[2] == "probe_approach"]
                first = [step for step in motion if step[2] == "probe_first"]
                if getattr(b, "guard_full_probe_search", False):
                    search_labels = ("probe_approach", "probe_first", "probe_second")
                    search = [step for step in motion if step[2] in search_labels]
                else:
                    b.follow(approach, b.probe_effort_Nm, record=record, stage="probe")
                    search_labels = ("probe_approach", "probe_first")
                    search = first
                trigger = b.follow(search, b.probe_effort_Nm, record=record, stage="probe",
                                   contact_cutoff_N=cutoff_N)
                motion = [step for step in motion if step[2] not in search_labels]
                if trigger is not None:
                    actual_q = b.joint_pos.copy()
                    contact_p, contact_r = b.kinematics.pose(actual_q)
                    actual_gap = float(b.last_command[6])
                    unload_target = getattr(getattr(b, "drive", None), "probe_unload_target", None)
                    unload_q = unload_target() if unload_target is not None else None
                    if unload_q is None:
                        unload_p = contact_p.copy()
                        unload_p[2] = min(unload_p[2] + .003, b.workspace_upper[2])
                        unload_r = contact_r
                    else:
                        unload_p, unload_r = b.kinematics.pose(unload_q)
                    sweep = points[2] - points[1]
                    sweep[2] = max(0., sweep[2])
                    follow_p = np.clip(unload_p + sweep, b.workspace_lower, b.workspace_upper)
                    retract_p = follow_p.copy()
                    retract_p[2] = max(retract_p[2], current_p[2])
                    unloaded_gap = max(actual_gap, b.hover_gap) if unload_q is not None else actual_gap
                    remainder = [
                        (follow_p, unload_r, unloaded_gap, "probe_second"),
                        (follow_p, unload_r, b.hover_gap, "probe_release"),
                        (retract_p, unload_r, b.hover_gap, "probe_retract"),
                        (current_p, b.hover_rotation, b.hover_gap, "probe_retract"),
                    ]
                    if unload_q is None:
                        motion = b.kinematics.plan(actual_q, actual_gap, [
                            (unload_p, unload_r, unloaded_gap, "probe_unload"), *remainder])
                    elif b.kinematics.connect(actual_q, unload_q, actual_gap, unloaded_gap):
                        motion = b.kinematics.plan(unload_q, unloaded_gap, remainder)
                        if motion is not None:
                            motion.insert(0, (unload_q, unloaded_gap, "probe_unload"))
                    else:
                        motion = None
                    if motion is None or not b.kinematics.connect(motion[-1][0], q0, b.hover_gap, b.hover_gap):
                        raise MotionStop("guarded_probe_retraction_rejected", unsafe=True)
            motion.append((q0, b.hover_gap, "neutral_hover"))
            b.follow(motion, b.probe_effort_Nm, record=record, stage="probe")
            b.settle_hover(b.probe_effort_Nm, record)
            if not b.at_hover():
                raise MotionStop("neutral_hover_not_reached")
        except MotionStop as error:
            stop = error
            b.stop()
        return trace, dict(stop_reason=str(stop) if stop else None, unsafe=stop.unsafe if stop else False,
                           physics_validity=stop.physics_valid if stop else True,
                           retracted=stop is None, rotation_scale=scale, probe_cutoff_trigger_N=trigger)


class MacroBagGraspEnv:
    def __init__(self, backend, rng, *, info_mode=InfoMode.V1_ALLOWED, rules=None):
        self.backend = backend
        self.info_mode = InfoMode(info_mode)
        self.probe = ProbeActionExecutor(backend)
        self.sampler = SE3GraspCandidateSampler(backend, rng, **({"rules": rules} if rules is not None else {}))
        self.phase = "new"

    def reset(self, *, seed=None):
        self.v0, private, info = self.backend.reset(seed=seed)
        self.phase = "initial"
        self.v1 = None
        self.a_try = np.zeros(13)
        self.trace = ProbeTrace()
        self.result = dict(task_success=False, physics_validity=True, invalid_stage=None,
                           failure_reason=None, unsafe=False, probe_ticks=0, probe_motion_m=0.)
        return self.v0, private, info

    def execute_probe(self, a_try):
        if self.phase != "initial":
            raise RuntimeError("Expected a fresh episode for the single Try")
        self.a_try = vector(a_try, 13)
        self.trace, info = self.probe.execute(self.a_try, self.v0)
        length = len(self.trace.frames)
        self.result.update(probe_ticks=length, probe_motion_m=self.trace.motion_m,
                           unsafe=info["unsafe"], physics_validity=info["physics_validity"],
                           invalid_stage=None if info["physics_validity"] else "probe")
        if info["retracted"]:
            if self.info_mode is InfoMode.V1_ALLOWED:
                try:
                    self.v1 = self.backend.observe()
                except VisualUnavailable as error:
                    info["v1_unavailable"] = str(error)
            self.phase = "ready"
        else:
            self.result["failure_reason"] = info["stop_reason"] or "neutral_hover_not_reached"
            self.phase = "done"
        frames = encoder_frames(self.raw_history)
        return frames, length, self.v1 if self.info_mode is InfoMode.V1_ALLOWED else None, info

    @property
    def raw_history(self):
        return np.asarray(self.trace.frames, dtype=float).reshape(-1, RAW_DIM)

    def public_context(self, info_mode=None):
        mode = self.info_mode if info_mode is None else InfoMode(info_mode)
        if mode is InfoMode.V1_ALLOWED and self.info_mode is not InfoMode.V1_ALLOWED:
            raise ValueError("Cannot elevate an episode's information mode")
        return PublicContext(mode, self.v0, lambda: self.v1, self.a_try)

    def generate_grasp_candidates(self, info_mode=None, K=None):
        if self.phase not in ("initial", "ready"):
            raise RuntimeError("Candidates require neutral hover")
        return self.sampler.sample(self.public_context(info_mode), K)

    def execute_grasp(self, g_world):
        if self.phase not in ("initial", "ready"):
            raise RuntimeError("Grasp requires neutral hover and an unfinished episode")
        self.phase = "done"
        if g_world is None:
            self.result.update(failure_reason="no_feasible_grasp")
            return self.get_task_result()
        try:
            self.result.update(self.backend.execute_grasp(vector(g_world, 11)))
        except MotionStop as error:
            self.backend.stop()
            self.result.update(physics_validity=error.physics_valid, unsafe=self.result["unsafe"] or error.unsafe,
                               invalid_stage=None if error.physics_valid else "grasp", failure_reason=str(error))
        return self.get_task_result()

    def get_task_result(self):
        return self.result.copy()
