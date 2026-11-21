"""Optional one-decision micro-lift and same-episode regrasp for side pickups."""
import hashlib

import mujoco
import numpy as np


class MicroLift:
    height = .025
    dwell = .30

    def __init__(self, choose, *, height=.025):
        self.height = height
        self.choose = choose
        self.decision = None
        self.prefix = hashlib.sha256()
        self.samples = []

    def observe(self, robot, phase, target, gap):
        """Record paired commands and proprioception before the first decision."""
        if self.decision is not None:
            return
        sim = robot.sim
        d = sim.d
        self.prefix.update(phase.encode() + b"\0")
        self.prefix.update(np.r_[d.time, target, gap, d.qpos, d.ctrl,
                                 d.actuator_force[robot.act]].astype("<f8").tobytes())
        if phase not in ("close", "lift") or robot.steps % round(.02 / sim.m.opt.timestep):
            return
        jp = np.empty((3, sim.m.nv))
        jr = np.empty_like(jp)
        mujoco.mj_jacSite(sim.m, d, jp, jr, robot.site)
        jac = np.vstack((jp[:, robot.dofs], jr[:, robot.dofs]))
        # Quasi-static motor torque above the modeled unloaded-arm bias.
        tau = d.actuator_force[robot.act] - (d.qfrc_bias - d.qfrc_gravcomp)[robot.dofs]
        load = np.linalg.lstsq(jac.T, tau, rcond=.001)[0]
        self.samples.append((float(d.time), phase, float(load[2]), robot.gap(),
                             float(d.site_xpos[robot.site, 2])))

    def lift(self, elapsed, speed, now):
        """Return commanded upward travel and a one-shot request to reposition."""
        arrival = self.height / speed
        self.lower_seconds = arrival
        if self.decision is None and elapsed >= arrival + self.dwell:
            lift_start = now - elapsed
            def window(phase, start, end):
                rows = [r[2:] for r in self.samples if r[1] == phase and start <= r[0] < end]
                if not rows:
                    raise RuntimeError(f"Missing {phase} observation window {start} to {end}")
                return np.median(rows, axis=0)
            zero = window("close", lift_start - .10, lift_start)
            early = window("lift", lift_start + arrival, lift_start + arrival + .10)
            late = window("lift", now - .10, now)
            signals = dict(early_load_N=float(early[0]-zero[0]), late_load_N=float(late[0]-zero[0]),
                           gap_change_m=float(late[1]-early[1]))
            retry = self.choose(signals)
            self.decision = dict(time_s=now, retry=retry, **signals,
                                 actual_rise_m=float(late[2]-zero[2]))
            if retry:
                return self.height, True
        progress = min(elapsed, arrival) + max(0., elapsed - arrival - self.dwell)
        return speed * progress, False

    def retry_pose(self, elapsed, lift_site, closed_gap, open_gap, approach):
        """Lower, open, then shift the open hand smoothly to a nearby fold."""
        lowering = min(elapsed / self.lower_seconds, 1.)
        opening = np.clip((elapsed - self.lower_seconds) / .40, 0., 1.)
        shift = np.clip((elapsed - self.lower_seconds - .40) / .40, 0., 1.)
        offset = np.array([.008, 0., 0.]) + .006 * approach
        target = lift_site + [0., 0., self.height * (1. - lowering)] + shift * offset
        return target, closed_gap + opening * (open_gap - closed_gap), elapsed >= self.lower_seconds + .80
