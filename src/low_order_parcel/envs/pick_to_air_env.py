"""Arm A tactile approach, hybrid grasp, and micro-lift verification."""

import torch

from ._parcel_env import ParcelEnv, columns, has_grasp, phase_encoding, scaled, select, smooth
from ..lift_probe import PROBE_RAMP_S, PROBE_SETTLE_S, probe_height
from ..lumped_contact_proxy import ContactState
from ..rl.asymmetric_ppo import PICK_DECISIONS
from ..unified_payload import DT


class PickToAirEnv(ParcelEnv):
    static_size, seq_size, priv_size = 15, 12, 86
    decisions = PICK_DECISIONS

    @torch.no_grad()
    def reset(self, *, forced_mode=None):
        self._setup()
        self.enabled = self.arm_mask & self.vector([1., 0.]).bool()
        self.mode_lock = torch.full_like(self.phase, -1) if forced_mode is None else forced_mode
        self.previous_mode = torch.full_like(self.phase, 2)
        self.previous_action = self.zeros(2)
        self.height = self.zeros()
        self.load_reference = self.zeros()
        self.force_previous = self.zeros()
        self.slipped = self.flags()
        self.rescue_pending = self.flags()
        self.rescued = self.flags()
        self.grasp_force = self.zeros()
        start = torch.stack((self.zeros(3) + .25 * self.up,
                             self.zeros(3) + .35 * (self.along + self.up)), dim=1)
        self._start_command(self._idle(start))
        self._touchdown(self.flags(True))
        return self.observation()

    def observation(self):
        rgb = self.rgb
        static = columns(rgb.top_box[:, :4], rgb.top_clearance[:, 2:], self.height,
                         phase_encoding(self.phase, 2), (3 - self.attempts) / 3.,
                         phase_encoding(self.previous_mode, 3), self.previous_action)
        priv = columns(self.sim.get_privileged_state(self.state), self.state.contacts_A,
                       self._outer(), self._hardware(False))
        locked = torch.where(self.phase == 0, self.mode_lock, -1)
        return self._obs(static, priv, locked)

    def _sequence(self, phase, mask, target):
        force = self.sensed.force[:, 0] - self.tare[:, 0]
        sign = torch.where(self.load_reference < 0., -1., 1.)
        z = force[:, 2] * sign
        reference = self.load_reference.abs().clamp_min(.25)
        error = self.sensed.position_error[:, 0]
        load_feature = z.abs() / target.clamp_min(.25)
        gamma = self.rgb.vis_lift_ratio
        if phase == 1:
            # Positive for extra downward wrist load relative to the closed grasp.
            load_feature = (self.load_reference - force[:, 2]) / target.clamp_min(.25)
            gamma = torch.where(self.rgb.vis_lift_valid, gamma, -1.)
        features = columns(phase_encoding(torch.full_like(self.phase, phase), 3),
                           (z / reference).clamp(-.5, 2.),
                           ((z - self.force_previous) / reference).clamp(-2., 2.),
                           (force[:, :2].norm(dim=1) / z.abs().clamp_min(.25)).clamp(0., 3.),
                           error[:, 2], error[:, :2].norm(dim=1),
                           load_feature.clamp(-3., 3.),
                           self.sensed.opening[:, 0], self.sensed.gripper_current[:, 0],
                           gamma)
        self._record(features, mask)
        self.force_previous = select(mask, z, self.force_previous)

    def _touchdown(self, active):
        self._clear_history(active)
        safe = self.command.center.clone()
        safe[:, 0, 2] = .25
        start = self.command
        for i in range(130):
            center = torch.lerp(start.center, safe, smooth(self.vector(i / 129.)))
            self._tick(start._replace(center=center, grip=self.zeros(2), half_span=self.zeros(2) + .04,
                                      feed=self.zeros(2), side=self.zeros(2).bool()), active)
        self.tare = select(active, self.sensed.force, self.tare)
        safe[:, 0, :2] = self.rgb.top_box[:, :2]
        start = self.command
        for i in range(113):
            self._tick(start._replace(center=torch.lerp(start.center, safe, smooth(self.vector(i / 112.)))),
                       active)
        # Pre-position with a guarded coarse search, then take the standardized
        # 20 mm touchdown trace starting just above the encoder contact height.
        coarse = self.flags()
        contact = self.zeros()
        goal = safe.clone()
        goal[:, 0, 2] = .003
        stiffness = self.vector([80., 80., 185.]).expand_as(safe)
        for i in range(625):
            center = torch.lerp(safe, goal, smooth(self.vector((i + 1) / 625.)))
            center[:, 0, 2] = torch.where(coarse, self.command.center[:, 0, 2], center[:, 0, 2])
            self._tick(self.command._replace(center=center, stiffness=stiffness), active)
            touched = active & ~coarse & self._touch_signal()
            contact = select(touched, self.actual[:, 0, 2], contact)
            coarse |= touched
        start = self.command.center.clone()
        clearance = torch.where(coarse, contact + .012, self.actual[:, 0, 2] + .020)
        ready = start.clone()
        ready[:, 0, 2] = clearance
        for i in range(125):
            self._tick(self.command._replace(center=torch.lerp(start, ready, smooth(self.vector((i + 1) / 125.)))),
                       active)
        detected = self.flags()
        height = self.zeros()
        for i in range(125):
            center = ready.clone()
            z = clearance - .020 * smooth(self.vector((i + 1) / 125.))
            center[:, 0, 2] = torch.where(detected, self.command.center[:, 0, 2], z)
            self._tick(self.command._replace(center=center), active)
            hit = active & ~detected & self._touch_signal()
            height = select(hit, self.actual[:, 0, 2], height)
            detected |= hit
            if i >= 29 and i % 10 == 9:
                self._sequence(0, active & self.sensed.valid[:, 0, 0], self.zeros() + 2.5)
        self.height = select(active, torch.where(detected, height, torch.where(coarse, contact, 0.)), self.height)

    def _touch_signal(self):
        return ((self.sensed.position_error[:, 0, 2].abs() > .004)
                | ((self.sensed.force[:, 0, 2] - self.tare[:, 0, 2]) > 2.5))

    @torch.no_grad()
    def step(self, action):
        route = (self.phase == 0) & ~self.done
        verify = (self.phase == 1) & ~self.done
        reward = self.zeros()
        if verify.any():
            reward += self._verify(action, verify)
        if route.any():
            reward += self._grasp(action, route)
        return self._transition(reward, mode=self.previous_mode.clone(), rescued=self.rescued.clone())

    def _grasp(self, action, active):
        a, b, c, d = action.continuous[:, :4].clamp(-1., 1.).unbind(dim=1)
        mode = torch.where(self.mode_lock >= 0, self.mode_lock, action.choice.clamp(0, 1))
        side = mode == 1
        self.previous_mode = select(active, mode, self.previous_mode)
        self.previous_action = select(active, columns(a, c), self.previous_action)
        self.grasp_force = select(active, scaled(c, 3., 36.), self.grasp_force)
        self.attempts += active.long()
        self._clear_history(active)
        box = self.rgb.top_box
        across = columns(box[:, 4], box[:, 5], self.zeros())
        across = torch.where(self.rgb.top_valid, across, self.eye_x)
        along = torch.linalg.cross(self.up.expand_as(across), across)
        direction = torch.where(self.rgb.top_clearance[:, 2] >= self.rgb.top_clearance[:, 3], 1., -1.)
        along = along * direction[:, None]
        center_xy = columns(box[:, :2], self.zeros())
        half_depth = (.5 * box[:, 3]).clamp(.024, .22)
        gather = torch.where(side, scaled(b, .012, .045), scaled(b, .012, .050))
        press = scaled(a, .30, 6.)
        entry_z = torch.where(side, self.height + scaled(a, -.010, .018), self.height - .004).clamp_min(.004)
        origin = self.command
        start_half = origin.half_span[:, 0]
        open_half = torch.where(side, .045, .048)
        close_half = (open_half - gather).clamp_min(.005)
        approach_time = torch.where(side, 1.1 - .50 * d, .65).clamp(.55, 1.65)
        peak_progress = self.best_progress
        for i in range(1250):
            t = i * DT
            descending = smooth(t / .38 + self.zeros())
            entering = smooth((t - .38) / approach_time)
            closing = smooth((t - 1.10) / .80 + self.zeros())
            gathering = smooth((t - 1.78) / .65 + self.zeros())
            y = -(half_depth + .026) * (1. - entering) - .006 * entering
            offset = torch.where(side[:, None], along * (y * (1. - gathering))[:, None],
                                 across * scaled(d, -.035, .035)[:, None])
            target = center_xy + offset + self.up * entry_z[:, None]
            side_start = center_xy - along * (half_depth + .026)[:, None] + self.up * entry_z[:, None]
            target = torch.where(side[:, None] & (t < .38), side_start, target)
            cmd = self.command.center.clone()
            cmd[:, 0] = torch.lerp(origin.center[:, 0], target, descending[:, None])
            if t >= .38:
                measured = (self.sensed.force[:, 0, 2] - self.tare[:, 0, 2]).clamp(-10., 12.)
                pressure_z = self.command.center[:, 0, 2] + DT * (.022 * (measured - press)).clamp(-.045, .045)
                cmd[:, 0, 2] = torch.where(side, cmd[:, 0, 2], pressure_z.clamp(.004, .25))
            span = origin.half_span.clone()
            span[:, 0] = torch.lerp(torch.lerp(start_half, open_half, descending), close_half, closing)
            grip = columns(self.grasp_force * smooth(self.vector((t - .91) / .28)), self.zeros())
            feed = columns(torch.where(side, (y + half_depth).clamp_min(0.) * entering, 0.), self.zeros())
            angle = origin.angle.clone()
            angle[:, 0, 2] = torch.atan2(across[:, 1], across[:, 0]) * descending
            self._tick(origin._replace(center=cmd, half_span=span, grip=grip, feed=feed, angle=angle,
                                       side=columns(side, self.flags()),
                                       stiffness=self.vector([85., 85., 440.]).expand_as(cmd)), active)
            slack = self.state.contacts_A[..., ContactState.SLACK].sum(dim=1)
            inward = self.result.net_feed[:, 0].mean(dim=1)
            progress = .35 * (slack / .035).clamp(0., 1.) + .25 * (inward / .07).clamp(0., 1.)
            peak_progress = torch.maximum(peak_progress, torch.where(active, progress, 0.))
        self.load_reference = select(active, self.sensed.force[:, 0, 2] - self.tare[:, 0, 2],
                                     self.load_reference)
        self.force_previous = select(active, self.load_reference.abs(), self.force_previous)
        slip_start = self.state.contacts_A[..., ContactState.SLIP_CUM]
        origin = self.command
        stroke = probe_height(self.height)
        for i in range(round((PROBE_RAMP_S + PROBE_SETTLE_S) / DT)):
            center = origin.center.clone()
            center[:, 0, 2] += stroke * min(i * DT / PROBE_RAMP_S, 1.)
            self._tick(origin._replace(center=center), active, lift=True, exterior_height=self.height[:, None])
            if i % 10 == 0:
                self._sequence(1, active & self.sensed.valid[:, 0, 0], self.grasp_force)
        slip = (self.state.contacts_A[..., ContactState.SLIP_CUM] - slip_start).amax(dim=1) > .002
        self.slipped = select(active, slip | ~has_grasp(self.state.contacts_A), self.slipped)
        progress_reward = peak_progress - self.best_progress
        self.best_progress = peak_progress
        self.phase = torch.where(active, 1, self.phase)
        self.done |= active & self.torn
        return progress_reward - .15 * (active & side).float() - 5. * (active & self.torn).float()

    def _verify(self, action, active):
        decision = action.choice.clamp(0, 2)
        retry = active & (decision == 2)
        lift = active & ~retry
        boost = decision == 1
        self.rescue_pending |= active & self.slipped & (boost | retry)
        self._clear_history(active)
        origin = self.command
        streak = torch.zeros_like(self.attempts)
        was_airborne = self.flags()
        if lift.any():
            gain = scaled(action.continuous[:, 0].clamp(-1., 1.), 1., 12.)
            speed = scaled(action.continuous[:, 1].clamp(-1., 1.), .35, .85)
            duration = torch.where(boost, 1.25 / speed, 1.25)
            for i in range(3520):
                t = i * DT
                center = origin.center.clone()
                center[:, 0, 2] += .265 * smooth(t / duration)
                grip = origin.grip.clone()
                grip[:, 0] = (grip[:, 0] + boost * gain * smooth(self.vector(t / .24))).clamp_max(42.)
                self._tick(origin._replace(center=center, grip=grip), lift & ~self.success & ~self.torn)
                airborne = self.result.table_clearance[:, 0] > .001
                self.dropped |= lift & was_airborne & (~airborne | ~has_grasp(self.state.contacts_A)) & ~self.success
                was_airborne |= lift & airborne
                stable = lift & ~self.dropped & self._stable(0)
                streak = torch.where(stable, streak + 1, 0)
                self.success |= lift & (streak >= 1500)
                if i % 50 == 49 and ((~lift) | self.success | self.dropped | self.torn).all():
                    break
        self.rescued |= lift & self.success & self.rescue_pending
        exhausted = retry & (self.attempts >= 3)
        restarts = retry & ~exhausted & ~self.torn
        self.done |= lift | exhausted | self.torn
        if restarts.any():
            self._touchdown(restarts)
            self.phase = torch.where(restarts, 0, self.phase)
        return (10. * (lift & self.success).float() + 1.75 * (lift & self.rescued).float()
                - 4. * (lift & self.dropped).float() - 2. * exhausted.float()
                - 5. * (active & self.torn).float() - .35 * restarts.float())
