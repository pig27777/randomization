"""Arm B interception, active tug verification, rescue, and gentle placement."""

import math

import torch

from ._parcel_env import ParcelEnv, columns, has_grasp, phase_encoding, scaled, select, smooth
from ..lumped_contact_proxy import ContactState
from ..rl.asymmetric_ppo import HANDOVER_DECISIONS
from ..unified_payload import DT, GRAVITY


class HandoverPlaceEnv(ParcelEnv):
    static_size, seq_size, priv_size = 18, 14, 102
    decisions = HANDOVER_DECISIONS
    tug_target_N = 6.
    tug_hold_s = .130
    tug_samples = 25

    @torch.no_grad()
    def reset(self, *, random_presentation=True):
        self._setup()
        p = self.episode
        payload = self.sim.payload
        swing, fold = p.swing_rad, p.fold_rad
        if random_presentation:
            rng = torch.Generator(device=self.device).manual_seed(self.seed + 2)
            draws = torch.rand((self.batch, 3), device=self.device, dtype=self.mass.dtype, generator=rng)
            swing = (draws[:, :1] - .5) * (math.pi / 6.)
            fold = torch.minimum(p.payload.theta_max, .59 * draws[:, 1:2])
        else:
            draws = self.zeros(3) + .5
        x, y, _ = payload.rest.unbind(dim=-1)
        z = payload.diagonal * (fold / 2.).sin() * payload.bend_sign
        c, s = swing.cos(), swing.sin()
        y, z = c * y - s * z, s * y + c * z + .30
        c, s = p.yaw_rad.cos(), p.yaw_rad.sin()
        pos = torch.stack((c*x - s*y, s*x + c*y, z), dim=-1)
        parcel = self.state.payload._replace(pos=pos)
        contacts = self.state.contacts_A.clone()
        contacts[..., ContactState.SLACK] = torch.minimum((.15 + .16 * draws[:, 1:2]) * self.sim.base_slack,
                                                          self.vector(.006))
        contacts[..., ContactState.CAPTURE] = 1.
        contacts[..., ContactState.HOLD_TIME] = .10
        self.state = self.state._replace(payload=parcel, contacts_A=contacts)
        anchors, _ = payload.anchors(parcel, self.sim.mount_weights[:, :2], parcel.d_surf)
        extension = self.mass * (GRAVITY / 2.) / p.proxy.k
        arm_a = anchors.mean(dim=1) + (p.proxy.L0.mean(dim=1, keepdim=True) + extension) * self.up
        arm_b = arm_a + scaled(draws[:, 2:3] * 2. - 1., .25, .35) * self.along
        grip = columns((12. + 18. * self.mass).clamp_max(35.), self.zeros())
        command = self._idle(torch.stack((arm_a, arm_b), dim=1), grip=grip,
                             half_span=columns(self.zeros() + .024, self.zeros() + .055))
        command.angle[:, :, 2] = p.yaw_rad
        return self._begin(command)

    @torch.no_grad()
    def reset_from_pick(self, pick):
        """Start only the successful rows of a completed pick batch."""
        if pick.episode is not self.episode or not bool(pick.done.all()):
            raise ValueError("Handover requires a terminal pick batch and its original Stage3 episode")
        self._setup(pick.state, pick.sim)
        arm_a = pick.command.center[:, 0]
        arm_b = arm_a + .30 * self.along
        cmd = pick.command._replace(center=torch.stack((arm_a, arm_b), dim=1),
                                    half_span=columns(pick.command.half_span[:, 0], self.zeros() + .055))
        self.offset[:, 0] = pick.offset[:, 0]
        self.offset_velocity[:, 0] = pick.offset_velocity[:, 0]
        self.load[:, 0] = pick.load[:, 0]
        return self._begin(cmd, pick.success)

    def _begin(self, command, active=None):
        if active is None:
            active = self.flags(True)
        self.done = ~active
        self.rescue_pending = self.flags()
        self.rescued = self.flags()
        self.ever_transferred = self.flags()
        self.peak_impact = self.zeros()
        self.release_height = self.zeros()
        self.release_recorded = self.flags()
        self.unload = self.zeros()
        self.settled = self.flags()
        self.nominal = command
        self.tug_signal = self.zeros(self.tug_samples, 2)
        self.tug_truth = self.zeros(self.tug_samples, 2)
        self.tug_capture = torch.zeros_like(self.tug_signal[..., 0], dtype=torch.bool)
        self._start_command(command)
        self._look(active)
        return self.observation()

    def _look(self, active):
        self._clear_history(active)
        for i in range(100):
            self._tick(self.command, active)
            if i % 10 == 9:
                self._sequence(0, active)
        # The last delivered target remains cached by VirtualRGBProjector
        # during the near-field approach.
        self.tare = select(active, self.sensed.force, self.tare)

    def observation(self):
        camera = self.rgb
        static = columns(self.actual[:, 1] - self.actual[:, 0], self.command.angle[:, 1] - self.command.angle[:, 0],
                         camera.wrist_target_rel, camera.wrist_profile, camera.wrist_sway_est,
                         camera.wrist_vis_valid, phase_encoding(self.phase, 3), (3 - self.attempts) / 3.)
        priv = columns(self.sim.get_privileged_state(self.state), self.state.contacts_A, self.state.contacts_B,
                       self._outer(), self.sim.base_slack, self.result.S_bag_total_eff, self._hardware(True))
        return self._obs(static, priv)

    def _pull(self, force):
        return force[:, :, 1] * self.vector([1., -1.])

    def _sequence(self, stage, mask):
        force = self.sensed.force - self.tare
        pull = self._pull(force)
        absolute = pull.abs()
        common = absolute.amax(dim=1).clamp_min(.25)
        balance = (pull.amin(dim=1) / common).clamp(-1., 1.)
        scale = self.command.grip.clamp_min(.25)
        conflict = ((pull[:, 0] - pull[:, 1]).abs() / common).clamp(0., 2.)
        err = self.sensed.position_error
        features = columns(phase_encoding(torch.full_like(self.phase, stage), 3),
                           (force.norm(dim=-1) / scale).clamp(0., 2.), balance, conflict,
                           err[:, 0, 2], err[:, 1, 2], err[:, 1, :2].norm(dim=-1),
                           self.sensed.gripper_current[:, 1], self.sensed.opening[:, 1],
                           self.unload.clamp(-.5, 2.), self.settled)
        self._record(features, mask)

    @torch.no_grad()
    def step(self, action):
        intercept = (self.phase == 0) & ~self.done
        verify = (self.phase == 1) & ~self.done
        place = (self.phase == 2) & ~self.done
        reward = self.zeros()
        if place.any():
            reward += self._place(action, place)
        if verify.any():
            reward += self._verify(action, verify)
        if intercept.any():
            reward += self._intercept(action, intercept)
        return self._transition(reward, transferred=self.ever_transferred.clone(), rescued=self.rescued.clone(),
                                impact_N=self.peak_impact.clone(), release_height_m=self.release_height.clone())

    def _lost(self, active):
        cap = self.state.contacts_A[..., ContactState.CAPTURE] > .5
        cap_b = self.state.contacts_B[..., ContactState.CAPTURE] > .5
        self.dropped |= active & (~(cap | cap_b).any(dim=1) | (self.result.table_clearance[:, 0] <= .001))
        return self.torn | self.dropped

    def _intercept(self, action, active):
        self._clear_history(active)
        self.attempts += active.long()
        tilt, compliance, dy, dz, speed, pinch = action.continuous[:, :6].clamp(-1., 1.).unbind(dim=1)
        start = self.command
        camera_center = self.actual[:, 1] + self.vector([0., .015, .015])
        estimate = camera_center + self.rgb.wrist_target_rel @ self.wrist_frame.transpose(0, 1)
        arm_a = start.center[:, 0]
        valid = self.rgb.wrist_vis_valid
        x = torch.where(valid[:, 0], estimate[:, 0], arm_a[:, 0])
        x = arm_a[:, 0] + (x - arm_a[:, 0]).clamp(-.025, .025)
        y = arm_a[:, 1] + scaled(dy, .013, .048)
        z_ref = arm_a[:, 2] - .047
        z = torch.where(valid[:, 0], estimate[:, 2], z_ref)
        z = z_ref + (z - z_ref).clamp(-.009, .009) + scaled(dz, -.015, .022)
        target = columns(x, y, z)
        radius = (.5 * self.rgb.wrist_profile[:, 1]).clamp(.042, .064)
        close = scaled(pinch, .025, .009)
        normal = scaled(pinch, 12., 39.)
        feed_limit = scaled(pinch, .025, .075)
        travel_time = scaled(speed, .92, .54)
        outer_edge = (y + .04).maximum(arm_a[:, 1] + radius)
        stiffness = start.stiffness.clone()
        stiffness[:, 0, :2] = scaled(compliance, 160., 55.)[:, None]
        for i in range(825):
            t = (i + 1) * DT
            approach = smooth(t / travel_time)
            closing = smooth((t - .92) / .57 + self.zeros())
            center = start.center.clone()
            center[:, 1] = torch.lerp(start.center[:, 1], target, approach[:, None])
            spans = columns(start.half_span[:, 0], torch.lerp(radius, close, closing))
            grips = columns(start.grip[:, 0], normal * smooth((t - .87) / .29 + self.zeros()))
            feed = columns(self.zeros(), (outer_edge - center[:, 1, 1]).clamp_min(0.).minimum(feed_limit))
            angle = start.angle.clone()
            angle[:, 0, 0] += scaled(tilt, -.18, .18) * smooth(self.vector(t / .65))
            self._tick(start._replace(center=center, half_span=spans, grip=grips, feed=feed,
                                      side=columns(self.flags(), self.flags(True)), angle=angle,
                                      stiffness=stiffness), active & ~self.dropped & ~self.torn)
            self._lost(active)
            if i in (804, 814, 824):
                self._sequence(0, active)
        self._active_tug(active & ~self.dropped & ~self.torn)
        self.phase = torch.where(active, 1, self.phase)
        failed = active & (self.torn | self.dropped)
        self.done |= failed
        return -5. * (active & self.torn).float() - 6. * (active & self.dropped).float()

    def _active_tug(self, active):
        origin = self.command
        self.tare = select(active, self.sensed.force, self.tare)
        self.tug_signal = select(active, self.zeros(self.tug_samples, 2), self.tug_signal)
        self.tug_truth = select(active, self.zeros(self.tug_samples, 2), self.tug_truth)
        self.tug_capture = select(active, torch.zeros_like(self.tug_capture), self.tug_capture)
        base_y = self.nominal.center[:, 0, 1]
        limit = .076 * self.tug_target_N / 6.
        target_y = columns(base_y - limit, base_y + limit)
        stiffness = self.vector([850., 1700., 900.]).expand_as(origin.center)
        z = self.nominal.center[:, 0, 2] - .047
        peak_center = origin.center.clone()
        peak_center[..., 1] = target_y
        peak_center[..., 2] = columns(z + .010, z + .010)
        trim = self.zeros()
        opposition = self.vector([-1., 1.])
        for i in range(250):
            t = (i + 1) * DT
            # Smooth separation followed by a compliant 130 ms hold.
            progress = smooth(self.vector(t / .22))
            release = smooth(self.vector((t - .22 - self.tug_hold_s) / .15))
            center = torch.lerp(origin.center, peak_center, progress * (1. - release))
            if .29 <= t <= .22 + self.tug_hold_s:
                sensed = self._pull(self.sensed.force - self.tare).amin(dim=1)
                # The short hold allows only a small, slow correction to the
                # calibrated travel. The feedback uses local current increments.
                trim = (trim + DT * (.001 * (.75 * self.tug_target_N - sensed)).clamp(-.003, .003)
                        ).clamp(-.001, .001)
            center[..., 1] += trim[:, None] * opposition * (1. - release)
            self._tick(origin._replace(center=center, stiffness=stiffness), active & ~self.torn & ~self.dropped)
            self._lost(active)
            if i % 10 == 9:
                frame = i // 10
                self.tug_signal[:, frame] = select(active, self._pull(self.sensed.force - self.tare),
                                                   self.tug_signal[:, frame])
                self.tug_truth[:, frame] = select(active, self._pull(self.load), self.tug_truth[:, frame])
                bilateral = has_grasp(self.state.contacts_A) & has_grasp(self.state.contacts_B)
                self.tug_capture[:, frame] = select(active, bilateral, self.tug_capture[:, frame])
                self._sequence(1, active & self.sensed.valid.all(dim=1)[:, 0])

    def _verify(self, action, active):
        decision = action.choice.clamp(0, 2)
        rescue = active & (decision == 2)
        transfer = active & ~rescue
        self._clear_history(active)
        self.rescue_pending |= rescue & ~has_grasp(self.state.contacts_B)
        if rescue.any():
            self._retract(rescue)
        if transfer.any():
            self._transfer(action, transfer)
        exhausted = rescue & (self.attempts >= 3)
        failed = active & (self.torn | self.dropped)
        self.done |= failed | exhausted | (transfer & ~self.ever_transferred)
        restarted = rescue & ~self.done
        if restarted.any():
            self.phase = torch.where(restarted, 0, self.phase)
            self._look(restarted)
        return -6. * (active & self.dropped).float() - 5. * (active & self.torn).float()

    def _retract(self, active):
        start = self.command
        for i in range(350):
            t = (i + 1) * DT
            open_jaw = smooth(self.vector(t / .18))
            back = smooth(self.vector((t - .18) / .51))
            center = torch.lerp(start.center, self.nominal.center, back)
            grip = columns(start.grip[:, 0], start.grip[:, 1] * (1. - open_jaw))
            span = columns(start.half_span[:, 0], torch.lerp(start.half_span[:, 1], self.nominal.half_span[:, 1], open_jaw))
            self._tick(start._replace(center=center, grip=grip, half_span=span, feed=self.zeros(2)),
                       active & ~self.torn & ~self.dropped)
            self._lost(active)

    def _transfer(self, action, active):
        boost = action.choice == 1
        start = self.command
        extra = scaled(action.continuous[:, 0].clamp(-1., 1.), 2., 10.) * boost
        ramp = torch.where(boost, scaled(action.continuous[:, 1].clamp(-1., 1.), .30, .64), .43)
        if (active & boost).any():
            for i in range(100):
                grip = columns(start.grip[:, 0],
                               (start.grip[:, 1] + extra * smooth(self.vector((i + 1) / 100.))).clamp_max(42.))
                self._tick(start._replace(grip=grip), active & boost & ~self.torn & ~self.dropped)
                self._lost(active & boost)
            self._active_tug(active & boost)
            start = self.command
        streak = torch.zeros_like(self.attempts)
        for i in range(4400):
            t = (i + 1) * DT
            align = smooth(self.vector(t / .95))
            release = smooth((t - 1.30) / ramp)
            stiffness = torch.lerp(start.stiffness, self.nominal.stiffness, smooth(self.vector(t / .38)))
            center = start.center.clone()
            center[:, 1] = torch.lerp(start.center[:, 1], self.nominal.center[:, 0] + .045 * self.up, align)
            center[:, 0, 2] += .045 * smooth(self.vector((t - 1.97) / .40))
            grip = columns(start.grip[:, 0] * (1. - release), start.grip[:, 1])
            half = columns(torch.lerp(start.half_span[:, 0], self.zeros() + .06, release), start.half_span[:, 1])
            self._tick(start._replace(center=center, grip=grip, half_span=half, stiffness=stiffness),
                       active & ~self.torn & ~self.dropped & ~self.ever_transferred)
            self._lost(active & ~self.ever_transferred)
            release_a = (self.state.contacts_A[..., ContactState.CAPTURE] == 0.).all(dim=1)
            steady = active & (release >= 1.) & release_a & self._stable(1) & ~self.dropped
            streak = torch.where(steady, streak + 1, 0)
            self.ever_transferred |= streak >= 250
            if i % 10 == 9:
                self._sequence(1, active)
            if i % 50 == 49 and ((~active) | self.ever_transferred | self.dropped | self.torn).all():
                break
        self.phase = torch.where(active & self.ever_transferred, 2, self.phase)

    def _place(self, action, active):
        self._clear_history(active)
        origin = self.command
        speed = scaled(action.continuous[:, 0].clamp(-1., 1.), .025, .11)
        release_ratio = scaled(action.continuous[:, 1].clamp(-1., 1.), .28, .65)
        baseline = self.sensed.force[:, 1, 2] - self.tare[:, 1, 2]
        encoder = self.sensed.position_error[:, 1, 2]
        contact_error = (.25 * encoder.abs()).clamp(.00015, .002)
        last_y = self.rgb.top_box[:, 1]
        contact = self.flags()
        release_clock = self.zeros()
        prospective_contact = self.zeros()
        distance = self.zeros()
        held_at_contact = self.flags()
        for i in range(9500):
            t = (i + 1) * DT
            distance += DT * speed * smooth(self.vector(t / .3)) * (~contact & (prospective_contact == 0.))
            distance = torch.minimum(distance, (origin.center[:, 1, 2] - .010).clamp_min(0.))
            release_clock += DT * contact
            prospective_contact = (prospective_contact - DT).clamp_min(0.)
            release = smooth(release_clock / .25)
            opening = smooth((release_clock - .26) / .20)
            center = origin.center.clone()
            center[:, 1, 2] += -distance + .065 * smooth((release_clock - .48) / .35)
            grip = columns(self.zeros(), origin.grip[:, 1] * (1. - release))
            # Unload the film before widening the jaws, which would tension it.
            span_b = torch.lerp(origin.half_span[:, 1], self.zeros() + .070, opening)
            span = columns(origin.half_span[:, 0], span_b)
            self._tick(origin._replace(center=center, grip=grip, half_span=span), active & ~self.dropped & ~self.torn)
            payload_on_table = self.result.table_clearance[:, 0] <= .001
            lost_grip = ~has_grasp(self.state.contacts_B)
            self.dropped |= active & lost_grip & ~held_at_contact & ~payload_on_table
            unladen = self.sensed.force[:, 1, 2] - self.tare[:, 1, 2]
            self.unload = 1. - (unladen / baseline.clamp_max(-.15)).clamp(-1., 2.)
            self.peak_impact = torch.maximum(self.peak_impact, torch.where(active, self.result.support_N[:, 0], 0.))
            if i % 10 == 9:
                top_y = self.rgb.top_box[:, 1]
                self.settled = (top_y - last_y).abs() < .0012
                last_y = top_y
                changed = self.sensed.position_error[:, 1, 2] - encoder > contact_error
                candidate = active & ~contact & changed & (prospective_contact == 0.)
                prospective_contact = torch.where(candidate, .18, prospective_contact)
                landed = ((self.unload > release_ratio) & changed & (prospective_contact < .07)
                          & (prospective_contact > 0.) & self.settled & self.rgb.top_valid[:, 0])
                hit = active & ~contact & landed
                held_at_contact |= hit & ~lost_grip & payload_on_table
                contact |= hit
                self._sequence(2, active)
            first_release = active & contact & lost_grip & ~self.release_recorded
            self.release_height = select(first_release, self.result.table_clearance[:, 0], self.release_height)
            self.release_recorded |= first_release
            self.success |= (active & held_at_contact & payload_on_table & ~self.dropped & ~self.torn
                             & (release_clock >= .86) & lost_grip)
            if i % 50 == 49 and ((~active) | self.success | self.dropped | self.torn).all():
                break
        self.rescued |= active & self.success & self.rescue_pending
        self.done |= active
        impact = .04 * self.peak_impact.clamp_max(60.) + 35. * self.release_height.clamp(0., .1)
        return (15. * (active & self.success).float() - impact * active.float()
                + 2.5 * (active & self.rescued).float()
                - 6. * (active & self.dropped).float() - 5. * (active & self.torn).float())
