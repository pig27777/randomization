"""Bounded top-pinch attachment, separate from the immutable v1 content mechanics.

The grasp executor supplies fingertip positions and its current grip-force reading.
This module detects top contacts and pins the pinch point (fingertip midpoint at the
actual depth) of every top segment covered between the fingertips to a mocap frame with
point connections (v1.3): position only, so the object can swing about the pinch.
Forces still pass through the original body/connection network.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

DEPTH_MIN = .003
LOAD_FILTER_TAU = .005   # low-pass of the connection load: rejects single-step contact spikes only
GRIP_STIFFNESS = 40000.  # N/m, fixed by the reviewed effect rule
GATHER_TRAVEL = .010    # total reduction in the fingertip gap before force begins


class PinchSpring:
    def __init__(self):
        self.reference_gap = None

    def arm(self, gap, eligible):
        if not math.isfinite(gap) or gap < 0:
            raise ValueError('finite nonnegative fingertip gap required')
        self.reference_gap = float(gap) if eligible else None

    def force(self, gap):
        if not math.isfinite(gap):
            raise ValueError('finite fingertip gap required')
        if self.reference_gap is None:
            return 0.
        return GRIP_STIFFNESS * max(self.reference_gap - GATHER_TRAVEL - gap, 0.)

    def release(self):
        self.reference_gap = None


def fragments(desc, position=(0, 0, 1)):
    frame = f'<body name="pinch_frame" mocap="true" pos="{position[0]} {position[1]} {position[2]}"/>'
    top = [] if desc.rigid else [i for i, s in enumerate(desc.segments) if s.idx[2] == desc.kappa.segments[2] - 1]
    eq = '\n'.join(f'<connect name="pinch_{i}" body1="pinch_frame" body2="s{i}" anchor="0 0 0" active="false" '
                   'solref="0.005 1" solimp="0.99 0.99 0.001"/>' for i in top)
    return frame, eq


class PinchAttachment:
    def __init__(self, sim, original_top, tip_width=.020):
        self.sim = sim
        self.coefficient = float(sim.desc.kappa.pinch_coeff)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('pinch_coeff must be finite and nonnegative')
        self.original_top = float(original_top)
        self.tip_width = float(tip_width)   # fingertip extent across the closing direction
        if not math.isfinite(self.original_top):
            raise ValueError('original top must be finite')
        self.top = [] if sim.desc.rigid else [i for i, s in enumerate(sim.desc.segments)
                                             if s.idx[2] == sim.desc.kappa.segments[2] - 1]
        self.equations = {i: sim.m.equality(f'pinch_{i}').id for i in self.top}
        frame = sim.m.body('pinch_frame').id
        self.mocap = sim.m.body_mocapid[frame]
        self.active = None        # equality ids of the active point connections, or None
        self.segment = None       # covered segment nearest the pinch point (material tracking)
        self.segments = []
        self.broken = False
        self.load = 0.          # low-passed connection load, compared with pinch_coeff x current grip force
        self.events = []
        self.peak_force = 0.

    def set_frame(self, position, quaternion=None):
        self.sim.d.mocap_pos[self.mocap] = position
        if quaternion is not None:
            self.sim.d.mocap_quat[self.mocap] = quaternion

    def top_contacts(self):
        """Existing contact bodies on upward-facing top surfaces, separately per fingertip."""
        sim = self.sim
        body_to_index = {sim.seg_bodies[i]: i for i in self.top}
        found = {p: set() for p in sim.pad_names}
        for c in sim.d.contact:
            if c.efc_address < 0 or sim.d.efc_force[c.efc_address] <= .05:
                continue
            for p in sim.pad_names:
                if c.geom1 in sim.pad_geoms[p]:
                    other, normal = c.geom2, -c.frame[:3]
                elif c.geom2 in sim.pad_geoms[p]:
                    other, normal = c.geom1, c.frame[:3]
                else:
                    continue
                body = sim.m.geom_bodyid[other]
                i = body_to_index.get(body)
                if i is not None and normal @ sim.d.xmat[body].reshape(3, 3)[:, 2] > .5:
                    found[p].add(i)
        return tuple(found[p] for p in sim.pad_names)

    def candidate(self, tips):
        if self.sim.desc.rigid or self.active is not None or self.broken:
            return None
        tips = np.asarray(tips, float)
        if tips.shape != (2, 3) or not np.all(np.isfinite(tips)):
            raise ValueError('two finite world-space fingertip positions are required')
        if np.any(self.original_top - tips[:, 2] < DEPTH_MIN - 1e-12):
            return None
        contacts = self.top_contacts()
        if len(contacts) != 2 or not all(contacts):
            return None
        midpoint = tips.mean(0)
        # Both contacts must belong to the connected patch below/between the tips.
        span = np.linalg.norm(tips[0, :2] - tips[1, :2]) / 2 + .020
        for group in contacts:
            if not any(np.linalg.norm(self.sim.d.xpos[self.sim.seg_bodies[i], :2] - midpoint[:2])
                       <= span + np.linalg.norm(self.sim.desc.segments[i].half[:2]) for i in group):
                return None
        covered = self.covered(tips)
        if not covered:
            return None
        return covered

    def covered(self, tips, samples=41, across=9):
        """Top segments whose top face meets the band between the fingertips (horizontal projection:
        tip to tip, one fingertip width across), nearest to the pinch point first. The width keeps a
        pinch over a seam on both neighbours even when squeezing opens the seam slightly."""
        sim = self.sim
        midpoint = tips.mean(0)
        along = tips[1, :2] - tips[0, :2]
        normal = np.array([-along[1], along[0]]) / max(np.linalg.norm(along), 1e-12)
        line = (tips[0, :2] + np.linspace(0, 1, samples)[:, None, None] * along
                + np.linspace(-.5, .5, across)[None, :, None] * self.tip_width * normal).reshape(-1, 2)
        samples = len(line)
        out = []
        for i in self.top:
            body = sim.seg_bodies[i]
            if sim.d.xpos[body, 2] > midpoint[2]:
                continue
            r = sim.d.xmat[body].reshape(3, 3)
            local = (np.c_[line, np.full(samples, midpoint[2])] - sim.d.xpos[body]) @ r
            half = sim.desc.segments[i].half
            if np.any(np.all(np.abs(local[:, :2]) <= half[:2] + 1e-9, axis=1)):
                out.append(i)
        return sorted(out, key=lambda j: (np.linalg.norm(sim.d.xpos[sim.seg_bodies[j]] - midpoint), j))

    def try_bind(self, tips):
        covered = self.candidate(tips)
        if covered is None:
            return False
        self.set_frame(np.asarray(tips).mean(0))
        self.bind_segments(covered)
        return True

    def bind_segment(self, i):
        self.bind_segments([i])

    def bind_segments(self, segments):
        """Pin the current pinch point of each segment to the frame, without moving the object."""
        if self.sim.desc.rigid:
            raise ValueError('rigid objects cannot be pinched')
        if not segments or any(i not in self.equations for i in segments):
            raise ValueError('pinch attachment requires top-layer segments')
        m, d = self.sim.m, self.sim.d
        point = d.mocap_pos[self.mocap].copy()
        ids = []
        for i in segments:
            body = self.sim.seg_bodies[i]
            eq = self.equations[i]
            m.eq_data[eq, 0:3] = 0.
            m.eq_data[eq, 3:6] = d.xmat[body].reshape(3, 3).T @ (point - d.xpos[body])
            d.eq_active[eq] = True
            ids.append(eq)
        self.active, self.segment, self.segments = ids, segments[0], list(segments)
        self.load = 0.
        self.events.append(dict(t=float(d.time), event='bind', segment=self.segment, segments=self.segments))

    def force_vector(self):
        """Resultant of the active point-connection forces (the frame-side reaction)."""
        if self.active is None:
            return np.zeros(3)
        d = self.sim.d
        rows = np.flatnonzero(d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        rows = rows[(d.efc_id[rows][:, None] == self.active).any(1)]
        if not len(rows):
            return np.zeros(3)  # Newly activated equality enters the next solver step.
        if len(rows) != 3 * len(self.active):
            raise RuntimeError(f'pinch connections have {len(rows)} constraint rows, expected {3 * len(self.active)}')
        return d.efc_force[rows].reshape(-1, 3).sum(0)

    def check_load(self, dt, grip_force, force_vector=None):
        """force_vector: this step's force_vector() if the caller already read it."""
        if not math.isfinite(grip_force) or grip_force < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt and nonnegative finite current grip force required')
        if self.active is None:
            return False
        force = float(np.linalg.norm(self.force_vector() if force_vector is None else force_vector))
        self.peak_force = max(self.peak_force, force)
        # v1.3 review: release as soon as the 5 ms low-passed load exceeds the capacity. A persistence
        # time would let the effective capacity rise with the loading rate.
        self.load += dt / (LOAD_FILTER_TAU + dt) * (force - self.load)
        limit = self.coefficient * grip_force
        if self.load > limit:
            self.release('overload', force=force, filtered_force=self.load, limit=limit)
            return True
        return False

    def release(self, reason='open', **evidence):
        if self.active is None:
            return
        self.sim.d.eq_active[self.active] = False
        self.events.append(dict(t=float(self.sim.d.time), event=reason, segment=self.segment,
                                segments=self.segments, **evidence))
        self.active = None
        self.broken = reason == 'overload'
