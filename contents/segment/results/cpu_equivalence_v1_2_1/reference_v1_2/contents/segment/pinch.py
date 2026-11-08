"""Bounded top-pinch attachment, separate from the immutable v1 content mechanics.

The grasp executor supplies fingertip positions and its current grip-force reading.
This module detects top contacts and couples one existing top segment to a mocap
frame; forces still pass through the original body/connection network.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

DEPTH_MIN = .003
OVERLOAD_TIME = .020
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
    eq = '\n'.join(f'<weld name="pinch_{i}" body1="pinch_frame" body2="s{i}" active="false" '
                   'solref="0.005 1" solimp="0.99 0.99 0.001" torquescale="1"/>' for i in top)
    return frame, eq


class PinchAttachment:
    def __init__(self, sim, original_top):
        self.sim = sim
        self.coefficient = float(sim.desc.kappa.pinch_coeff)
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError('pinch_coeff must be finite and nonnegative')
        self.original_top = float(original_top)
        if not math.isfinite(self.original_top):
            raise ValueError('original top must be finite')
        self.top = [] if sim.desc.rigid else [i for i, s in enumerate(sim.desc.segments)
                                             if s.idx[2] == sim.desc.kappa.segments[2] - 1]
        self.equations = {i: sim.m.equality(f'pinch_{i}').id for i in self.top}
        frame = sim.m.body('pinch_frame').id
        self.mocap = sim.m.body_mocapid[frame]
        self.active = None
        self.segment = None
        self.broken = False
        self.overload_s = 0.
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
        candidates = [i for i in self.top if self.sim.d.xpos[self.sim.seg_bodies[i], 2] <= midpoint[2]]
        if not candidates:
            return None
        i = min(candidates, key=lambda j: (np.linalg.norm(self.sim.d.xpos[self.sim.seg_bodies[j]] - midpoint), j))
        return i

    def try_bind(self, tips):
        i = self.candidate(tips)
        if i is None:
            return False
        midpoint = np.asarray(tips).mean(0)
        self.set_frame(midpoint)
        self.bind_segment(i)
        return True

    def bind_segment(self, i):
        """Set the weld's rest pose to the current relative pose, without moving the object."""
        if self.sim.desc.rigid:
            raise ValueError('rigid objects cannot be pinched')
        if i not in self.equations:
            raise ValueError('pinch attachment requires a top-layer segment')
        m, d = self.sim.m, self.sim.d
        body = self.sim.seg_bodies[i]
        inv, rel, position = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(inv, d.mocap_quat[self.mocap])
        mujoco.mju_mulQuat(rel, inv, d.xquat[body])
        mujoco.mju_rotVecQuat(position, d.xpos[body] - d.mocap_pos[self.mocap], inv)
        eq = self.equations[i]
        m.eq_data[eq, 3:10] = np.r_[position, rel]
        d.eq_active[eq] = True
        self.active, self.segment = eq, i
        self.overload_s = 0.
        self.events.append(dict(t=float(d.time), event='bind', segment=i))

    def force_vector(self):
        if self.active is None:
            return np.zeros(3)
        d = self.sim.d
        rows = np.flatnonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == self.active))
        if not len(rows):
            return np.zeros(3)  # Newly activated equality enters the next solver step.
        if len(rows) != 6:
            raise RuntimeError(f'pinch weld has {len(rows)} constraint rows, expected 6')
        return d.efc_force[rows[:3]].copy()

    def check_load(self, dt, grip_force):
        if not math.isfinite(grip_force) or grip_force < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt and nonnegative finite current grip force required')
        if self.active is None:
            return False
        force = float(np.linalg.norm(self.force_vector()))
        self.peak_force = max(self.peak_force, force)
        limit = self.coefficient * grip_force
        self.overload_s = self.overload_s + dt if force > limit else 0.
        if self.overload_s >= OVERLOAD_TIME - 1e-12:
            self.release('overload', force=force, limit=limit)
            return True
        return False

    def release(self, reason='open', **evidence):
        if self.active is None:
            return
        self.sim.d.eq_active[self.active] = False
        self.events.append(dict(t=float(self.sim.d.time), event=reason, segment=self.segment,
                                overload_s=self.overload_s, **evidence))
        self.active = None
        self.broken = reason == 'overload'
