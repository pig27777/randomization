"""Differential IK and collision checks against the empty ABC workcell."""

import mujoco
import numpy as np

from yam_platform.robot import rotation_error
from yam_platform.yam_model import arm_indices

from .geometry import exp_rotation


def descendants(model, root):
    children = {model.body(root).id}
    for body, parent in enumerate(model.body_parentid):
        if parent in children:
            children.add(body)
    return children


class WorkcellKinematics:
    def __init__(self, model, lower, upper):
        self.model = model
        self.data = mujoco.MjData(model)
        joints, self.qadr, dofs = arm_indices(model)
        self.dofs = dofs[:6]
        self.lower, self.upper = model.jnt_range[joints[:6]].T
        self.workspace = (lower, upper)
        self.site = model.site("grasp_site").id
        self.jacp = np.empty((3, model.nv))
        self.jacr = np.empty_like(self.jacp)
        arm = descendants(model, "arm")
        self.arm_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(arm))))
        self.base_geoms = set(np.flatnonzero(model.geom_bodyid == model.body("arm").id))
        self.table = model.geom("table").id
        self.damping = .0002 * np.eye(6)

    def _set(self, q, gap):
        self.data.qpos[self.qadr[:6]] = q
        self.data.qpos[self.qadr[6:]] = gap / 2
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def pose(self, q, velocity=None):
        self._set(q, 0.)
        p = self.data.site_xpos[self.site].copy()
        rot = self.data.site_xmat[self.site].reshape(3, 3).copy()
        if velocity is None:
            return p, rot
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
        return p, rot, self.jacp[:, self.dofs] @ velocity

    def clear(self, q, gap):
        self._set(q, gap)
        # mj_fwdPosition updates broad-phase collision transforms, including flex.
        mujoco.mj_fwdPosition(self.model, self.data)
        for contact in self.data.contact:
            a, b = contact.geom
            if contact.dist >= -.00005 or (a not in self.arm_geoms and b not in self.arm_geoms):
                continue
            if ((a == self.table and b in self.base_geoms) or (b == self.table and a in self.base_geoms)):
                continue
            return False
        return True

    def connect(self, start, end, gap0, gap1):
        # Bound interpolation in joint and jaw space before checking the next segment.
        steps = max(1, int(np.ceil(max(np.max(np.abs(end - start)) / .075, abs(gap1 - gap0) / .012))))
        return all(self.clear(start + a * (end - start), gap0 + a * (gap1 - gap0))
                   for a in np.arange(1, steps + 1) / steps)

    def solve(self, position, rotation, seed, gap):
        lo, hi = self.workspace
        if np.any(position < lo) or np.any(position > hi):
            return None
        q = seed.copy()
        for _ in range(26):
            self._set(q, gap)
            p = self.data.site_xpos[self.site]
            r = self.data.site_xmat[self.site].reshape(3, 3)
            ep = position - p
            er = rotation_error(r, rotation)
            if np.linalg.norm(ep) < .0008 and np.linalg.norm(er) < .008:
                return q
            mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
            jac = np.vstack((4 * self.jacp[:, self.dofs], self.jacr[:, self.dofs]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + self.damping, np.r_[4 * ep, er])
            q = np.clip(q + np.clip(delta, -.15, .15), self.lower, self.upper)
        return None

    def plan(self, start, gap, targets):
        q = start
        result = []
        p0, r0 = self.pose(start)
        for position, rotation, next_gap, stage in targets:
            # Translation keyframes keep the commanded approach and lift Cartesian.
            # Intermediate orientation follows the geodesic from the prior keyframe.
            rv = rotation_error(r0, rotation)
            d = np.linalg.norm(position - p0)
            steps = max(1, int(np.ceil(max(d / .027, np.linalg.norm(rv) / .32))))
            start_gap = gap
            for a in np.arange(1, steps + 1) / steps:
                target_rot = exp_rotation(rv * a) @ r0
                target_p = p0 + a * (position - p0)
                opening = start_gap + a * (next_gap - start_gap)
                new = self.solve(target_p, target_rot, q, opening)
                if new is None or not self.connect(q, new, gap, opening):
                    return None
                result.append((new, opening, stage))
                q, gap = new, opening
            p0, r0 = position, rotation
        return result
