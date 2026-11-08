"""Explicit CPU adapter for the shared Warp mechanics and effect kernels.

The frozen runtime is left unchanged. The validator enables all operators in a
context manager, so a failed candidate cannot silently become the default.
"""
from contextlib import contextmanager
import math

import numpy as np
import warp as wp

import mj_adapter as A
import pinch
import top_grasp
import warp_effects as E
from warp_mechanics import WarpBushings


def view(value, dtype=wp.float64):
    return wp.array(np.asarray(value)[None], dtype=dtype, device="cpu", copy=False)


def contacts(sim):
    c = sim.d.contact
    return [wp.array(c.geom.astype(np.int32)[None], device="cpu"),
            view(c.frame), wp.array(c.efc_address.astype(np.int32)[None], device="cpu"),
            view(sim.d.efc_force)]


def active_view(sim):
    return view(sim.d.eq_active.view(np.bool_), bool)


class WarpCompaction:
    def __init__(self, sim):
        b = sim.bushings
        arrays = ((sim.seg_bodies, int), (sim._obj_geom_mask, bool), (sim.welds, int),
                  (b.node_a, int), (b.node_b, int), (sim._seg_qadr, int))
        self.fixed = [wp.array(value, dtype=dtype, device="cpu") for value, dtype in arrays]
        self.filtered = wp.zeros(1, dtype=wp.float64, device="cpu")
        self.raw = wp.zeros(1, dtype=wp.float64, device="cpu")

    def apply(self, sim, external):
        self.filtered.assign(np.array([sim.external_compaction_load]))
        wp.launch(E.compaction, dim=1, device="cpu", inputs=[
            *self.fixed, view(external), *contacts(sim), view(sim.bushings.orthogonal_compaction),
            view(sim.d.qpos), sim.m.opt.timestep, A.COMPACTION_LOAD_TAU, A.WELD_ON,
            A.WELD_RELEASE, A.COMPACTION_LOAD_MIN, self.filtered, self.raw,
            active_view(sim), view(sim.m.eq_data)])
        sim.external_compaction_load = float(self.filtered.numpy()[0])
        sim.external_compaction_load_raw = float(self.raw.numpy()[0])


def switch_welds(sim, external):
    if not hasattr(sim, "_warp_compaction"):
        sim._warp_compaction = WarpCompaction(sim)
    sim._warp_compaction.apply(sim, external)


class WarpPinchSpring(pinch.PinchSpring):
    def __init__(self):
        super().__init__()
        self._force = wp.zeros(1, dtype=wp.float64, device="cpu")

    def force(self, gap):
        if not math.isfinite(gap):
            raise ValueError("finite fingertip gap required")
        reference = self.reference_gap
        wp.launch(E.spring_force, dim=1, device="cpu", inputs=[gap, reference or 0., reference is not None,
                  pinch.GATHER_TRAVEL, pinch.GRIP_STIFFNESS, self._force])
        return float(self._force.numpy()[0])


class WarpPinchAttachment(pinch.PinchAttachment):
    def __init__(self, sim, original_top):
        super().__init__(sim, original_top)
        half = np.array([s.half for s in sim.desc.segments])
        masks = np.array([sim._pad_geom_mask[p] for p in sim.pad_names], dtype=bool)
        arrays = ((self.top, int), (sim.seg_bodies, int), (half, wp.vec3d),
                  (sim.m.geom_bodyid, int), (masks, bool))
        self._fixed = [wp.array(value, dtype=dtype, device="cpu") for value, dtype in arrays]
        self._contacts = wp.zeros((1, 2, len(self.top)), dtype=int, device="cpu")
        self._selected = wp.zeros(1, dtype=int, device="cpu")
        self._reaction = wp.zeros(1, dtype=wp.vec3d, device="cpu")
        self._rows = wp.zeros(1, dtype=int, device="cpu")
        self._load = wp.zeros((1, 4), dtype=wp.float64, device="cpu")
        self._detached = wp.zeros(1, dtype=bool, device="cpu")

    def _select(self, tips):
        wp.launch(E.pinch_candidate, dim=1, device="cpu", inputs=[
            *self._fixed, *contacts(self.sim), view(self.sim.d.xpos, wp.vec3d),
            view(self.sim.d.xmat.reshape(-1, 3, 3), wp.mat33d), view(tips, wp.vec3d),
            self.original_top, -1 if self.active is None else self.active, self.broken,
            pinch.DEPTH_MIN, self._contacts, self._selected])

    def top_contacts(self):
        if len(self.sim.pad_names) != 2 or not self.top:
            return tuple(set() for _ in self.sim.pad_names)
        self._select(np.zeros((2, 3)))
        mask = self._contacts.numpy()[0]
        return tuple({self.top[i] for i in np.flatnonzero(row)} for row in mask)

    def candidate(self, tips):
        if self.sim.desc.rigid or self.active is not None or self.broken:
            return None
        tips = np.asarray(tips, float)
        if tips.shape != (2, 3) or not np.isfinite(tips).all():
            raise ValueError("two finite world-space fingertip positions are required")
        if len(self.sim.pad_names) != 2:
            return None
        self._select(tips)
        selected = int(self._selected.numpy()[0])
        return None if selected < 0 else selected

    def bind_segment(self, i):
        if self.sim.desc.rigid:
            raise ValueError("rigid objects cannot be pinched")
        if i not in self.equations:
            raise ValueError("pinch attachment requires a top-layer segment")
        sim, eq = self.sim, self.equations[i]
        wp.launch(E.bind_pose, dim=1, device="cpu", inputs=[sim.seg_bodies[i], eq, self.mocap,
            view(sim.d.xpos, wp.vec3d), view(sim.d.xquat, wp.vec4d), view(sim.d.mocap_pos, wp.vec3d),
            view(sim.d.mocap_quat, wp.vec4d), active_view(sim), view(sim.m.eq_data)])
        self.active, self.segment = eq, i
        self.overload_s = 0.
        self.events.append(dict(t=float(sim.d.time), event="bind", segment=i))

    def force_vector(self):
        if self.active is None:
            return np.zeros(3)
        d = self.sim.d
        wp.launch(E.reaction, dim=1, device="cpu", inputs=[self.active,
            int(A.mujoco.mjtConstraint.mjCNSTR_EQUALITY),
            wp.array(d.efc_type.astype(np.int32)[None], device="cpu"),
            wp.array(d.efc_id.astype(np.int32)[None], device="cpu"), view(d.efc_force),
            self._reaction, self._rows])
        rows = int(self._rows.numpy()[0])
        if rows not in (0, 6):
            raise RuntimeError(f"pinch weld has {rows} constraint rows, expected 6")
        return self._reaction.numpy()[0].copy()

    def check_load(self, dt, grip_force):
        if not math.isfinite(grip_force) or grip_force < 0 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt and nonnegative finite current grip force required")
        if self.active is None:
            return False
        wp.launch(E.overload, dim=1, device="cpu", inputs=[dt, grip_force, self.coefficient,
            pinch.OVERLOAD_TIME, wp.vec3d(self.force_vector()), self.overload_s, self.peak_force,
            self._load, self._detached])
        self.overload_s, self.peak_force, force, limit = map(float, self._load.numpy()[0])
        if bool(self._detached.numpy()[0]):
            self.release("overload", force=force, limit=limit)
            return True
        return False

    def release(self, reason="open", **evidence):
        if self.active is None:
            return
        wp.launch(E.deactivate, dim=1, device="cpu", inputs=[self.active, active_view(self.sim)])
        self.events.append(dict(t=float(self.sim.d.time), event=reason, segment=self.segment,
                                overload_s=self.overload_s, **evidence))
        self.active = None
        self.broken = reason == "overload"


@contextmanager
def enabled():
    before = A.Bushings, A.Sim.switch_welds, top_grasp.PinchAttachment, top_grasp.PinchSpring
    A.Bushings, A.Sim.switch_welds = WarpBushings, switch_welds
    top_grasp.PinchAttachment, top_grasp.PinchSpring = WarpPinchAttachment, WarpPinchSpring
    try:
        yield
    finally:
        A.Bushings, A.Sim.switch_welds, top_grasp.PinchAttachment, top_grasp.PinchSpring = before
