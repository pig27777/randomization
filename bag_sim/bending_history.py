"""Stage 2 for the Newton backend: bending history through per-edge rest angle and stiffness.

VBD's bending energy per edge is linear, 0.5 * k * l * (theta - theta_rest)^2. After each frame
this module advances the same history as the MuJoCo backend (bag_model.BagModel):

  plastic    associative return map with isotropic hardening (permanent rest-angle change)
  knee       bilinear equilibrium moment, slope 1 below the knee angle and post_ratio above
  memory     Maxwell branch, moment m * S * (delta - z), z relaxing with recovery_tau_s
  crease     irreversible bounded softening S = K * (1 - damage(peak angle))

and writes each hinge as the linear spring that reproduces the full moment at the current angle:

  K_eff = S * (slope + m),   theta_rest = theta_rest0 + delta - M(delta) / K_eff

so the next frame starts from the right moment and tangent stiffness. Values change once per
frame (frame_dt_s), a one-frame lag for fast folds. With no history enabled K_eff = K and the
rest angle stays at its reference: stage 1 is reproduced exactly.

Angles use Newton's own dihedral definition on its edge ordering (o0, o1, v0, v1); history is
kept relative to Newton's reference rest angle, so the sign convention cancels.
"""
from __future__ import annotations

import numpy as np


def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def newton_dihedrals(q, ids):
    """Newton VBD's dihedral angle for edges given as (o0, o1, v0, v1) vertex ids."""
    x0, x1, x2, x3 = (q[ids[:, k]] for k in range(4))
    n1 = np.cross(x2 - x0, x3 - x0)
    n2 = np.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-20)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-20)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-20)
    return np.arctan2(np.einsum("ij,ij->i", np.cross(n1, n2), e), np.einsum("ij,ij->i", n1, n2))


class BendingHistory:
    """Per-hinge history; `update(q, dt)` returns the per-edge (rest angle, ke) to write."""

    def __init__(self, params, stiffness, yield_angles, knee, edge_ids, rest_angles, lengths):
        p = params
        self.plasticity = bool(p["plasticity"])
        self.memory_ratio = float(p["memory_ratio"])
        self.tau = float(p["recovery_tau_s"])
        self.knee = knee
        self.post = float(p["bending_post_ratio"])
        self.hardening = p["plastic_hardening_ratio"] / (1 - p["plastic_hardening_ratio"])
        self.crease = bool(p["crease_softening"])
        self.onset = np.deg2rad(p["crease_onset_deg"])
        self.width = np.deg2rad(p["crease_width_deg"])
        self.max_softening = float(p["crease_max_softening"])
        self.enabled = self.plasticity or self.memory_ratio > 0 or knee is not None or self.crease
        self.base = np.asarray(stiffness, float).copy()
        self.yield_angles = np.asarray(yield_angles, float)
        self.ids = np.asarray(edge_ids)
        self.rest0 = np.asarray(rest_angles, float).copy()
        self.lengths = np.asarray(lengths, float)
        n = len(self.base)
        self.last = self.rest0.copy()
        self.total = np.zeros(n)
        self.plastic = np.zeros(n)
        self.accumulated = np.zeros(n)
        self.memory = np.zeros(n)
        self.peak = np.zeros(n)
        self.damage = np.zeros(n)
        self.plastic_dissipated_j = 0.0

    def _equilibrium(self, elastic):
        if self.knee is None:
            return elastic, np.ones_like(elastic)
        a = np.abs(elastic)
        above = a > self.knee
        moment = np.sign(elastic) * np.where(above, self.knee + self.post * (a - self.knee), a)
        return moment, np.where(above, self.post, 1.0)

    def update(self, q, dt):
        angle = newton_dihedrals(q, self.ids)
        self.total += _wrap(angle - self.last)
        self.last = angle
        delta = self.total
        if self.plasticity:
            trial = delta - self.plastic
            increment = np.maximum(np.abs(trial) - self.yield_angles - self.hardening * self.accumulated, 0) / (1 + self.hardening)
            self.plastic = self.plastic + increment * np.sign(trial)
            self.accumulated = self.accumulated + increment
            self.plastic_dissipated_j += float(np.sum(self.base * self.yield_angles * increment))
        if self.memory_ratio > 0:
            self.memory += -np.expm1(-dt / self.tau) * (delta - self.memory)
        if self.crease:
            self.peak = np.maximum(self.peak, np.abs(delta))
            self.damage = self.max_softening * -np.expm1(-np.maximum(self.peak - self.onset, 0) / self.width)
        softened = self.base * (1 - self.damage)
        equilibrium, slope = self._equilibrium(delta - self.plastic)
        moment = softened * (equilibrium + self.memory_ratio * (delta - self.memory))
        k_eff = softened * (slope + self.memory_ratio)
        rest = self.rest0 + delta - moment / np.maximum(k_eff, 1e-30)
        return _wrap(rest), k_eff / self.lengths

    def report(self):
        return {"max_permanent_crease_deg": float(np.rad2deg(np.max(np.abs(self.plastic), initial=0))),
                "max_memory_deg": float(np.rad2deg(np.max(np.abs(self.memory), initial=0))),
                "max_crease_softening": float(self.damage.max(initial=0)),
                "plastic_dissipated_j": self.plastic_dissipated_j}
