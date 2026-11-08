"""Engine-independent part of the grasp task in test/policy/README.md: grasp-parameter mapping,
gripper closing, pad trajectory and outcome classification. No simulator imports.

Choices the README leaves open (also listed in test/proxy_eval/REPORT.md; to be confirmed by the
overall session so the reduced-model backend uses the same ones):
- pads start 5 mm outside the observed top-view box along the closing line and approach at 50 mm/s
  each; closing then follows test/content_model/README.md (each pad waits at touch > 0.05 N, then both
  squeeze at 25 mm/s each, slowing linearly over the last 30 % of F, and stop at F or the 50 N cap);
- pad centre height max(h / 2, 22 mm), so a 40 mm pad on a thin package clears the table by 2 mm;
- lift 0.10 m at the lift speed, hold 0.2 s; minimum-jerk transport of 0.5 m along the box long side
  (+x of the observed box) with peak acceleration a, hold 0.3 s; lower 0.09 m at the lift speed (release
  10 mm above the grasp height, so a package that slipped is not pressed into the table);
  open 10 mm per pad in 0.2 s; settle 0.3 s;
- the target is the 0.2 x 0.2 m square centred 0.5 m from the observed box centre along the transport;
- closing stops after CLOSE_TIMEOUT (4 s) at the latest and the carry starts with whatever grip there is;
- grip force = mean of the two pads' normal forces; over-squeeze when it exceeds the class limit at
  any time; grasp_fail when the package centre of mass rose less than half the lift; drop when, during
  the carry, both pads lose contact for 30 ms or the centre of mass falls back below a quarter of the lift;
- an intact carry that lands outside the target has no outcome of its own in the README: it is
  reported as "drop" with detail "misplaced".
"""
from __future__ import annotations

import math

import numpy as np

PAD_HALF = (0.02, 0.005, 0.02)   # tangential, closing, vertical half extents (m)
PAD_FILLET = 0.004
F_CAP = 50.0
F_TOUCH = 0.05
F_TAU = 0.005
APPROACH_SPEED = 0.05             # per pad, before touch
SQUEEZE_SPEED = 0.025             # per pad, after both touch
SLOW_BAND = 0.3                   # squeeze slows linearly over the last 30 % of the target force
START_GAP = 0.005
PAD_MIN_Z = PAD_HALF[2] + 0.002
LIFT_H = 0.10
RELEASE_H = 0.01                  # pads stop this far above the grasp height before opening
TRAVEL = 0.5
TARGET_HALF = 0.1
HOLD_AFTER_CLOSE, HOLD_AFTER_LIFT, HOLD_AFTER_MOVE = 0.2, 0.2, 0.3
OPEN_S, OPEN_DIST, FINAL_SETTLE = 0.2, 0.01, 0.3
CLOSE_TIMEOUT = 4.0
LOST_CONTACT_S = 0.03
SQUEEZE_LIMIT = {"compressible": 15.0, "bendable": 25.0, "hard": F_CAP}

PARAM_RANGES = {  # name: (low, high)
    "u": (-0.5, 0.5), "v": (-0.5, 0.5), "psi": (0.0, math.pi), "force_N": (2.0, 50.0),
    "lift_speed": (0.05, 0.3), "max_accel": (0.5, 4.0),
}


def check_params(p):
    for k, (lo, hi) in PARAM_RANGES.items():
        if k not in p:
            raise KeyError(f"grasp parameter {k!r} missing")
        if not lo <= p[k] <= hi:
            raise ValueError(f"grasp parameter {k}={p[k]} outside [{lo}, {hi}]")


def top_view_box(xy):
    """Minimum-area oriented rectangle of points (n, 2): center, (length, width), yaw of the long side."""
    best = None
    for yaw in np.radians(np.arange(0.0, 90.0, 0.25)):
        c, s = math.cos(yaw), math.sin(yaw)
        local = xy @ np.array([[c, -s], [s, c]])
        lo, hi = local.min(0), local.max(0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, yaw, lo, hi)
    _, yaw, lo, hi = best
    size = hi - lo
    if size[1] > size[0]:
        yaw, size, lo, hi = yaw + math.pi / 2, size[::-1], np.array([lo[1], -hi[0]]), np.array([hi[1], -lo[0]])
    c, s = math.cos(yaw), math.sin(yaw)
    center = np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
    return center, size, yaw


def grasp_frame(box_center, box_size, box_yaw, height, p):
    """World grasp point, closing direction d (unit, xy), pad start distance per side and pad height.

    Pad +1 starts at grasp + s_plus * d and moves along -d; pad -1 starts at grasp - s_minus * d."""
    long_axis = np.array([math.cos(box_yaw), math.sin(box_yaw)])
    short_axis = np.array([-long_axis[1], long_axis[0]])
    grasp = box_center + p["u"] * box_size[0] * long_axis + p["v"] * box_size[1] * short_axis
    d = math.cos(p["psi"]) * long_axis + math.sin(p["psi"]) * short_axis
    corners = np.array([box_center + a * box_size[0] / 2 * long_axis + b * box_size[1] / 2 * short_axis
                        for a in (-1, 1) for b in (-1, 1)])
    along = (corners - grasp) @ d
    s_plus = max(float(along.max()), 0.0) + START_GAP + PAD_HALF[1]
    s_minus = max(float(-along.min()), 0.0) + START_GAP + PAD_HALF[1]
    return grasp, d, (s_plus, s_minus), max(height / 2, PAD_MIN_Z), long_axis


class Closer:
    """Each pad approaches until it touches (F > F_TOUCH) and waits; once both touch, both squeeze until
    the weaker pad reaches the target force (or the cap), slowing down near it. Forces are low-pass
    filtered (F_TAU). Returns each pad's inward travel (m). Rules of contents/segment/protocols.py, except that
    the target is the grasp force F and the slowdown acts over the last SLOW_BAND of it."""

    def __init__(self, target):
        self.target = min(target, F_CAP)
        self.s = [0.0, 0.0]
        self.touch = [None, None]
        self.squeeze = 0.0
        self.f = [0.0, 0.0]
        self.done = False

    def __call__(self, dt, forces):
        a = dt / (F_TAU + dt)
        self.f = [f + a * (g - f) for f, g in zip(self.f, forces)]
        if self.done:
            return self.s
        if None in self.touch:
            for i in (0, 1):
                if self.touch[i] is None:
                    if self.f[i] > F_TOUCH:
                        self.touch[i] = self.s[i]
                    else:
                        self.s[i] += APPROACH_SPEED * dt
        elif min(self.f) >= self.target:
            self.done = True
        else:
            slow = float(np.clip((1 - min(self.f) / self.target) / SLOW_BAND, 0.05, 1.0))
            self.squeeze += SQUEEZE_SPEED * slow * dt
            self.s = [t + self.squeeze for t in self.touch]
        return self.s


def min_jerk(t, T, D):
    s = min(max(t / T, 0.0), 1.0)
    return D * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def carry_schedule(p):
    """Phase start times after closing ends, and the pad centre offset (dx along transport, dz) at time t."""
    t_lift = LIFT_H / p["lift_speed"]
    t_move = math.sqrt(10 / math.sqrt(3) * TRAVEL / p["max_accel"])
    phases = [("hold_close", HOLD_AFTER_CLOSE), ("lift", t_lift), ("hold_lift", HOLD_AFTER_LIFT), ("move", t_move),
              ("hold_move", HOLD_AFTER_MOVE), ("lower", (LIFT_H - RELEASE_H) / p["lift_speed"]), ("open", OPEN_S), ("settle", FINAL_SETTLE)]
    starts, t = {}, 0.0
    for name, dur in phases:
        starts[name] = (t, t + dur)
        t += dur

    def offset(tc):
        lift = min(max(tc - starts["lift"][0], 0.0) * p["lift_speed"], LIFT_H)
        lower = min(max(tc - starts["lower"][0], 0.0) * p["lift_speed"], LIFT_H - RELEASE_H)
        dx = min_jerk(tc - starts["move"][0], t_move, TRAVEL)
        opening = min(max(tc - starts["open"][0], 0.0) / OPEN_S, 1.0) * OPEN_DIST
        return dx, lift - lower, opening

    return starts, offset, t


def classify(events, final_com_xy, target_center, transport_dir):
    """Outcome from the recorded events (dict of flags) and the final centre of mass."""
    if events.get("sim_fail"):
        return "sim_fail", events["sim_fail"]
    if events.get("over_squeeze"):
        return "over_squeeze", events["over_squeeze"]
    if events.get("grasp_fail"):
        return "grasp_fail", events["grasp_fail"]
    if events.get("drop"):
        return "drop", events["drop"]
    rel = np.asarray(final_com_xy) - target_center
    side = np.array([-transport_dir[1], transport_dir[0]])
    if abs(rel @ transport_dir) <= TARGET_HALF and abs(rel @ side) <= TARGET_HALF:
        return "success", ""
    return "drop", f"misplaced: centre of mass {np.linalg.norm(rel) * 1e3:.0f} mm from target centre"
