"""C1 (test/bag_model/REPORT.md round 1, compressed): Newton kraft bag with a full / loose hard box, R2 and R3
of test/content_model/README.md, once each, as the qualitative reference for the MuJoCo loose-bag
representations (mj_loose.py). Newton 1.6.1 (real2sim/bubble_mailer/.venv; 1.7.0 cannot build the bag, I-23).

Scene as proxy_eval/objects.py X7 / X8 (wrap_generic, box content, 40 mm rounded pads that touch only the
bag); the bag is the kraft mesh (length along y, open mouth at +y, flap to +0.265 m). R2 grips the package
end at the flap (the mouth end, so the box slides towards the closed bottom) with the end 50 mm over the edge
of a table box; R3 grips the long sides at the middle and carries 300 mm along y. Pads are driven by
proxy_eval/evaluate.Runner (5 ms periods, contact-force probes); closing and the 20 N hold copy
contents/segment/protocols.Closer (which needs mujoco, not installed here).

Usage: <newton python> -B c1_newton.py r2|r3 full|loose [--out c1_newton.jsonl]
"""
from __future__ import annotations

import argparse
import os
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "proxy_eval"), str(HERE), str(HERE / "legacy")]
import newton  # noqa: E402
from objects import PARK, _add_pads  # noqa: E402
from evaluate import PERIOD, Runner  # noqa: E402
from bag_model import parameters  # noqa: E402
from contents import normalize_contents  # noqa: E402
from newton_bag import content_shape_config, load_profile_newton  # noqa: E402
from wrap import winding_number, wrap_generic  # noqa: E402

OUT = HERE / "outputs" / "round1"
BOXES = {"full": ((0.15, 0.20, 0.03), 0.5), "loose": ((0.08, 0.12, 0.03), 0.3)}   # long side along the bag (y)
PAD_HALF = 0.02          # pad half side; pad half thickness 5 mm (proxy_eval/task.py PAD_HALF)
PAD_T = 0.005
GAP, SETTLE, F_GRIP, F_TOUCH, F_TAU = 0.002, 0.3, 20.0, 0.05, 0.005
SPEED_FLAG = 1.0


class C1Scene:
    """Attributes proxy_eval/evaluate.Runner expects, for the kraft bag + box (+ table box for R2)."""

    def __init__(self, fill, table_y_max=None):
        profile = json.loads((HERE / "profiles" / "kraft.json").read_text())
        p = profile["params"]
        radius = p["collision_radius_m"]
        with np.load(HERE / profile["geometry"]) as g:
            rest, faces, pid = np.array(g["vertices"], float), np.array(g["faces"]), np.array(g["panel_ids"])
        size, mass = BOXES[fill]
        item = normalize_contents([dict(shape="box", size_m=list(size), mass_kg=mass)])
        initial, report = wrap_generic(rest, rest.copy(), faces, pid, p["self_contact_front_panel_ids"],
                                       p["self_contact_back_panel_ids"], item[0], radius)
        center = item[0]["position"]
        params = parameters(p)
        built = {}

        def extra(builder, contact):
            cfg = content_shape_config(contact, params, density=mass / float(np.prod(size)))
            body = builder.add_body(xform=wp.transform(wp.vec3(*map(float, center)), wp.quat_identity()), label="content")
            half = np.asarray(size) / 2
            built["content_shape"] = builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
            built["bodies"] = [body]
            mu = max(0.0, 2 * p["table_friction"] - p["friction"])     # as NewtonBag's ground (I-23)
            if table_y_max is not None:
                tcfg = content_shape_config(contact, params, mu=mu, voxel_m=0.004)
                y0 = -0.45
                built["table"] = builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, (y0 + table_y_max) / 2, -0.025),
                                                                                wp.quat_identity()),
                                                       hx=0.25, hy=(table_y_max - y0) / 2, hz=0.025, cfg=tcfg)
                builder.add_shape_collision_filter_pair(built["table"], built["content_shape"])
            pad_cfg = newton.ModelBuilder.ShapeConfig(ke=float(contact["shape_ke"]), kd=float(contact["shape_kd"]),
                                                      mu=float(p["friction"]), density=0.)
            built["pads"] = _add_pads(builder, pad_cfg)
            for s in built["pads"][1]:
                builder.add_shape_collision_filter_pair(s, built["content_shape"])
                if "table" in built:
                    builder.add_shape_collision_filter_pair(s, built["table"])

        self.bag = bag = load_profile_newton("kraft", initial_vertices=initial, extra=extra,
                                             support=table_y_max is None)
        self.faces = faces
        self.wrap_report = report
        self.bodies = built["bodies"]
        self.pads, self.pad_shapes = built["pads"]
        self.particles = slice(0, bag.start + len(bag.rest))
        self.bag_particles = slice(bag.start, bag.start + len(bag.rest))
        self.model, self.solver, self.pipeline, self.contacts = bag.model, bag.solver, bag.pipeline, bag.contacts
        self.control = bag.control
        self.substeps, self.dt = bag.substeps, bag.dt
        self.drag = bag.params["velocity_damping_per_s"]
        m = self.model
        self.particle_mass = m.particle_mass.numpy().astype(float)[self.bag_particles]
        self.body_mass = m.body_mass.numpy().astype(float)[self.bodies]
        self.mass = float(self.particle_mass.sum() + self.body_mass.sum())
        self.box_half = np.asarray(size) / 2

    @property
    def s0(self):
        return self.bag.s0

    @s0.setter
    def s0(self, v):
        self.bag.s0 = v

    @property
    def s1(self):
        return self.bag.s1

    @s1.setter
    def s1(self, v):
        self.bag.s1 = v

    # ---------- reads
    def bag_q(self):
        q = self.s0.particle_q.numpy()[self.bag_particles].astype(float)
        if not np.isfinite(q).all():
            raise FloatingPointError("non-finite bag state")
        return q

    def box_pose(self):
        bq = self.s0.body_q.numpy()[self.bodies[0]].astype(float)
        if not np.isfinite(bq).all():
            raise FloatingPointError("non-finite content state")
        return bq[:3], bq[3:]

    def pkg_com(self, q=None):
        q = self.bag_q() if q is None else q
        return (self.particle_mass @ q + self.body_mass[0] * self.box_pose()[0]) / self.mass

    def content_outside(self, q=None):
        """Fraction of (box centre, 8 corners pulled 3 mm in) outside the bag, by winding number; the open
        mouth/flap boundary is closed with a fan (as mj_loose.LooseSim.bag_surface for flex)."""
        q = self.bag_q() if q is None else q
        pos, quat = self.box_pose()
        x, y, z, w = quat
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        h = self.box_half - 0.003
        pts = np.array([[0, 0, 0]] + [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * h
        pts = pos + pts @ R.T
        e = np.sort(np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1)
        u, c = np.unique(e, axis=0, return_counts=True)
        loop = np.unique(u[c == 1])
        cen = q[loop].mean(0)
        loop = loop[np.argsort(np.arctan2(q[loop, 2] - cen[2], q[loop, 0] - cen[0]))]
        v = np.vstack([q, cen])
        f = np.vstack([self.faces, np.c_[loop, np.roll(loop, -1), np.full(len(loop), len(q))]])
        inside = np.abs(winding_number(v, f, pts)) >= 0.5
        return float(1 - inside[0]), float(1 - inside[1:].mean())


class Closer:
    """contents/segment/protocols.Closer (2 pads): approach at `speed` to touch, squeeze together slowing near
    f_stop, freeze at f_stop or `squeeze`; hold() keeps the weaker pad at f_stop with a slow gap correction."""

    def __init__(self, speed, squeeze, f_stop=F_GRIP):
        self.speed, self.squeeze, self.f_stop = speed, squeeze, f_stop
        self.s, self.touch, self.f = [0.0, 0.0], [None, None], [0.0, 0.0]
        self.s_sq, self.frozen, self.hold_sq = 0.0, False, 0.0

    def __call__(self, dt, forces):
        a = dt / (F_TAU + dt)
        self.f = [f + a * (g - f) for f, g in zip(self.f, forces)]
        if self.frozen:
            return self.s
        if None in self.touch:
            for i in (0, 1):
                if self.touch[i] is None:
                    if self.f[i] > F_TOUCH:
                        self.touch[i] = self.s[i]
                    else:
                        self.s[i] += self.speed * dt
        elif min(self.f) >= self.f_stop or self.s_sq >= self.squeeze:
            self.frozen = True
        else:
            slow = float(np.clip(1 - min(self.f) / self.f_stop, 0.02, 1.0))
            self.s_sq = min(self.s_sq + self.speed * slow * dt, self.squeeze)
            self.s = [t + self.s_sq for t in self.touch]
        return self.s

    def hold(self, dt, forces):
        a = dt / (F_TAU + dt)
        self.f = [f + a * (g - f) for f, g in zip(self.f, forces)]
        gain = min(.001, max(self.s_sq, 1e-8) / (.05 * self.f_stop))
        v = np.clip(gain * (self.f_stop - min(self.f)), -.005, .005)
        self.hold_sq = float(np.clip(self.hold_sq + v * dt, -self.s_sq, self.squeeze - self.s_sq))
        return [s + self.hold_sq for s in self.s]


def run(test, fill):
    t_wall = time.perf_counter()
    if test == "r2":
        probe_y_end = 0.265                       # flap tip of geometry/kraft.npz (package end, +y)
        sc = C1Scene(fill, table_y_max=probe_y_end - 0.05)
    else:
        sc = C1Scene(fill)
    for _ in range(round(0.5 / .01)):             # the bag model's own settle (self-contact ramp)
        sc.bag.step()
    q = sc.bag_q()
    lo, hi = q.min(0), q.max(0)
    radius = sc.bag.params["collision_radius_m"]
    if test == "r2":
        y_end = hi[1] + radius
        grip = np.array([0.5 * (lo[0] + hi[0]), y_end - PAD_HALF])
        near = q[np.abs(q[:, 1] - grip[1]) < PAD_HALF + 0.01]
        top = float(near[:, 2].max()) + radius
        d = np.array([0.0, 0.0, 1.0])                  # closing axis: pad 0 above, pad 1 below
        base = np.array([[*grip, 0.0], [*grip, 0.0]])
        s_plus, s_minus = top + GAP + PAD_T, GAP + PAD_T
        rot = wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), math.pi / 2)
        L = hi[1] - lo[1]
        lift_h = max(0.1, L + 0.05)
        phases = [("lift", lift_h / 0.05, lambda t: np.array([0, 0, min(0.05 * t, lift_h)])),
                  ("hold", 2.0, lambda t: np.array([0, 0, lift_h]))]
        squeeze = 0.01
    else:
        c = 0.5 * (lo + hi)
        h = hi[2] + radius
        d = np.array([1.0, 0.0, 0.0])                  # across the width (x)
        base = np.array([[c[0], c[1], h / 2], [c[0], c[1], h / 2]])
        s_plus, s_minus = hi[0] + radius - c[0] + GAP + PAD_T, c[0] - lo[0] + radius + GAP + PAD_T
        rot = wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), -math.pi / 2)
        from_profile = lambda t, a=2.0, ta=0.3, tc=0.2: (0.5 * a * t * t if t < ta else
            0.5 * a * ta * ta + a * ta * (t - ta) if t < ta + tc else
            0.5 * a * ta * ta + a * ta * tc + a * ta * (t - ta - tc) - 0.5 * a * (t - ta - tc) ** 2 if t < 2 * ta + tc
            else a * ta * ta + a * ta * tc)                                   # protocols.move_profile
        up = 0.05
        phases = [("lift", 1.0, lambda t: np.array([0, 0, min(0.05 * t, up)])),
                  ("pre", 0.5, lambda t: np.array([0, 0, up])),
                  ("move", 0.8, lambda t: np.array([0, from_profile(t), up])),
                  ("hold", 1.0, lambda t: np.array([0, from_profile(0.8), up]))]
        squeeze = 0.45 * (hi[0] - lo[0])
    runner = Runner(sc, rot)
    closer = Closer(0.025, squeeze)

    def pad_pos(s, o):
        out = base.copy()
        out[0] += (s_plus - s[0]) * d + o
        out[1] -= (s_minus - s[1]) * d
        out[1] += o
        return out

    pads = pad_pos(closer.s, np.zeros(3))
    runner.p_to.assign(pads.astype(np.float32))
    prev_imp = runner.impulse.numpy().copy()
    rows, ev, fail, flag = [], {}, None, None
    t, closed, t_phase, phase_i, track = 0.0, None, None, 0, None
    prev = None
    try:
        while t < float(os.environ.get("C1_TMAX", 40)):  # C1_TMAX: smoke runs only
            if t < SETTLE:
                s, o = closer.s, np.zeros(3)
            elif closed is None or t < closed + 0.2:
                s, o = closer(PERIOD, f), np.zeros(3)
                if closer.frozen and closed is None:
                    closed = ev["closed"] = t
                if t - SETTLE > 8.0 and closed is None:
                    closed = ev["close_timeout"] = t
                    closer.frozen = True
            else:
                if t_phase is None:
                    t_phase = ev[phases[0][0]] = t
                while phase_i < len(phases) and t - t_phase >= phases[phase_i][1] - 1e-9:
                    t_phase += phases[phase_i][1]
                    phase_i += 1
                    if phase_i < len(phases):
                        ev[phases[phase_i][0]] = t_phase
                if phase_i >= len(phases):
                    break
                o = phases[phase_i][2](t - t_phase)
                s = closer.hold(PERIOD, f)
            pads = pad_pos(s, o)
            runner.period(pads)
            t += PERIOD
            imp = runner.impulse.numpy()
            force = -(imp - prev_imp) / PERIOD
            prev_imp = imp.copy()
            f = [float(force[sc.pad_shapes[0]] @ d), float(-(force[sc.pad_shapes[1]] @ d))]
            if runner.periods % 2:
                continue
            q = sc.bag_q()
            com = sc.pkg_com(q)
            grip = pads.mean(0)
            box = sc.box_pose()[0]
            if prev is not None:
                hh = 2 * PERIOD
                com_v = (com - prev[1]) / hh
                node = np.linalg.norm((q - prev[0]) / hh - com_v, axis=1).max()
                if node > SPEED_FLAG and flag is None:
                    flag = f"bag node {node:.2f} m/s relative to the package COM at {t:.2f} s"
            prev = (q, com)
            if "lift" in ev and track is None:
                i = int(np.argmin(np.linalg.norm(q - grip, axis=1)))
                track = (i, q[i] - grip, box - grip)
            row = dict(t=t, F=float(np.mean(f)), F_min=float(min(f)), grip=grip, com=com, box=box,
                       bag_zmin=float(q[:, 2].min()))
            if track is not None:
                row["bag_slip"] = float(np.linalg.norm(q[track[0]] - grip - track[1]))
                row["content_rel"] = box - grip - track[2]
            rows.append(row)
            if runner.periods % 20 == 0:
                c_out, _ = sc.content_outside(q)
                if c_out > 0:
                    fail = f"content outside the bag at {t:.2f} s"
                    break
    except FloatingPointError as err:
        fail = str(err)
    wp.synchronize()
    out = dict(test=test.upper(), fill=fill, backend="newton-1.6.1", fail=fail, speed_flag=flag, sim_t=t,
               wall_s=time.perf_counter() - t_wall, substep_s=sc.dt, membrane_scale=float(sc.bag.membrane_scale),
               events={k: round(v, 3) for k, v in ev.items()}, pkg_mass=sc.mass)
    if fail is None:
        out["outside_end"] = sc.content_outside()
    if fail is not None or "hold" not in ev:
        out["completed"] = False
        return out
    hold = [r for r in rows if r["t"] >= ev["hold"]]
    lifted = [r for r in rows if r["t"] >= ev["lift"] and "content_rel" in r]
    last = [r for r in rows if r["t"] >= rows[-1]["t"] - 0.5]
    crel = np.array([r["content_rel"] for r in lifted])
    out.update(completed=True, grip_force_hold=float(np.mean([r["F"] for r in last])),
               dropped=bool(last[-1]["F_min"] < F_TOUCH),
               pkg_com_rise_mm=float(np.mean([r["com"][2] for r in last]) - rows[0]["com"][2]) * 1e3,
               content_com_rise_mm=float(np.mean([r["box"][2] for r in last]) - rows[0]["box"][2]) * 1e3,
               bag_slip_mm=float(max(r["bag_slip"] for r in lifted)) * 1e3,
               content_rel_max_mm=float(np.linalg.norm(crel, axis=1).max()) * 1e3,
               content_rel_end_mm=(crel[-1] * 1e3).round(1).tolist(),
               pad_gap_mm=float(abs((pads[0] - pads[1]) @ d)) * 1e3 - 2 * PAD_T * 1e3,
               touch_mm=[None if x is None else round(x * 1e3, 1) for x in closer.touch],
               squeeze_mm=closer.s_sq * 1e3)
    if test == "r2":
        out["droop_bag"] = float(np.mean([r["grip"][2] - r["bag_zmin"] for r in hold[-50:]]) / L)
    else:
        mv = [r for r in rows if r["t"] >= ev["move"]]
        F = np.array([r["F"] for r in mv])
        after = np.array([r["com"] - r["grip"] for r in hold])
        dev = np.linalg.norm(after - after[-20:].mean(0), axis=1)
        out.update(com_swing_mm=float(dev.max()) * 1e3, force_p2p=float(F.max() - F.min()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["r2", "r3"])
    ap.add_argument("fill", choices=sorted(BOXES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wp.config.quiet = True
    res = run(a.test, a.fill)
    line = json.dumps(res, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(line, flush=True)
    if a.out:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / a.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
