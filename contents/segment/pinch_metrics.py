"""Version-independent top-pinch measurements (v1.3 acceptance; also run on the v1.2 package).

HangObserver is passed as `observer` to top_grasp.evaluate; it reads only the simulation state and the
pinch frame, so the same code measures the v1.2 weld and the v1.3 point connections.
e4_monotonicity applies the reviewed E4 criteria (2026-11-06) to scan rows.
"""
from __future__ import annotations

import math

import numpy as np

LIFT_HEIGHT, CARRY_DISTANCE, STOP_WINDOW = .250, .5, .3


def snapshot(sim, binding, pinch):
    d = sim.d
    bodies = [sim.seg_bodies[i] for i in binding.top]
    axis = np.mean([d.xmat[b].reshape(3, 3)[:, 2] for b in bodies], axis=0)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    lowest = min(float((d.xpos[b] + (signs * s.half) @ d.xmat[b].reshape(3, 3).T)[:, 2].min())
                 for b, s in zip(sim.seg_bodies, sim.desc.segments))
    masses = sim.m.body_mass[sim.seg_bodies]
    com_v = (sim.seg_velocity() * masses[:, None]).sum(0) / masses.sum()
    com = sim.com()
    return dict(com=com.tolist(), pinch=pinch.tolist(),
                com_pinch_horizontal_mm=float(np.linalg.norm(com[:2] - pinch[:2]) * 1000),
                tilt_deg=float(math.degrees(math.acos(min(1., axis[2] / np.linalg.norm(axis))))),
                lowest_point_mm=lowest * 1000, com_speed_mm_s=float(np.linalg.norm(com_v) * 1000),
                attached=binding.active is not None)


class HangObserver:
    """Hang posture at the end of the lift hold and of an optional hang; COM swing after the carry stops.

    The carry is the protocol's cycloid, ending with zero velocity at carry_T; the swing is the largest
    horizontal deviation of (COM - pinch point) from its mean over the following 0.3 s window.
    """

    def __init__(self, grasp):
        self.lift_T = LIFT_HEIGHT / grasp.get('v_lift', .1)
        self.carry_T = math.sqrt(2 * math.pi * CARRY_DISTANCE / grasp.get('a_max', 2.))
        self.phase, self.phase_t = None, 0.
        self.lift_end = self.hang_end = None
        self.window = []

    def __call__(self, sim, binding, phase, t, force):
        if phase != self.phase:
            self.phase, self.phase_t = phase, t
        if sim.desc.rigid:
            return
        tau = t - self.phase_t
        pinch = sim.d.mocap_pos[binding.mocap].copy()
        if phase == 'lift' and tau >= self.lift_T:
            self.lift_end = snapshot(sim, binding, pinch)
        elif phase == 'hang':
            self.hang_end = snapshot(sim, binding, pinch)
        elif phase == 'carry' and tau >= self.carry_T - 1e-9 and binding.active is not None:
            self.window.append(sim.com()[:2] - pinch[:2])

    def summary(self):
        swing = None
        if len(self.window) >= 2:
            w = np.array(self.window)
            swing = float(np.linalg.norm(w - w.mean(0), axis=1).max() * 1000)
        return dict(lift_end=self.lift_end, hang_end=self.hang_end, stop_swing_mm=swing,
                    stop_window_samples=len(self.window))


def held(result):
    """Bound before the carry, never released before it, and off the table after the lift hold."""
    t = next((p['t'] for p in result['phases'] if p['phase'] == 'carry'), None)
    return bool(t is not None and result['lift_table'] and not result['lift_table']['contact']
                and any(e['event'] == 'bind' and e['t'] <= t for e in result['pinch_events'])
                and not any(e['event'] != 'bind' and e['t'] <= t for e in result['pinch_events']))


def e4_monotonicity(rows, scales=(.25, .5, 1.)):
    rows = [r for r in rows if not r['execution_error']]  # callers fail the run on execution errors
    key = lambda r: (r['grasp']['w'], r['grasp']['d'], r['grasp']['F'], r['seed'])
    categories, violations = {}, []
    for c in ('bendable', 'compressible'):
        by = {s: {key(r): r for r in rows if r['category'] == c and r['coefficient_scale'] == s} for s in scales}
        holds = {s: {k for k, r in part.items() if held(r['result'])} for s, part in by.items()}
        counts = {s: sum(r['result']['outcome'] == 'success' for r in part.values()) for s, part in by.items()}
        bad = [k for k in by[scales[0]]
               if any(by[lo][k]['result']['outcome'] == 'success' and by[hi][k]['result']['outcome'] != 'success'
                      for lo, hi in zip(scales, scales[1:]))]
        violations += [dict(category=c, w=k[0], d=k[1], F=k[2], seed=k[3]) for k in bad]
        categories[c] = dict(success_counts={str(s): v for s, v in counts.items()},
                             success_region=counts[scales[-1]] > 0,
                             success_count_nondecreasing=all(counts[a] <= counts[b] for a, b in zip(scales, scales[1:])),
                             held_counts={str(s): len(v) for s, v in holds.items()},
                             held_sets_nested=all(holds[a] <= holds[b] for a, b in zip(scales, scales[1:])),
                             counterexample_actions=len(bad), paired_actions=len(by[scales[0]]))
    pairs = sum(c['paired_actions'] for c in categories.values())
    hard = [r for r in rows if r['category'] == 'hard']
    data = dict(categories=categories, counterexamples=violations, paired_actions=pairs,
                counterexample_rate=len(violations) / pairs if pairs else None,
                hard_all_rejected=bool(hard) and all(r['result']['outcome'] == 'grasp_fail' for r in hard),
                confirmed_failures=sum(r['confirmed_failure'] for r in rows))
    data['passed'] = (all(c['success_region'] and c['success_count_nondecreasing'] and c['held_sets_nested']
                          for c in categories.values())
                      and data['counterexample_rate'] <= .05 and data['hard_all_rejected']
                      and data['confirmed_failures'] == 0)
    return data
