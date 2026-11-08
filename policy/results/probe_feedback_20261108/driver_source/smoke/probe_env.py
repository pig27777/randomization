"""Paired side-grasp experiment on an explicitly loaded frozen content backend.

The policy sees only envelope dimensions and two noisy force/encoder readings.
No pinch attachment, synthetic pinch spring, reset after probing, or class input.
"""
from __future__ import annotations

import hashlib
import math
import time

import mujoco
import numpy as np

from model import Kappa
from top_grasp import (TopGrasp, THRESHOLD, contact_reactions, reaction_index,
                       table_contacts, task_scene)

FORCES = (6., 10., 14., 20.)
PROBE_FORCE = 3.
LIFT_HEIGHT = .25
CARRY_DISTANCE = .5
LIFT_SPEED = .1
ACCELERATION = 2.
CLOSE_SPEED = .025


def sensor_features(compression_mm, relaxed_force, seed):
    rng = np.random.default_rng(seed + 99991)
    return [float(compression_mm + rng.normal(0., .05)),
            float(relaxed_force + rng.normal(0., .10))]


def evaluate(spec, *, probe, selector, seed, trace=False):
    """selector(geometry, noisy_probe_or_None) -> target force; never receives spec."""
    wall = time.perf_counter()
    k = Kappa.from_dict(spec)
    if k.size[1] > .085:
        raise ValueError('diagnostic side grasp requires width <= 85 mm')
    sim, _, _ = task_scene(k, TopGrasp(w=.095), 0.)
    starts = {p: np.array([0., sign * (.095 / 2 + .002), max(.022, k.size[2] / 2 + .0001)])
              for p, sign in (('pA', -1), ('pB', 1))}
    for p, pos in starts.items():
        sim.set_pad(p, pos, np.zeros(3))
    mujoco.mj_forward(sim.m, sim.d)
    index = reaction_index(sim)
    dt = sim.m.opt.timestep
    check_every = round(.01 / dt)
    direction = float(np.random.default_rng(seed).uniform(0, 2 * math.pi))
    carry = CARRY_DISTANCE * np.array([math.cos(direction), math.sin(direction), 0.])
    carry_time = math.sqrt(2 * math.pi * CARRY_DISTANCE / ACCELERATION)
    phase, phase_t, t, n = 'settle', 0., 0., 0
    close = {p: 0. for p in starts}
    touched, filtered = {}, {p: 0. for p in starts}
    previous = {p: pos.copy() for p, pos in starts.items()}
    arm = np.zeros(3)
    squeeze = measured = max_force = lost = 0.
    target = PROBE_FORCE
    chosen = None
    outcome = reason = None
    com_rest = touch_gap = None
    max_squeeze = 0.
    probe_readings, phases, rows = [], [], []
    feedback = probe_raw = probe_digest = None
    decision_time = None
    lift_height = None
    release_squeeze = 0.
    table_force = np.zeros(3)
    probe_hash = hashlib.sha256()
    threshold = THRESHOLD[k.category]

    def gap():
        return float(sim.pad_pos('pB')[1] - sim.pad_pos('pA')[1] - .004)

    def go(name):
        nonlocal phase, phase_t
        phase, phase_t = name, t
        phases.append({'t': t, 'phase': name})

    def decide():
        nonlocal target, chosen, decision_time, probe_raw, feedback, probe_digest
        if probe:
            probe_raw = [float(np.mean([r[0] for r in probe_readings])),
                         float(np.mean([r[1] for r in probe_readings]))]
            feedback = sensor_features(*probe_raw, seed)
            probe_hash.update(sim.d.qpos.tobytes())
            probe_hash.update(sim.d.qvel.tobytes())
            probe_digest = probe_hash.hexdigest()
        chosen = float(selector(list(k.size), feedback))
        if chosen not in FORCES:
            raise ValueError(f'policy returned unsupported target force {chosen}')
        target = chosen
        decision_time = t
        go('squeeze')

    while outcome is None:
        tau = t - phase_t
        ctrl = .5 * sum(filtered.values())
        if phase == 'settle':
            if tau >= .3:
                com_rest = sim.com().copy()
                go('approach')
        elif phase == 'approach':
            for p in close:
                if p not in touched:
                    if filtered[p] > .05:
                        touched[p] = close[p]
                    else:
                        close[p] += CLOSE_SPEED * dt
            if len(touched) == 2:
                touch_gap = gap()
                max_squeeze = max(0., (touch_gap - .002) / 2)
                if probe:
                    go('probe_load')
                else:
                    decide()
            elif tau >= 8.:
                outcome, reason = 'grasp_fail', 'both fingertips did not touch'
        elif phase == 'probe_load':
            if ctrl >= .95 * PROBE_FORCE or tau >= 2. or squeeze >= max_squeeze:
                go('probe_hold')
        elif phase == 'probe_hold':
            if tau >= .20:
                go('probe_relax')
        elif phase == 'probe_relax':
            if tau >= .15:
                if not probe_readings:
                    raise RuntimeError('probe completed without sensor samples')
                decide()
        elif phase == 'squeeze':
            if ctrl >= .95 * target or tau >= 8. or squeeze >= max_squeeze:
                go('hold')
        elif phase == 'hold':
            if tau >= .2:
                go('lift')
        elif phase == 'lift':
            arm[2] = min(LIFT_SPEED * tau, LIFT_HEIGHT)
            if tau >= LIFT_HEIGHT / LIFT_SPEED + .2:
                lift_height = float((sim.com()[2] - com_rest[2]) * 1000)
                if table_contacts(sim):
                    outcome, reason = 'grasp_fail', 'object still contacts table after lift hold'
                else:
                    go('carry')
        elif phase == 'carry':
            a = min(tau / carry_time, 1.)
            progress = a - math.sin(2 * math.pi * a) / (2 * math.pi)
            arm = np.array([0., 0., LIFT_HEIGHT]) + progress * carry
            if tau >= carry_time + .3:
                if sim.com()[2] - com_rest[2] < .03:
                    outcome, reason = 'drop', 'COM returned near table after carry'
                else:
                    go('lower')
        elif phase == 'lower':
            arm = carry + [0., 0., max(LIFT_HEIGHT - .2 * tau, .005)]
            if table_force[2] >= .3 * sim.mass * 9.81 or arm[2] <= .005:
                release_squeeze = squeeze
                go('open')
        elif phase == 'open':
            squeeze = release_squeeze - min(.05 * tau, .03)
            if tau >= 1.1:
                offset = sim.com()[:2] - com_rest[:2] - carry[:2]
                c, s = math.cos(direction), math.sin(direction)
                local = np.array([[c, s], [-s, c]]) @ offset
                outcome = 'success' if np.all(np.abs(local) <= .1) else 'drop'
                reason = None if outcome == 'success' else 'COM outside target square'
                break

        if phase in ('probe_load', 'probe_hold', 'squeeze', 'hold', 'lift', 'carry', 'lower'):
            velocity = float(np.clip(.0025 * (target - ctrl), -CLOSE_SPEED, CLOSE_SPEED))
            squeeze = float(np.clip(squeeze + dt * velocity, -.005, max_squeeze))
        if len(touched) == 2:
            for p in close:
                close[p] = touched[p] + squeeze
        for p, sign in (('pA', 1), ('pB', -1)):
            pos = starts[p] + [0., sign * close[p], 0.] + arm
            sim.set_pad(p, previous[p], (pos - previous[p]) / dt)
            previous[p] = pos
        sim.step()
        t += dt
        n += 1
        forces, table_force = contact_reactions(sim, index)
        raw = {'pA': max(0., -float(forces['pA'][1])), 'pB': max(0., float(forces['pB'][1]))}
        for p in raw:
            limited = min(raw[p], max(2 * target, filtered[p] + 20.))
            filtered[p] += dt / (.005 + dt) * (limited - filtered[p])
        measured += dt / (.020 + dt) * (.5 * sum(raw.values()) - measured)
        max_force = max(max_force, measured)
        # First-event convention, with confirmed simulator failures retained as invalid evidence.
        if measured > threshold and outcome is None:
            outcome, reason = 'over_squeeze', f'{measured:.6f} N in {phase}'
        if n % check_every == 0:
            failure = sim.failure()
            if failure is not None:
                outcome, reason = 'sim_fail', failure
            if phase in ('lift', 'carry') and outcome is None:
                lost = lost + check_every * dt if max(raw.values()) < .05 else 0.
                if lost > .05:
                    outcome = 'grasp_fail' if phase == 'lift' else 'drop'
                    reason = 'both contacts lost for >50 ms'
            if probe and decision_time is None:
                probe_hash.update(np.array([t, gap(), measured, *filtered.values()], dtype=np.float64).tobytes())
            if phase == 'probe_relax' and t - phase_t >= .10:
                probe_readings.append(((touch_gap - gap()) * 1000., measured))
            if trace:
                rows.append(dict(t=t, phase=phase, force_N=measured, gap_mm=gap() * 1000,
                                 com=sim.com().tolist(), target_N=target))
        if t > 35. and outcome is None:
            outcome, reason = 'grasp_fail', 'execution timeout (35 s)'

    return dict(outcome=outcome, reason=reason, selected_force_N=chosen, feedback=feedback,
                probe_raw=probe_raw, probe_digest=probe_digest, decision_time_s=decision_time,
                max_grip_force_N=max_force, lift_height_mm=lift_height, sim_s=t,
                wall_s=time.perf_counter() - wall, phases=phases, trace=rows,
                threshold_N=threshold, dt_s=dt, category=k.category)
