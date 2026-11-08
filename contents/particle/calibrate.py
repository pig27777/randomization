"""Closed-loop gains: run the virtual squeeze, compare with the target, correct, repeat.

Usage: calibrate.py <shape> [--rounds 4] [--out results_particle]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import particle_soft as ps
import particle_tests as pt

HERE = Path(__file__).resolve().parent

TARGET_HYSTERESIS = 0.5
EPS2 = 0.30


def measure(shape, x, out):
    meta = pt.BUILDERS[shape](x)[4]
    res = pt.press(shape, x, pt.platen_half(shape, x), (0, 0), 700.0, meta['thick'] * EPS2 * 1.03, creep_f=x.p_creep * meta['area'])
    load = res['load']
    area, thick = res['area'], res['thick']
    p_at = lambda eps: float(np.interp(eps * thick, load[:, 0], load[:, 1]) / area)
    target2 = x.p_knee + (EPS2 - x.eps_knee) / (x.eps_lock - x.eps_knee) * (x.p_lock - x.p_knee)
    return dict(p_knee=p_at(x.eps_knee), p2=p_at(EPS2), target2=target2, creep=res.get('creep_depth'), creep_target=thick * x.creep_eps,
                hysteresis=res.get('hysteresis'), residual=res.get('residual'), drop=res['monotonic_drop'], reached=load[-1, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shape')
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--out', default=str(HERE / 'results_particle'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    gains = dict(g_knee=1.0, g_lock=1.0, g_creep=1.0, g_hyst=1.0)
    history = []
    for round_id in range(args.rounds):
        x = ps.SoftX(**{**pt.SHAPE_DEFAULTS.get(args.shape, {}), **gains})
        try:
            m = measure(args.shape, x, out)
        except FloatingPointError as err:
            print(args.shape, round_id, 'unstable', err, flush=True)
            gains['g_lock'] *= 0.7
            gains['g_creep'] *= 1.5
            continue
        history.append(dict(round=round_id, gains=dict(gains), **m))
        print(args.shape, json.dumps(history[-1]), flush=True)
        ratio = lambda target, value: float(np.clip(target / max(value, 1e-6), 0.25, 4.0))
        gains['g_knee'] *= ratio(x.p_knee, m['p_knee'])
        gains['g_lock'] *= ratio(m['target2'], m['p2'])
        if m['creep'] is not None:
            gains['g_creep'] *= ratio(m['creep_target'], m['creep'])
        if m['hysteresis'] is not None:
            gains['g_hyst'] *= ratio(TARGET_HYSTERESIS, max(m['hysteresis'], 0.01))
    (out / f'gains_{args.shape}.json').write_text(json.dumps(gains, indent=2))
    (out / f'calibration_{args.shape}.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
