"""Read-only v1.3 analysis of finished runs: P4 summary, w = 60 mm, swing vs v1.2, P7 trends.

Usage (from contents/segment): python results/pinch_connect_v1_3/analyze.py p4 | p7 RUN_ID
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def rows(run):
    return json.loads((BASE / run / 'summary.json').read_text())['rows']


def key(r):
    g = r['grasp']
    return r['category'], g['w'], g['d'], g['F'], r['seed']


def stats(values):
    v = np.array([x for x in values if x is not None])
    return dict(n=int(len(v)), median=float(np.median(v)) if len(v) else None,
                p90=float(np.percentile(v, 90)) if len(v) else None, max=float(v.max()) if len(v) else None)


def p4(run='r2_p4_e4', v12='p4_e4_v12_default'):
    summary = json.loads((BASE / run / 'summary.json').read_text())
    new = [r for r in summary['rows'] if r['category'] != 'hard' and r['coefficient_scale'] == 1.]
    old = {key(r): r for r in rows(v12)}
    out = dict(monotonicity=summary.get('monotonicity'), groups=summary['groups'],
               static_following=summary.get('static_following'), execution_errors=summary['execution_errors'])
    for c in ('bendable', 'compressible'):
        mine = [r for r in new if r['category'] == c]
        paired = [(r['metrics']['stop_swing_mm'], old[key(r)]['metrics']['stop_swing_mm']) for r in mine
                  if r['metrics']['stop_swing_mm'] is not None and old[key(r)]['metrics']['stop_swing_mm'] is not None]
        w60 = lambda rs: f"{sum(r['result']['outcome'] == 'success' for r in rs)}/{len(rs)}"
        out[c] = dict(
            swing_v13=stats(r['metrics']['stop_swing_mm'] for r in mine),
            swing_v12=stats(old[key(r)]['metrics']['stop_swing_mm'] for r in mine),
            swing_paired=dict(n=len(paired), median_ratio=float(np.median([a / b for a, b in paired if b > 0]))
                              if paired else None, v13_larger=sum(a > b for a, b in paired)),
            lift_end_tilt_v13=stats((r['metrics']['lift_end'] or {}).get('tilt_deg') for r in mine),
            lift_end_tilt_v12=stats((old[key(r)]['metrics']['lift_end'] or {}).get('tilt_deg') for r in mine),
            default_outcomes_v13=dict(Counter(r['result']['outcome'] for r in mine)),
            default_outcomes_v12=dict(Counter(old[key(r)]['result']['outcome'] for r in mine)),
            w60_success_v13=w60([r for r in mine if r['grasp']['w'] == .06]),
            w60_success_v12=w60([old[key(r)] for r in mine if r['grasp']['w'] == .06]),
            w60_by_F_v13={str(F): w60([r for r in mine if r['grasp']['w'] == .06 and r['grasp']['F'] == F])
                          for F in sorted({r['grasp']['F'] for r in mine})})
    full = [r for r in summary['rows'] if r['category'] != 'hard' and r['coefficient_scale'] == 1.]
    region = {}
    for r in full:
        k = (r['category'], r['grasp']['w'], r['grasp']['d'], r['grasp']['F'])
        region[k] = region.get(k, 0) + (r['result']['outcome'] == 'success')
    out['all_seed_success_combos'] = sorted([dict(category=k[0], w=k[1], d=k[2], F=k[3]) for k, n in region.items()
                                             if n == 3], key=lambda x: (x['category'], x['w'], x['d'], x['F']))
    return out


def p7(run):
    out = {}
    for r in rows(run):
        g = r['grasp']
        k = (r['category'], g['u'], g['v'])
        m = r['metrics']
        e = out.setdefault(k, dict(outcomes=Counter(), tilt=[], lowest=[], swing=[], overload=0, bound=0))
        e['outcomes'][r['result']['outcome']] += 1
        e['bound'] += any(x['event'] == 'bind' for x in r['result']['pinch_events'])
        e['overload'] += any(x['event'] == 'overload' for x in r['result']['pinch_events'])
        if m['lift_end']:
            e['tilt'].append(m['lift_end']['tilt_deg'])
            e['lowest'].append(m['lift_end']['lowest_point_mm'])
        if m['stop_swing_mm'] is not None:
            e['swing'].append(m['stop_swing_mm'])
    return [dict(category=c, u=u, v=v, outcomes=dict(e['outcomes']), bound=e['bound'], overload=e['overload'],
                 tilt_deg=stats(e['tilt']), lowest_point_mm=stats(e['lowest']), stop_swing_mm=stats(e['swing']))
            for (c, u, v), e in sorted(out.items())]


if __name__ == '__main__':
    data = p4() if sys.argv[1] == 'p4' else p7(sys.argv[2])
    print(json.dumps(data, indent=1, ensure_ascii=False, default=str))
