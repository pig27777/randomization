"""E2-E4 of test/proxy_eval/REPORT.md: 50 Latin-hypercube grasp-parameter sets per object, one evaluate each.

  python -B run_e2.py run X1 X2 ... [--limit n]   append to results/e2_<object>.jsonl, skipping samples already there
  python -B run_e2.py summary         results/e2_summary.json and a Markdown table on stdout
The same 50 parameter sets (fixed LHS seed) are used for every object; sample k uses evaluate seed k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import task

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
N, LHS_SEED = 50, 20261104
OUTCOMES = ['success', 'drop', 'grasp_fail', 'over_squeeze', 'sim_fail']


def lhs(n=N, seed=LHS_SEED):
    rng = np.random.default_rng(seed)
    names = list(task.PARAM_RANGES)
    u = (np.stack([rng.permutation(n) for _ in names], 1) + rng.random((n, len(names)))) / n
    return [{k: float(lo + x * (hi - lo)) for k, (lo, hi), x in zip(names, task.PARAM_RANGES.values(), row)} for row in u]


def run(objects, limit=N):
    import warp as wp
    from evaluate import evaluate
    wp.config.quiet = True
    RESULTS.mkdir(exist_ok=True)
    samples = lhs()
    for name in objects:
        path = RESULTS / f'e2_{name}.jsonl'
        done = {json.loads(line)['sample'] for line in path.read_text().splitlines()} if path.exists() else set()
        for k, params in enumerate(samples[:limit]):
            if k in done:
                continue
            r = evaluate(name, params, seed=k)
            with path.open('a') as f:
                f.write(json.dumps(dict(sample=k, **r)) + '\n')
            print(name, k, r['outcome'], r['wall_s'], r['detail'], flush=True)


def summary():
    rows, out = [], {}
    for path in sorted(RESULTS.glob('e2_X*.jsonl'), key=lambda p: int(p.stem[4:])):
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        name = path.stem[3:]
        counts = {o: sum(r['outcome'] == o for r in recs) for o in OUTCOMES}
        wall = [r['wall_s'] for r in recs]
        out[name] = dict(n=len(recs), counts=counts, wall_median_s=float(np.median(wall)), wall_max_s=float(np.max(wall)),
                         sim_fail=[dict(sample=r['sample'], params=r['params'], detail=r['detail'])
                                   for r in recs if r['outcome'] == 'sim_fail'])
        rows.append(f"| {name} | {len(recs)} | " + ' | '.join(str(counts[o]) for o in OUTCOMES)
                    + f" | {np.median(wall):.1f} | {np.max(wall):.1f} |")
    all_recs = [json.loads(line) for p in RESULTS.glob('e2_X*.jsonl') for line in p.read_text().splitlines()]
    if all_recs:
        out['all'] = dict(n=len(all_recs), sim_fail_pct=100 * sum(r['outcome'] == 'sim_fail' for r in all_recs) / len(all_recs),
                          wall_median_s=float(np.median([r['wall_s'] for r in all_recs])))
    (RESULTS / 'e2_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print('| 物体 | 次数 | 成功 | 掉落 | 夹取失败 | 过度挤压 | sim_fail | 中位耗时 s | 最长 s |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    print('\n'.join(rows))
    if all_recs:
        print(json.dumps(out['all']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('objects', nargs='+')
    r.add_argument('--limit', type=int, default=N, help='only the first LIMIT samples (bag objects, see REPORT.md)')
    sub.add_parser('summary')
    args = ap.parse_args()
    run(args.objects, args.limit) if args.cmd == 'run' else summary()


if __name__ == '__main__':
    main()
