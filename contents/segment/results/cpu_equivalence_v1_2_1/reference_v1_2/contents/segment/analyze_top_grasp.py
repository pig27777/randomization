"""Summarize the recorded T1 regression and frozen-v1 paper distribution; no simulation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from response_distribution import save

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/top_grasp_v1_2'


def regression():
    baseline = json.loads((BASE / 't1_reference_formal/summary.json').read_text())
    candidate = json.loads((BASE / 't1_candidate_formal/summary.json').read_text())
    old = {(r['category'], r['test']): r for r in baseline['rows']}
    comparisons = []
    for row in candidate['rows']:
        key = row['category'], row['test']
        ref = old[key]
        field = 'stiffness' if key[1].startswith('r1') else 'droop' if key[1] == 'r2' else 'slip_mm'
        a, b = ref.get('result', {}).get(field), row.get('result', {}).get(field)
        failed = bool(ref['execution_error'] or row['execution_error']
                      or ref.get('result', {}).get('fail') or row.get('result', {}).get('fail'))
        delta = (b - a) / abs(a) if a is not None and b is not None and a != 0 else None
        passed = not failed and delta is not None and abs(delta) <= .10
        comparisons.append(dict(category=key[0], test=key[1], field=field, v1=a, candidate=b,
                                relative_change=delta, passed=passed, confirmed_failure=row.get('result', {}).get('fail'),
                                lower_bound=row.get('result', {}).get('lower_bound', False)))
    tag = json.loads((ROOT / 'results/freeze_v1_newton/release_v1/v1.json').read_text())
    archive = ROOT / 'results/freeze_v1_newton/release_v1/v1.tar.gz'
    preserved = hashlib.sha256(archive.read_bytes()).hexdigest() == tag['archive_sha256']
    assert preserved, 'frozen v1 archive changed'
    geom = json.loads((BASE / 't1_candidate_formal/geometry.json').read_text())
    pass_t1 = all(r['passed'] for r in comparisons) and not geom['errors'] and not geom['failures']
    result = dict(frozen=False, tag=None, v1_archive_unchanged=preserved, T1=dict(passed=pass_t1,
                  passed_records=sum(r['passed'] for r in comparisons), total=len(comparisons), rows=comparisons),
                  T2='not run', T3='not run', T4='not run', T5='not run',
                  T6='hard 80 mm range covered by formal surface unit check; no full v1.2 release',
                  remaining_gate='T1' if not pass_t1 else 'T2',
                  mass_moments=json.loads((BASE / 't1_candidate_formal/mass_moments.json').read_text()))
    save(BASE / 'acceptance.json', result)
    for row in comparisons:
        print(row)


def distribution():
    source = BASE / 'paper_distribution_formal/summary.json'
    data = json.loads(source.read_text())
    out = BASE / 'paper_analysis'
    out.mkdir()
    columns = ['sampler', 'category', 'index', 'squeeze_y_N_per_mm', 'droop', 'valid_r1y', 'valid_r2',
               'valid_joint', 'r1y_lower_bound', 'confirmed_failure', 'r1y_failure', 'r2_failure',
               'execution_error', 'r2_dropped', 'r2_table_touch', 'r2_pad_fz_ratio']
    with (out / 'responses.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in data['rows']:
            a, b = row.get('r1y', {}), row.get('r2', {})
            writer.writerow(dict(sampler=row['sampler'], category=row['category'], index=row['index'],
                squeeze_y_N_per_mm=a.get('stiffness'), droop=b.get('droop'), valid_r1y=row.get('valid_r1y', False),
                valid_r2=row.get('valid_r2', False), valid_joint=row.get('valid_joint', False),
                r1y_lower_bound=a.get('lower_bound', False), confirmed_failure=row.get('confirmed_failure', False),
                r1y_failure=a.get('fail'), r2_failure=b.get('fail'), execution_error=bool(row['errors']),
                r2_dropped=b.get('dropped'), r2_table_touch=b.get('table_touch'), r2_pad_fz_ratio=b.get('pad_fz_ratio')))
    save(out / 'distribution.json', dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                      groups=data['groups'], paired_groups=data['paired_groups']))
    for name, group in data['groups'].items():
        print(name, group)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('regression', 'distribution'))
    action = ap.parse_args().action
    regression() if action == 'regression' else distribution()
