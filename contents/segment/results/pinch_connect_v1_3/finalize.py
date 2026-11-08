"""Audit the v1.3 formal runs against the current sources and tag v1.3 only if every gate passed."""
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
SEGMENT = BASE.parents[1]
PROJECT = SEGMENT.parents[1]
RUNS = ('r2_p1_units', 'r2_p2_r4', 'r2_p3_hang', 'r2_p4_e4', 'r2_p5_e5', 'r2_p6_e6', 'r2_p7_positions')
V12_TAG = SEGMENT / 'results/pinch_effect_v1_2/v22_review/release_v1_2/v1.2.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    stages = {}
    for run in RUNS:
        m = read(BASE / run / 'metadata.json')
        s = read(BASE / run / 'summary.json')
        stages[run] = dict(passed=s['passed'], execution_errors=s.get('execution_errors', 0),
                           source_matches=all(sha(SEGMENT / p) == h for p, h in m['source_sha256'].items()),
                           protocol_matches=sha(PROJECT / 'test/policy/README.md') == m['task_protocol_sha256'])
    trend = read(BASE / 'r2_p7_trend.json')
    gates = all(v['passed'] and not v['execution_errors'] and v['source_matches'] and v['protocol_matches']
                for v in stages.values()) and trend['trend_reasonable']
    files = [p for p in read(V12_TAG)['source_sha256']] + ['contents/segment/pinch_metrics.py']
    acceptance = dict(version='v1.3', rule='point connections; release when the 5 ms low-passed load exceeds '
                      'pinch_coeff x current grip force', stages=stages, p7_trend=trend, passed=gates)
    save(BASE / 'acceptance.json', acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=1))
    if not gates:
        return 2
    release = BASE / 'release_v1_3'
    release.mkdir()
    archive = release / 'v1.3.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for f in files:
            tar.add(PROJECT / f, arcname=f)
    tag = dict(tag='v1.3', frozen=True, frozen_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
               format='local immutable source archive and SHA256 tag; project has no Git metadata',
               protocol='v2.2', base='v1.2 with the top pinch as point connections (v1.3 plan and P2 review)',
               archive=str(archive.relative_to(SEGMENT)), archive_sha256=sha(archive),
               source_sha256={f: sha(PROJECT / f) for f in files},
               table=read(V12_TAG)['table'], acceptance=str((BASE / 'acceptance.json').relative_to(SEGMENT)),
               acceptance_sha256=sha(BASE / 'acceptance.json'), checks=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
               limits=['GPU/Warp files (warp_cpu, warp_effects, warp_mechanics, mjwarp_backend, validate_warp, '
                       'validate_acceleration) target the v1.2 weld and are not part of v1.3',
                       'P5 is a 200-object sample (1/200 pre-binding compression failure); seed 20261201 had 4/200 '
                       'failures that replay identically on v1.2',
                       'COM swing after the carry stops is larger than v1.2 by design (point connection)',
                       'off-centre pinches overload through the pinch-point / fingertip-edge couple',
                       'no rotational damping added; 40 N/mm spring and 10 mm gathering unchanged'])
    save(release / 'v1.3.json', tag)
    print('tagged', tag['archive_sha256'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
