"""Prepare, run and summarize the preregistered paired probe-feedback experiment.

Run from the repository root; formal stages use the shared flock externally.
The frozen source archive and policy JSON hashes make concurrent work harmless.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'policy/results/yam_probe_feedback_20261108'
FORCES = (6., 10., 14., 20.)
ARMS = ('direct', 'probe_blind', 'probe_feedback')
DEV_SEED, TEST_SEED = 2026110801, 2026110802
BACKEND_FILES = ('model.py', 'mj_adapter.py', 'metrics.py', 'bushing.py', 'top_grasp.py', 'pinch.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(run):
    release = ROOT / 'contents/segment/results/pinch_connect_v1_3/release_v1_3'
    tag = json.loads((release / 'v1.3.json').read_text())
    archive = release / 'v1.3.tar.gz'
    if digest(archive) != tag['archive_sha256']:
        raise RuntimeError('frozen v1.3 archive SHA mismatch')
    source = run / 'source'
    source.mkdir(parents=True, exist_ok=True)
    table = 'contents/segment/' + tag['table']['path']
    wanted = ['contents/segment/' + name for name in BACKEND_FILES] + [table]
    hashes = {}
    with tarfile.open(archive) as tar:
        for name in wanted:
            data = tar.extractfile(name).read()
            expected = tag['table']['sha256'] if name == table else tag['source_sha256'][name]
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise RuntimeError(f'frozen source mismatch: {name}')
            dest = source / name
            if dest.exists() and dest.read_bytes() != data:
                raise RuntimeError(f'refusing to overwrite modified frozen input: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            hashes[name] = actual
    save(run / 'source_manifest.json', dict(archive_sha256=digest(archive), files=hashes))
    print(json.dumps({'source': str(source), 'verified_files': len(hashes)}, ensure_ascii=False))


def load_backend(run):
    manifest = json.loads((run / 'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(run / 'source' / name) != expected:
            raise RuntimeError(f'frozen input changed: {name}')
    sys.path.insert(0, str(run / 'source/contents/segment'))
    import model
    if not model.load_table():
        raise RuntimeError('calibration table missing')
    return model


def make_objects(run, split):
    model = load_backend(run)
    rng = np.random.default_rng(DEV_SEED if split == 'dev' else TEST_SEED)
    objects = []
    for group in range(4 if split == 'dev' else 8):
        size = tuple(rng.uniform([.18, .060, .050], [.24, .080, .070]))
        mass = float(rng.uniform(.20, .65))
        mu = float(rng.uniform(.4, .9))
        offset = tuple(rng.uniform([-.08, -.04, -.02], [.08, .04, .02]))
        for category, anchor in model.ANCHORS.items():
            k = replace(anchor, size=size, mass=mass, surface_friction=mu,
                        com_offset=offset, mobile=False, skin_compliance=0.)
            # Fail on uncovered constitutive targets instead of using extrapolation.
            desc = model.build(k)
            objects.append(dict(id=f'{split}_{group:02d}_{category}', group=group,
                                kappa=k.to_dict(), internal=desc.internal))
    return objects


def feature(geometry, feedback):
    # Geometry is deliberately held out of all three force selectors: C-B isolates feedback.
    return np.array([np.log1p(max(0., feedback[0])), feedback[1]])


def choose(policy, arm, geometry, feedback):
    if arm != 'probe_feedback':
        return policy[arm]['force_N']
    if feedback is None:
        raise ValueError('feedback policy requires sensor readings')
    model = policy['probe_feedback']
    x = feature(geometry, feedback)
    X = np.asarray(model['features'])
    distance = np.sum(((X - x) / np.asarray(model['scale'])) ** 2, axis=1)
    nearest = np.argsort(distance, kind='stable')[:model['k']]
    scores = np.asarray(model['rewards'])[nearest].mean(0)
    return FORCES[int(np.argmax(scores))]


def job_run(job):
    from probe_env import evaluate
    def selector(geometry, feedback):
        if job.get('force_N') is not None:
            return job['force_N']
        return choose(job['policy'], job['arm'], geometry, feedback)
    result = evaluate(job['object']['kappa'], probe=job['arm'] != 'direct',
                      selector=selector, seed=job['seed'], trace=job.get('trace', False))
    return dict(object_id=job['object']['id'], group=job['object']['group'],
                geometry=job['object']['kappa']['size'], arm=job['arm'], seed=job['seed'],
                requested_force_N=job.get('force_N'), **result)


def execute(run, stage, jobs, workers):
    path = run / f'{stage}.jsonl'
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    import mujoco
    from yam_probe_robot import platform_source_files, snapshot_platform
    platform_sources = list(platform_source_files().values())
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               [ROOT / 'policy/probe_env.py', ROOT / 'policy/probe_feedback.py',
                ROOT / 'policy/yam_probe_robot.py', ROOT / 'test/policy/PROBE_REPORT.md', *platform_sources]}
    metadata = dict(stage=stage, host=platform.node(), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__,
                    workers=workers, episodes=len(jobs), code_sha256=sources,
                    source_manifest_sha256=digest(run / 'source_manifest.json'),
                    policy_sha256=digest(run / 'policy.json') if (run / 'policy.json').exists() else None,
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    save(run / f'{stage}_metadata.json', metadata)
    from yam_probe_robot import asset_fingerprint
    save(run / 'yam_asset_sha256.json', asset_fingerprint())
    # Snapshot experiment source for this stage as other sessions share the checkout.
    snapshot = run / 'driver_source' / stage
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ('probe_env.py', 'probe_feedback.py', 'yam_probe_robot.py'):
        (snapshot / name).write_bytes((ROOT / 'policy' / name).read_bytes())
    snapshot_platform(snapshot)
    t0, counts, rows = time.perf_counter(), Counter(), []
    with path.open('x') as stream, ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('fork')) as pool:
        for row in pool.map(job_run, jobs, chunksize=1):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
            counts[row['outcome']] += 1
            if len(rows) % 6 == 0 or len(rows) == len(jobs):
                print(json.dumps(dict(stage=stage, done=len(rows), total=len(jobs),
                                      wall_s=round(time.perf_counter()-t0, 1), outcomes=counts)), flush=True)
    metadata.update(completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'), wall_s=time.perf_counter()-t0,
                    outcomes=dict(counts), output_sha256=digest(path))
    save(run / f'{stage}_metadata.json', metadata)
    return rows


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def fit(run):
    rows = read_rows(run / 'dev.jsonl')
    if len(rows) != 96:
        raise RuntimeError(f'expected all 96 development episodes, found {len(rows)}')
    policy = dict(dev_sha256=digest(run / 'dev.jsonl'), candidate_forces_N=list(FORCES))
    for arm in ('direct', 'probe_blind'):
        rates = [np.mean([r['outcome'] == 'success' for r in rows
                         if r['arm'] == arm and r['requested_force_N'] == F]) for F in FORCES]
        policy[arm] = dict(force_N=FORCES[int(np.argmax(rates))], development_success_rates=rates)
    X, Y, ids = [], [], []
    for obj in sorted({r['object_id'] for r in rows}):
        rs = [next(r for r in rows if r['object_id'] == obj and r['arm'] == 'probe_blind'
                   and r['requested_force_N'] == F) for F in FORCES]
        digests = {r['probe_digest'] for r in rs}
        if len(digests) != 1:
            raise RuntimeError('development probe prefix depends on future action')
        if rs[0]['feedback'] is None:
            # Pre-decision failure is counted in development rates, not given fabricated feedback.
            continue
        X.append(feature(rs[0]['geometry'], rs[0]['feedback']).tolist())
        Y.append([float(r['outcome'] == 'success') for r in rs])
        ids.append(obj)
    if len(X) < 5:
        raise RuntimeError('fewer than five valid development probes')
    scale = np.maximum(np.std(X, axis=0), [.1, .1])
    policy['probe_feedback'] = dict(k=5, features=X, rewards=Y, scale=scale.tolist(), development_ids=ids,
            input_names=['log1p_nonnegative_compression_mm','relaxed_force_N'])
    if (run / 'policy.json').exists():
        raise FileExistsError('policy already frozen')
    save(run / 'policy.json', policy)
    (run / 'policy.sha256').write_text(digest(run / 'policy.json') + '\n')
    print(json.dumps({k: v for k, v in policy.items() if k != 'probe_feedback'}, ensure_ascii=False))


def statistics(run):
    rows = read_rows(run / 'test.jsonl')
    if len(rows) != 144:
        raise RuntimeError(f'expected 144 test episodes, found {len(rows)}')
    out = dict(episodes=len(rows), physical_objects=24, independent_geometry_groups=8, arms={}, comparisons={})
    for arm in ARMS:
        rs = [r for r in rows if r['arm'] == arm]
        counts = Counter(r['outcome'] for r in rs)
        valid = len(rs) - counts['sim_fail']
        good = [r for r in rs if r['outcome'] == 'success']
        out['arms'][arm] = dict(n=len(rs), outcomes=dict(counts), valid=valid,
                success_lower=counts['success']/len(rs), success_valid=counts['success']/valid if valid else None,
                mean_sim_s=float(np.mean([r['sim_s'] for r in rs])),
                success_mean_sim_s=float(np.mean([r['sim_s'] for r in good])) if good else None,
                selected_forces=dict(Counter(str(r['selected_force_N']) for r in rs)),
                per_category={c: dict(Counter(r['outcome'] for r in rs if r['category']==c))
                              for c in ('hard', 'bendable', 'compressible')})
    lookup = {(r['object_id'], r['seed'], r['arm']): r for r in rows}
    mismatches = []
    for r in rows:
        if r['arm'] == 'probe_feedback':
            b = lookup[r['object_id'], r['seed'], 'probe_blind']
            if (r['probe_digest'], r['probe_raw'], r['feedback']) != (b['probe_digest'], b['probe_raw'], b['feedback']):
                mismatches.append([r['object_id'], r['seed']])
    out['probe_prefix_mismatches'] = mismatches
    for baseline in ('direct', 'probe_blind'):
        grouped, valid_grouped, excluded, wins, losses, squeeze_diff = {}, {}, 0, 0, 0, []
        for r in rows:
            if r['arm'] != 'probe_feedback':
                continue
            b = lookup[r['object_id'], r['seed'], baseline]
            delta = int(r['outcome']=='success') - int(b['outcome']=='success')
            grouped.setdefault(r['group'], []).append(delta)
            wins += delta == 1
            losses += delta == -1
            squeeze_diff.append(int(r['outcome']=='over_squeeze')-int(b['outcome']=='over_squeeze'))
            if 'sim_fail' in (r['outcome'], b['outcome']):
                excluded += 1
            else:
                valid_grouped.setdefault(r['group'], []).append(delta)
        group_means = np.array([np.mean(grouped[g]) for g in sorted(grouped)])
        rng = np.random.default_rng(2026110899)
        samples = rng.choice(group_means, (10000, len(group_means)), replace=True).mean(1)
        ci = np.percentile(samples, [2.5, 97.5]).tolist()
        valid_values = [v for values in valid_grouped.values() for v in values]
        out['comparisons']['probe_feedback_minus_' + baseline] = dict(
            delta=float(group_means.mean()), ci95_cluster=ci, wins=int(wins), losses=int(losses),
            valid_pairs=48-excluded, excluded_simfail_pairs=excluded,
            delta_valid_pairs=float(np.mean(valid_values)) if valid_values else None,
            oversqueeze_delta=float(np.mean(squeeze_diff)))
    primary = out['comparisons']['probe_feedback_minus_probe_blind']
    valid = not mismatches and all(a['outcomes'].get('sim_fail',0)/a['n'] <= .05 for a in out['arms'].values())
    out['validity_passed'] = valid
    out['supports_feedback_benefit'] = bool(valid and primary['delta'] >= .10 and
            primary['ci95_cluster'][0] > 0 and primary['oversqueeze_delta'] <= 0)
    out['policy_sha256'] = digest(run / 'policy.json')
    out['test_sha256'] = digest(run / 'test.jsonl')
    save(run / 'summary.json', out)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'smoke', 'dev', 'fit', 'test', 'summary'))
    ap.add_argument('--run', type=Path, default=DEFAULT_RUN)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    run = args.run.resolve()
    if args.stage == 'prepare':
        prepare(run)
        return
    load_backend(run)
    if args.stage == 'fit':
        fit(run)
        return
    if args.stage == 'summary':
        statistics(run)
        return
    if args.stage == 'test':
        if digest(run / 'policy.json') != (run / 'policy.sha256').read_text().strip():
            raise RuntimeError('frozen policy changed')
        policy = json.loads((run / 'policy.json').read_text())
        if policy['dev_sha256'] != digest(run / 'dev.jsonl'):
            raise RuntimeError('development evidence changed after fitting')
        objects = make_objects(run, 'test')
        save(run / 'test_objects.json', objects)
        jobs = [dict(object=o, arm=arm, seed=41000+2*o['group']+rep, policy=policy,
                     trace=(o['group']==0 and o['kappa']['category']=='bendable' and arm=='probe_feedback' and rep==0))
                for o in objects for rep in range(2) for arm in ARMS]
    else:
        objects = make_objects(run, 'dev')
        save(run / 'dev_objects.json', objects)
        if args.stage == 'smoke':
            jobs = [dict(object=o, arm=arm, seed=31000, force_N=10., trace=True)
                    for o in objects[:3] for arm in ('direct', 'probe_blind')]
        else:
            jobs = [dict(object=o, arm=arm, seed=31000+o['group'], force_N=F)
                    for o in objects for arm in ('direct', 'probe_blind') for F in FORCES]
    execute(run, args.stage, jobs, args.workers)


if __name__ == '__main__':
    main()
