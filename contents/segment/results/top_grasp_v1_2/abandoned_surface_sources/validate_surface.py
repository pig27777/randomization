"""T1 surface regression against the frozen v1 cuboid, with immutable run folders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/top_grasp_v1_2"


def save(path, value):
    from response_distribution import save as write
    write(path, value)


def measure(job):
    category, test, destination, reference = job
    import model
    import mj_adapter as adapter
    import protocols
    destination = Path(destination)
    destination.mkdir()
    os.chdir(destination)
    k = model.ANCHORS[category]
    if not reference:
        k = replace(k, surface=True)
    start = time.perf_counter()
    row = dict(category=category, test=test, kappa=k.to_dict())
    try:
        desc = model.build(k, omega_max=adapter.OMEGA_MAX)
        row.update(segment_count=len(desc.segments), coarse_joint_count=len(desc.joints),
                   surface_count=len(getattr(desc, 'surface_links', [])))
        row['result'] = (protocols.r1(k, test[-1], desc=desc) if test.startswith('r1') else
                         protocols.r2(k, 'edge', desc=desc) if test == 'r2' else protocols.r3(k, desc=desc))
        if row['result'].get('fail'):
            row['result']['failure_state'] = protocols.measurement_runner.failure_state
        row['execution_error'] = False
    except Exception:
        row.update(execution_error=True, traceback=traceback.format_exc())
    row['wall_s'] = time.perf_counter() - start
    save(destination / 'result.json', row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--categories', nargs='+', default=['hard', 'bendable', 'compressible'])
    ap.add_argument('--tests', nargs='+', default=['r1y', 'r1z', 'r2', 'r3'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out = (BASE / args.run_id).resolve()
    out.mkdir()
    runtime = (BASE / 'reference_v1/contents/segment').resolve() if args.reference else ROOT
    sys.path.insert(0, str(runtime))
    import model
    import mj_adapter
    import protocols
    from response_distribution import observe_protocols
    observe_protocols(protocols)
    files = sorted(runtime.glob('*.py')) + sorted((runtime / 'tests').glob('*.py'))
    save(out / 'metadata.json', dict(reference=args.reference, runtime=str(runtime),
        source_sha256={str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        solver=mj_adapter.SOLVER, timestep=mj_adapter.BUSHING_DT))
    if not args.reference:
        import unittest
        import numpy as np
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_surface.py')
        checks = unittest.TextTestRunner(verbosity=2).run(suite)
        save(out / 'geometry.json', dict(tests_run=checks.testsRun, errors=len(checks.errors), failures=len(checks.failures)))
        if not checks.wasSuccessful():
            raise RuntimeError('surface geometry/unit checks failed; no dynamic regression was run')
        # Anchor mass moments, separately readable without parsing test output.
        from test_surface import moments
        rows = []
        for cat in args.categories:
            k = model.ANCHORS[cat]
            baseline = model.build(k, omega_max=mj_adapter.OMEGA_MAX)
            fine = model.build(replace(k, surface=True), omega_max=mj_adapter.OMEGA_MAX)
            a, b = moments(baseline), moments(fine)
            rows.append(dict(category=cat, mass_error=abs(a[0] - b[0]), com_error=float(np.linalg.norm(a[1] - b[1])),
                             inertia_relative_error=float(np.linalg.norm(a[2] - b[2]) / np.linalg.norm(a[2])),
                             coarse_segments=len(baseline.segments), fine_segments=len(fine.segments)-len(baseline.segments)))
        save(out / 'mass_moments.json', rows)
    jobs = [(cat, test, str(out / f'{cat}_{test}'), args.reference) for cat in args.categories for test in args.tests]
    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(measure, jobs):
            rows.append(row)
            print(f"{row['category']} {row['test']}: error={row['execution_error']} result={row.get('result')} "
                  f"wall_s={row['wall_s']:.2f}", flush=True)
    save(out / 'summary.json', dict(rows=rows, execution_errors=sum(r['execution_error'] for r in rows)))


if __name__ == '__main__':
    main()
