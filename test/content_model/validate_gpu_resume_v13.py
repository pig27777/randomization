"""Formal v1.3 GPU acceptance continuation, using the unchanged reference paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CONTENT = Path(__file__).resolve().parents[2] / "contents/segment"
sys.path.insert(0, str(CONTENT))
from source_manifest import source_paths, source_name

import validate_warp_v13 as V


def fingerprints():
    files=source_paths(V.ROOT)
    return {source_name(p, V.ROOT):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def aggregate(directory):
    original=json.loads((V.BASE/'g2_formal/summary.json').read_text())
    first=next(r for r in original['rows'] if r['job']==dict(category='bendable',test='r1y'))
    rows=[dict(run='g2_formal',row=first)]
    for i in range(2,6):
        p=directory/f'r1y_{i}/summary.json';r=json.loads(p.read_text())
        if not r['formal'] or not r['sources_unchanged']:raise ValueError(f'unverified formal result: {p}')
        rows.append(dict(run=f'r1y_{i}',row=r['row']))
    cpu=np.array([r['row']['reference']['stiffness'] for r in rows])
    gpu=np.array([r['row']['candidate']['stiffness'] for r in rows])
    relative=gpu/cpu-1
    median=float(np.median(gpu));reference=float(np.median(cpu))
    median_error=abs(median/reference-1)
    stable=not any(r['row']['gpu_fail'] or r['row']['cpu_fail'] for r in rows)
    passed=stable and median_error<=.02 and bool(np.all(np.abs(relative)<=.05))
    result=dict(formal=True,passed=passed,rows=rows,cpu_values=cpu.tolist(),gpu_values=gpu.tolist(),
        relative_errors=relative.tolist(),median_gpu=median,median_cpu=reference,median_relative_error=median_error,
        stiffness_range=float(np.ptp(gpu)),relative_error_range=float(np.ptp(relative)),stable=stable,
        rule='5 formal runs; median response error <=2%, every run <=5%; exploratory runs excluded')
    V.save(directory/'r1y_five_formal.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return passed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['r1y','aggregate'])
    ap.add_argument('--directory',required=True,type=Path)
    a=ap.parse_args()
    if a.action=='aggregate':return 0 if aggregate(a.directory) else 2
    a.directory.mkdir(parents=True,exist_ok=False)
    before=fingerprints();V.metadata(a.directory,a)
    row=V.g2_response(dict(category='bendable',test='r1y'))
    used=set()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(V.ROOT):used.add(str(path.relative_to(V.ROOT)))
    unchanged=all(n in before and hashlib.sha256((V.ROOT/n).read_bytes()).hexdigest()==before[n] for n in used)
    V.save(a.directory/'summary.json',dict(formal=True,sources_unchanged=unchanged,
        checked_imported_sources=sorted(used),source_sha256=before,row=row))
    print(json.dumps(dict(cpu=row['reference']['stiffness'],gpu=row['candidate']['stiffness'],
        relative=row['candidate']['stiffness']/row['reference']['stiffness']-1,source_unchanged=unchanged)),flush=True)
    if not unchanged:raise RuntimeError('an existing source file changed during formal run')
    return 0 if not row['gpu_fail'] and not row['cpu_fail'] else 2


if __name__=='__main__':raise SystemExit(main())
