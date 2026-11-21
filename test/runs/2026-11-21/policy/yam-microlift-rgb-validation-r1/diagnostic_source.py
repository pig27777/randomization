import sys, json
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import run_yam_microlift as run

def probe(job):
    kind, mass, dest = job
    if kind == 'taut':
        run.TYPES['taut'] = ('rigid_box', 1., 0.)
    original = run.audit_yam.Config
    run.audit_yam.Config = lambda **kw: original(**kw, complete_motion=True)
    try:
        return run.trial((kind,mass,'continue',dest,False,True,True))
    finally:
        run.audit_yam.Config = original

if __name__ == '__main__':
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    jobs=[(kind,mass,out/f'{kind}-{mass:.2f}') for mass in (.18,.52) for kind in ('hard','soft','loose','taut')]
    with ProcessPoolExecutor(max_workers=8) as p: rows=list(p.map(probe,jobs))
    (out/'results.json').write_text(json.dumps(rows,indent=2))
