"""Audit record helpers; all output paths are new run directories."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import importlib.util
import shutil
import subprocess
import sys
from datetime import datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))

# Audit the recorded implementation even while another chat edits the live model.
# Only this audit process's import table is redirected; repository files stay intact.
SNAPSHOT=Path(os.environ.get('PARCEL_AUDIT_MODEL_SNAPSHOT',
    ROOT/'test/runs/2026-11-14/bag/parcel-physics-audit-kernels-r2/source')).resolve()
if SNAPSHOT.exists():
    import contents
    spec=importlib.util.spec_from_file_location('contents.parcel',SNAPSHOT/'__init__.py',
        submodule_search_locations=[str(SNAPSHOT)])
    package=importlib.util.module_from_spec(spec)
    sys.modules['contents.parcel']=package
    spec.loader.exec_module(package)
    contents.parcel=package
else:
    raise FileNotFoundError(f'Frozen audit source is missing: {SNAPSHOT}')


def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.void):return {k:clean(x[k]) for k in x.dtype.names}
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def setup(output,mode):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    files=[p for p in SNAPSHOT.iterdir() if p.is_file()]
    hashes={('test/content_model/parcel/yam_adapter.py' if p.name=='yam_adapter.py'
             else 'contents/parcel/'+p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    snap=out/'source';snap.mkdir()
    for p in files:shutil.copyfile(p,snap/p.name)
    audit=out/'audit_source';audit.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,audit/p.name)
    write(out/'metadata.json',dict(time=datetime.now().astimezone().isoformat(),mode=mode,
        command=sys.argv,machine=platform.node(),seed=7,source_origin=str(SNAPSHOT),source_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        versions={k:importlib.metadata.version(k) for k in ('mujoco','mujoco-warp','warp-lang','numpy')},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()))
    return out
