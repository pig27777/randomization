"""Audit record helpers; all output paths are new run directories."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))


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
    files=list((ROOT/'contents/parcel').glob('*'))+[ROOT/'test/content_model/parcel/yam_adapter.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    snap=out/'source';snap.mkdir()
    for p in files:
        if p.is_file():shutil.copyfile(p,snap/p.name)
    audit=out/'audit_source';audit.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,audit/p.name)
    write(out/'metadata.json',dict(time=datetime.now().astimezone().isoformat(),mode=mode,
        command=sys.argv,machine=platform.node(),seed=7,source_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        versions={k:importlib.metadata.version(k) for k in ('mujoco','mujoco-warp','warp-lang','numpy')},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()))
    return out
