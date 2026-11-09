"""Verified activation of the exploratory GPU backend for coupled bag models."""
import hashlib
import json
from pathlib import Path
import sys

from mj_loose import CONTENT_RELEASE_SHA256

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/soft_joint_20261108'
GPU_SOURCE=OUT/'gpu_source'
# Only GPU operators come from the exploratory snapshot. CPU material imports
# have already been activated by loose_protocols and remain frozen.
manifest=json.loads((GPU_SOURCE/'manifest.json').read_text())
for name,entry in manifest.items():
    if hashlib.sha256((GPU_SOURCE/name).read_bytes()).hexdigest()!=entry['sha256']:
        raise RuntimeError(f'GPU exploration source checksum mismatch: {name}')
sys.path.insert(1,str(GPU_SOURCE))
from mjwarp_backend import BatchStepper, write_equalities
from warp_v13 import failure_metrics
