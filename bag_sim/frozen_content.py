"""Read-only imports from the verified v1.3 content release, never the live content line."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / 'contents/segment/results/pinch_connect_v1_3/release_v1_3'
FROZEN = HERE / 'outputs/round3_flex/source_v1_3'


def activate():
    manifest = json.loads((RELEASE / 'v1.3.json').read_text())
    archive = RELEASE / 'v1.3.tar.gz'
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['archive_sha256']:
        raise RuntimeError('v1.3 archive checksum mismatch')
    if not (FROZEN / 'contents/segment/pinch.py').exists():
        FROZEN.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(FROZEN, filter='data')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen v1.3 source checksum mismatch: {name}')
    path = str(FROZEN / 'contents/segment')
    if path not in sys.path:
        sys.path.insert(0, path)
    return digest
