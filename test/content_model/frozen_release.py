"""Verify immutable releases and their current model files after driver relocation."""
import hashlib
from pathlib import Path
import tarfile

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parents[1]


def verify(tag, model_root, project=PROJECT, tools=TOOLS):
    archive = model_root / tag["archive"]
    archive_ok = hashlib.sha256(archive.read_bytes()).hexdigest() == tag["archive_sha256"]
    archive_changed = []
    runtime_changed = []
    relocated = {}
    with tarfile.open(archive) as stream:
        for name, expected in tag["source_sha256"].items():
            source = stream.extractfile(name)
            if source is None or hashlib.file_digest(source, "sha256").hexdigest() != expected:
                archive_changed.append(name)
            current = project / name
            if current.is_file():
                if hashlib.sha256(current.read_bytes()).hexdigest() != expected:
                    runtime_changed.append(name)
                continue
            relative = Path(name)
            if relative.parts[:2] == ("contents", "segment"):
                driver = tools / Path(*relative.parts[2:])
                if driver.is_file():
                    relocated[name] = dict(path=str(driver.relative_to(project)),
                        sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
                    continue
            runtime_changed.append(name)
    return dict(intact=archive_ok and not archive_changed and not runtime_changed,
        archive_intact=archive_ok and not archive_changed,
        changed_sources=runtime_changed, changed_archive_sources=archive_changed,
        relocated_drivers=relocated)
