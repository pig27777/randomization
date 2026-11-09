"""Source inventory for current drivers and historical frozen runtimes."""
import os
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MODEL = TOOLS.parents[1] / "contents/segment"


def source_paths(runtime=MODEL):
    runtime = Path(runtime).resolve()
    paths = list(runtime.glob("*.py")) + list((runtime / "tests").glob("*.py"))
    if runtime == MODEL.resolve():
        paths += list(TOOLS.glob("*.py")) + list((TOOLS / "tests").glob("*.py"))
    return sorted(paths)


def source_name(path, runtime=MODEL):
    return os.path.relpath(path, runtime)
