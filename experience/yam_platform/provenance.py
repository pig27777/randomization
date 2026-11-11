"""Source paths for recording the actual shared implementation in new runs."""
from pathlib import Path
import shutil


def source_files():
    package = Path(__file__).resolve().parent
    return {f"yam_{p.stem}": p for p in sorted(package.glob("*.py"))}


def snapshot(destination):
    """Keep archived policy drivers runnable with their own platform and YAM assets."""
    package = Path(__file__).resolve().parent
    shutil.copytree(package, Path(destination) / "experience/yam_platform",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
