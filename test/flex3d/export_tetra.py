"""Compatibility entrypoint; implementation lives in experience/yam_platform."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experience"))
if __name__ == "__main__":
    import runpy
    runpy.run_module("yam_platform.export_tetra", run_name="__main__")
else:
    from yam_platform import export_tetra as _implementation
    sys.modules[__name__] = _implementation
