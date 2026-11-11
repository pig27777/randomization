"""Compatibility entrypoint; implementation lives in experience/yam_platform."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experience"))
from yam_platform import render_comparison as _implementation
sys.modules[__name__] = _implementation

if __name__ == "__main__":
    _implementation.main()
