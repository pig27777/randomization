"""Compatibility entrypoint; implementation lives in experience/yam_platform."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experience"))
from bag_sim.frozen_content import activate
RELEASE_SHA = activate()
from yam_platform import protocol as _implementation
_implementation.RELEASE_SHA = RELEASE_SHA
sys.modules[__name__] = _implementation
