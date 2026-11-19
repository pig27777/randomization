"""Create a local MuJoCo Warp copy with consistent 2-D flex contact normals."""

import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
from shutil import copytree, ignore_patterns


ORIGINAL = "      best_normal = gjk_normal\n      if did >= 0:\n"
CORRECTED = ("      best_normal = gjk_normal\n"
             "      if did >= 0 and geom2_type != int(GeomType.TRIANGLE):\n")


def main():
    package_version = version("mujoco-warp")
    if package_version != "4.0.1":
        raise RuntimeError(f"This diagnostic targets MuJoCo Warp 4.0.1, found {package_version}")
    source = Path(find_spec("mujoco_warp").origin).parent
    relative = Path("_src/collision_flex.py")
    original = (source / relative).read_bytes()
    code = original.decode()
    if code.count(ORIGINAL) != 1:
        raise RuntimeError("Expected exactly one mesh-flex normal selection block")
    patched = code.replace(ORIGINAL, CORRECTED).encode()
    run = Path(__file__).resolve().parent
    overlay = run / "isolated-package" / "mujoco_warp"
    copytree(source, overlay, ignore=ignore_patterns("__pycache__"))
    (overlay / relative).write_bytes(patched)
    record = dict(version=package_version, module=str(relative),
                  source_sha256=hashlib.sha256(original).hexdigest(),
                  overlay_sha256=hashlib.sha256(patched).hexdigest())
    (run / "overlay-info.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
