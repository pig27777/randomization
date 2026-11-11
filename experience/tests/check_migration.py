"""Compare the shared robot against its pre-migration Git source on CPU.

Run from randomization with its existing MuJoCo environment. This is an
organization regression check, not a material or grasp-success acceptance test.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experience"), str(ROOT / "test/flex3d"), str(ROOT / "policy")]


def historical_module(name, path, revision, replacements=()):
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    for before, after in replacements:
        source = source.replace(before, after)
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def compare(before, after):
    assert before.sim.built.xml == after.sim.built.xml
    for name in ("body_mass", "body_inertia", "jnt_range", "actuator_gainprm", "actuator_biasprm",
                 "actuator_forcerange", "actuator_ctrlrange", "geom_size", "geom_friction"):
        np.testing.assert_array_equal(getattr(before.sim.m, name), getattr(after.sim.m, name))
    peak = 0.
    for step in range(25):
        target = before.initial_target + [0., 0., step * .00001]
        for robot in (before, after):
            robot.command(target, .075)
            robot.sim.step()
        for name in ("qpos", "qvel", "ctrl", "actuator_force"):
            a, b = getattr(before.sim.d, name), getattr(after.sim.d, name)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            peak = max(peak, float(np.max(np.abs(a - b))))
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="7cf1d67")
    args = parser.parse_args()
    from bag_sim.frozen_content import activate
    activate()
    from model import ANCHORS
    from yam_platform import rigid_control
    from yam_platform.robot import ASSET
    from yam_platform.provenance import snapshot
    import yam_probe_robot

    old_robot = historical_module("_migration_original_yam", "policy/yam_probe_robot.py", args.baseline)
    # Use identical vendored bytes on both sides; remove host-path differences.
    old_robot.ASSET = ASSET
    old_platform = historical_module("_migration_original_protocol", "test/flex3d/grasp_platform.py", args.baseline,
                                    [("from yam_probe_robot import", "from _migration_original_yam import")])
    old_rigid = historical_module("_migration_original_rigid", "test/flex3d/rigid_control.py", args.baseline,
                                 [("from grasp_platform import", "from _migration_original_protocol import")])
    errors = {}
    k = replace(ANCHORS["bendable"], size=(.20, .07, .05))
    for seed in (0, 1):
        errors[f"side_grasp_seed{seed}"] = compare(old_robot.Yam(k, seed), yam_probe_robot.Yam(k, seed))
    cfg = rigid_control.Config()
    errors["shared_rigid_fixture"] = compare(old_rigid.scene(cfg), rigid_control.scene(cfg))

    asset = ASSET.parent
    manifest = json.loads((asset / "MANIFEST.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((asset / name).read_bytes()).hexdigest() == digest, name

    # A fresh interpreter must import the generic platform without frozen models.
    subprocess.run([sys.executable, "-B", "-c",
        "import sys; from yam_platform import protocol; "
        "assert not {'model', 'pinch', 'mj_adapter', 'bag_sim.frozen_content'} & set(sys.modules)"],
        cwd=ROOT / "experience", check=True)
    # Archived policy drivers carry a runnable copy, including relative assets.
    with tempfile.TemporaryDirectory(prefix="yam-experience-snapshot-") as temp:
        path = Path(temp)
        snapshot(path)
        (path / "yam_probe_robot.py").write_bytes((ROOT / "policy/yam_probe_robot.py").read_bytes())
        subprocess.run([sys.executable, "-B", "-c",
            "from pathlib import Path; import yam_probe_robot as y; "
            "assert y.ASSET.is_relative_to(Path.cwd()); assert y.ASSET.is_dir()"], cwd=path, check=True)
    print(json.dumps(dict(baseline=args.baseline, steps_per_case=25, max_absolute_state_difference=errors,
                          asset_manifest_files=len(manifest), independent_import=True, archived_import=True)))


if __name__ == "__main__":
    main()
