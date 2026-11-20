"""Set YAM's pre-contact braking height from the actual open-pad footprints."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(RUN.with_name("rss2027-stage1-top-ridge-selection-r1")),
               str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1"))]
from record_faster_approach import FasterApproach, protocol  # noqa: E402
from refine_parcel import main as record  # noqa: E402
from rss2027_yam_capture_audit import clip_plane  # noqa: E402

PAD_HALF_FOOTPRINT_M = [.010, .004]


def surface_below_pad(model, data, name, triangles):
    pad = model.geom(name)
    axes = data.xmat[int(pad.bodyid[0])].reshape(3, 3)
    if np.linalg.norm(axes[2, :2]) > .01:
        raise ValueError("The measured footprint assumes a vertical pinch frame")
    local = triangles.copy()
    local[:, :, :2] = (local[:, :, :2] - data.geom_xpos[pad.id, :2]) @ axes[:2, :2]
    half = np.asarray(PAD_HALF_FOOTPRINT_M)
    candidates = (local[:, :, :2].min(axis=1) < half).all(axis=1)
    candidates &= (local[:, :, :2].max(axis=1) > -half).all(axis=1)
    z = []
    for polygon in local[candidates]:
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
            polygon = clip_plane(polygon, axis, sign, half[axis])
            if len(polygon) == 0:
                break
        if len(polygon) >= 3:
            z.extend(polygon[:, 2])
    if not z:
        raise ValueError(f"The projected footprint of {name} contains no membrane")
    return float(max(z) + model.flex_radius[0])


def prepare(case):
    trial = case / "settle/trial"
    result = json.loads((trial / "result.json").read_bytes())
    if (result["outcome"] != "smoke_only" or result["sim_time_s"] < .3 or result["warnings"]
            or result["peaks"]["penetration_mm"] > 1. or result["peaks"]["grip_N"] or result["peaks"]["down_N"]):
        raise ValueError("A local footprint requires a clean, complete settle trace")
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    with np.load(trial / "trajectory.npz") as archive:
        data.qpos[:] = archive["qpos"][0]
        mujoco.mj_fwdPosition(model, data)
        original = float(data.flexvert_xpos[:, 2].max() + model.flex_radius[0])
        data.qpos[:] = archive["final_qpos"]
        mujoco.mj_fwdPosition(model, data)
    faces = model.flex_elem.reshape(-1, 3)
    triangles = data.flexvert_xpos[faces]
    pad_heights = [surface_below_pad(model, data, name, triangles) for name in ("left_pad", "right_pad")]
    local_height = max(pad_heights)
    target = case / "local-height"
    target.mkdir(exist_ok=False)
    proof = dict(source_time_s=result["sim_time_s"], global_braking_surface_m=original,
                 settled_pad_surfaces_m=pad_heights, local_braking_surface_m=local_height,
                 difference_m=original - local_height)
    (target / "height.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    inputs = json.loads((case / "smoke.json").read_bytes())
    for name, seconds in (("smoke", .02), ("settle", .30), ("config", 0.)):
        (target / f"{name}.json").write_text(json.dumps(inputs | {"smoke_seconds": seconds}, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prepare"] and len(sys.argv) == 3:
        prepare(Path(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, required=True)
        args, _ = parser.parse_known_args()
        height = json.loads((args.config.parent / "height.json").read_bytes())

        class LocalSurfacePress(FasterApproach):
            def __init__(self, target_N, surface_z):
                np.testing.assert_allclose(surface_z, height["global_braking_surface_m"], atol=.0001, rtol=0.)
                super().__init__(target_N, height["local_braking_surface_m"])

        with patch.object(protocol, "NormalPress", LocalSurfacePress):
            record()
