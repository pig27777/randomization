"""Set YAM's pre-contact braking height from the actual open-pad footprints."""

import argparse
from contextlib import nullcontext
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
from refine_parcel import backend, main as record  # noqa: E402
from rss2027_yam_capture_audit import clip_plane  # noqa: E402

PAD_HALF_FOOTPRINT_M = [.010, .004]
INTEGRATION_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def surface_below_pad(model, data, name, triangles, shift_xy=(0., 0.)):
    pad = model.geom(name)
    axes = data.xmat[int(pad.bodyid[0])].reshape(3, 3)
    if np.linalg.norm(axes[2, :2]) > .01:
        raise ValueError("The measured footprint assumes a vertical pinch frame")
    local = triangles.copy()
    local[:, :, :2] = (local[:, :, :2] - data.geom_xpos[pad.id, :2] - shift_xy) @ axes[:2, :2]
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


def at_top_center(center, top_reference_m=None):
    attach = backend.attach_robot

    def shifted(cfg, built, factory, _center, top, size, mass, approach=None):
        if approach is not None or cfg.grasp_face != "top":
            raise ValueError("A measured top center requires a top grasp")
        reference = top if top_reference_m is None else top_reference_m
        return attach(cfg, built, factory, np.asarray(center), reference, size, mass)

    return patch.object(backend, "attach_robot", shifted)


def prepare(case, use_current_case=False):
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
    proof = dict(source_time_s=result["sim_time_s"], global_braking_surface_m=original,
                 settled_pad_surfaces_m=pad_heights, local_braking_surface_m=local_height,
                 difference_m=original - local_height)
    if use_current_case:
        with (case / "press-braking-height.json").open("x") as stream:
            stream.write(json.dumps(proof, indent=2, allow_nan=False) + "\n")
        return
    target = case / "local-height"
    target.mkdir(exist_ok=False)
    (target / "height.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    inputs = json.loads((case / "smoke.json").read_bytes())
    for name, seconds in (("smoke", .02), ("settle", .30), ("config", 0.)):
        (target / f"{name}.json").write_text(json.dumps(inputs | {"smoke_seconds": seconds}, indent=2) + "\n")


def integration_state(sim):
    state = np.empty(mujoco.mj_stateSize(sim.m, INTEGRATION_SPEC))
    mujoco.mj_getState(sim.m, sim.d, state, INTEGRATION_SPEC)
    return state


def prepare_pressure_priority(case):
    source = case.resolve()
    target = source.with_name("ridge-pressure-priority")
    target.mkdir(exist_ok=False)
    height = json.loads((source / "height.json").read_bytes())
    height["inward_speed_limit_m_s"] = .008
    (target / "height.json").write_text(json.dumps(height, indent=2, allow_nan=False) + "\n")
    for filename in ("smoke.json", "settle.json", "config.json"):
        inputs = json.loads((source / filename).read_bytes())
        inputs.update(simulation_seconds=25., wall_limit=2600.)
        (target / filename).write_text(json.dumps(inputs, indent=2, allow_nan=False) + "\n")


def record_with_braking_height(height=None):
    controller = FasterApproach
    if height is not None:
        speed_limit = height.get("inward_speed_limit_m_s")
        if speed_limit is not None and (not np.isfinite(speed_limit) or speed_limit <= 0.):
            raise ValueError("The inward speed limit must be finite and positive")

        class LocalSurfacePress(FasterApproach):
            def __init__(self, target_N, surface_z):
                np.testing.assert_allclose(surface_z, height["global_braking_surface_m"], atol=.0001, rtol=0.)
                super().__init__(target_N, height["local_braking_surface_m"])

            def closure_rate(self, nominal_m_s, force_N, support_N):
                if speed_limit is None or nominal_m_s >= 0.:
                    return nominal_m_s
                lower, upper = .8 * self.target_N, 1.2 * self.target_N
                if (support_N > self.support_limit_N or not lower <= force_N <= upper
                        or not lower <= self.filtered_N <= upper):
                    return 0.
                return max(nominal_m_s, -speed_limit)

        controller = LocalSurfacePress
    center = height.get("target_center_xy_m") if height is not None else None
    placement = at_top_center(center) if center is not None else nullcontext()
    save_state = height is not None and height.get("record_integration_state", False)
    state_recording = (patch.multiple(backend, create=True, STATE_SPEC=INTEGRATION_SPEC,
                                      record_state=integration_state) if save_state else nullcontext())
    disabled = height.get("disabled_pad_geoms", ()) if height is not None else ()
    geometry = nullcontext()
    if disabled:
        from select_side_flap import local_pad_contacts
        geometry = local_pad_contacts({}, disabled)
    with placement, patch.object(protocol, "NormalPress", controller), state_recording, geometry:
        record()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--prepare"] and len(sys.argv) == 3:
        prepare(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["--prepare-current-case"] and len(sys.argv) == 3:
        prepare(Path(sys.argv[2]), use_current_case=True)
    elif sys.argv[1:2] == ["--prepare-pressure-priority"] and len(sys.argv) == 3:
        prepare_pressure_priority(Path(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args, _ = parser.parse_known_args()
        source = args.config.parent / "height.json"
        record_with_braking_height(json.loads(source.read_bytes()))
        (args.output / "press-braking-height.json").write_bytes(source.read_bytes())
