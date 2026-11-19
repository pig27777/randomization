"""Screen existing top membrane geometry before the first contact."""

import json
from pathlib import Path

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
BASE = RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1") / "refined"
CENTER = np.array([-.25, .04])
ALONG = np.linspace(-.028, .028, 29)
ACROSS = np.array([-.008, 0., .008])
ANGLE = (-45., 0., 45., 90.)


def heights(xy, triangles):
    a, b, c = np.moveaxis(triangles, 1, 0)
    u, v = b - a, c - a
    area = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    nonvertical = abs(area) > 1e-16
    divisor = np.where(nonvertical, area, 1.)
    diff = xy[:, None] - a[:, :2]
    b_fraction = (diff[:, :, 0] * v[:, 1] - diff[:, :, 1] * v[:, 0]) / divisor
    c_fraction = (u[:, 0] * diff[:, :, 1] - u[:, 1] * diff[:, :, 0]) / divisor
    a_fraction = 1. - b_fraction - c_fraction
    inside = (a_fraction >= -1e-10) & (b_fraction >= -1e-10) & (c_fraction >= -1e-10) & nonvertical
    z = a[:, 2] + b_fraction * u[:, 2] + c_fraction * v[:, 2]
    return np.where(inside, z, -np.inf).max(axis=1)


def position(center, yaw, triangles):
    angle = np.deg2rad(yaw)
    closing = np.array([-np.sin(angle), np.cos(angle)])
    width = np.array([np.cos(angle), np.sin(angle)])
    xy = center + ACROSS[:, None, None] * width + ALONG[None, :, None] * closing
    z = heights(xy.reshape(-1, 2), triangles).reshape(len(ACROSS), len(ALONG))
    if not np.isfinite(z).all():
        return None
    left = z[:, ALONG <= -.020 + 1e-12].max(axis=1)
    right = z[:, ALONG >= .020 - 1e-12].max(axis=1)
    centers = z[:, len(ALONG) // 2]
    central = z[len(ACROSS) // 2]
    arclength = float(np.hypot(np.diff(ALONG), np.diff(central)).sum())
    chord = float(np.hypot(np.ptp(ALONG), central[-1] - central[0]))
    return dict(center_xy_m=center.tolist(), xy_offset_m=(center - CENTER).tolist(), yaw_deg=yaw,
                first_tip_height_m=float(max(left.max(), right.max())),
                pad_height_difference_m=float(abs(left.max() - right.max())),
                ridge_over_pads_m=float(np.min(centers - np.maximum(left, right))),
                centerline_excess_m=arclength - chord)


def main():
    a = np.array([[[0., 0., 0.], [1., 0., 1.], [0., 1., 0.]]])
    np.testing.assert_allclose(heights(np.array([[.3, .1], [.1, .2]]), a), [.3, .1], rtol=0., atol=1e-12)
    trial = BASE / "recording/trial"
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    states = json.loads((trial / "trace.json").read_bytes())
    final_settle = max(i for i, row in enumerate(states) if row["phase"] == "settle")
    with np.load(trial / "trajectory.npz") as npz:
        np.testing.assert_equal(npz["time"][final_settle], states[final_settle]["t"])
        data.qpos[:] = npz["qpos"][final_settle]
    mujoco.mj_fwdPosition(model, data)
    with np.load(BASE / "prepared_mesh.npz") as asset:
        faces = model.flex_elem.reshape(-1, 3)
        np.testing.assert_array_equal(faces, asset["faces"])
        triangles = data.flexvert_xpos[faces[asset["panel_ids"] == 0]]
    offsets = np.linspace(-.08, .08, 17)
    results = [row for yaw in ANGLE for x in offsets for y in offsets
               if (row := position(CENTER + [x, y], yaw, triangles)) is not None]
    balanced = [row for row in results if row["pad_height_difference_m"] <= .002]
    qualified = sorted((row for row in balanced if row["ridge_over_pads_m"] >= .0005),
                       key=lambda row: -row["ridge_over_pads_m"])
    summary = dict(frame=final_settle, time_s=states[final_settle]["t"], centers=17**2, angles=ANGLE,
                   poses_with_full_film=len(results), balanced_poses=len(balanced), qualified_poses=len(qualified),
                   original=position(CENTER, 0., triangles), strongest=qualified[:12],
                   best_balanced=max(balanced, key=lambda row: row["ridge_over_pads_m"]))
    with (RUN / "ridge-selection.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
