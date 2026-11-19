"""Rebuild native fingertip contact geometry at the same archived film shape."""

import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def interpolate(reference, faces, current, points):
    x, y = (np.unique(reference[:, axis]) for axis in (0, 1))
    columns = np.clip(np.searchsorted(x, points[:, 0]) - 1, 0, len(x) - 2)
    rows = np.clip(np.searchsorted(y, points[:, 1]) - 1, 0, len(y) - 2)
    candidates = 2 * (rows * (len(x) - 1) + columns)
    cells = faces[candidates[:, None] + [0, 1]]
    triangles = reference[cells, :2]
    first = triangles[:, :, 0]
    basis = (triangles[:, :, 1:] - first[:, :, None]).swapaxes(-1, -2)
    uv = np.linalg.solve(basis, (points[:, None, :2] - first)[..., None])[..., 0]
    weights = np.concatenate((1. - uv.sum(axis=-1, keepdims=True), uv), axis=-1)
    inside = (weights >= -1e-9).all(axis=-1)
    if not inside.any(axis=1).all():
        raise ValueError("Refined material coordinates fall outside the source triangles")
    index = np.arange(len(points))
    side = inside.argmax(axis=1)
    return np.einsum("ij,ijk->ik", weights[index, side], current[cells[index, side]])


def collision(model, reference, current, sample):
    data = mujoco.MjData(model)
    film = model.jnt_qposadr[model.body_jntadr[model.flex_vertbodyid, None] + np.arange(3)]
    bodies = np.array([model.body(name).id for name in ("left", "right")])
    pads = model.jnt_qposadr[model.body_jntadr[bodies, None] + np.arange(3)]
    pad_lookup = {int(geom): pad for pad, body in enumerate(bodies)
                  for geom in np.flatnonzero(model.geom_bodyid == body)}
    data.qpos[film] = current - reference
    half = .0115 - float(sample["gather_per_pad_m"])
    height = float(sample["pad_z_m"])
    centers = np.array([[-half, 0., height], [half, 0., height]])
    data.qpos[pads] = centers - model.body_pos[bodies]
    mujoco.mj_fwdPosition(model, data)
    assert np.allclose(data.flexvert_xpos, current, atol=1e-12, rtol=0.)
    inner = [[], []]
    for contact in data.contact:
        if 0 not in contact.flex or contact.dist >= 0.:
            continue
        for slot, geom in enumerate(contact.geom):
            pad = pad_lookup.get(int(geom))
            if pad is not None and (2 * pad - 1) * (2 * slot - 1) * contact.frame[0] > .7:
                inner[pad].append((float(-contact.dist), abs(float(contact.frame[2])), contact.pos.copy()))
    stats = []
    for candidates in inner:
        positions = np.array([p for _, _, p in candidates]).reshape(-1, 3)
        stats.append(dict(count=len(candidates), depth_max_m=max((d for d, _, _ in candidates), default=0.),
                          normal_z_abs_median=float(np.median([nz for _, nz, _ in candidates])) if candidates else None,
                          height_range_m=[float(positions[:, 2].min()), float(positions[:, 2].max())]
                          if candidates else None))
    return stats


def load_archived(case):
    model = mujoco.MjModel.from_binary_path(str(case / "model.mjb"))
    with (case / "curve.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(case / "coupon_nodes.npz") as nodes:
        return model, nodes["reference_m"], nodes["faces"], nodes["current_m"], rows


def main():
    run = Path(__file__).parent
    coarse = run.parent / "rss2027-stage1-pe-offcenter-convergence-r1/fine-extended"
    refined = run / "refined-aperture"
    a, ref_a, face_a, positions, rows = load_archived(coarse)
    b, ref_b, face_b, _, _ = load_archived(refined)
    shared = interpolate(ref_a, face_a, ref_a, ref_b)
    assert np.allclose(shared, ref_b, atol=1e-12, rtol=0.)
    refined_lookup = {tuple(point): index for index, point in enumerate(ref_b)}
    shared_indices = [refined_lookup[tuple(point)] for point in ref_a]
    centroids = ref_b[face_b].mean(axis=1)
    near_pad = (abs(centroids[:, 0]) < .016) & (abs(centroids[:, 1]) < .010)
    transferred_centroids = interpolate(ref_a, face_a, ref_a, centroids)
    assert np.allclose(transferred_centroids, centroids, atol=1e-12, rtol=0.)
    pull_indices = [i for i, row in enumerate(rows) if row["phase"] == "pull"]
    t0 = float(rows[pull_indices[0]]["t_s"])
    output = []
    for elapsed in (.120, .388):
        index = min(pull_indices, key=lambda i: abs(float(rows[i]["t_s"]) - t0 - elapsed))
        row, frame = rows[index], positions[index]
        upsampled = interpolate(ref_a, face_a, frame, ref_b)
        assert np.allclose(upsampled[shared_indices], frame, atol=1e-12, rtol=0.)
        reference_shape = interpolate(ref_a, face_a, frame, centroids)
        error = np.linalg.norm(reference_shape - upsampled[face_b].mean(axis=1), axis=1)
        output.append(dict(pull_elapsed_s=float(row["t_s"]) - t0,
                           aperture_centroid_error_max_m=float(error[near_pad].max()),
                           aperture_centroid_error_median_m=float(np.median(error[near_pad])),
                           grid_1_25_mm=collision(a, ref_a, frame, row),
                           grid_0_625_mm=collision(b, ref_b, upsampled, row)))
    print(json.dumps(dict(scope="Contact geometry on archived 1.25 mm poses", samples=output), indent=2))


if __name__ == "__main__":
    main()
