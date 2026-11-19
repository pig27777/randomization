"""Refine the existing PE asset near the measured top closure path."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
ORIGINAL = RUN.with_name("rss2027-stage1-randomized-top-relaxed-r1") / "sample-004"
sys.path.insert(0, str(ROOT / "test/bag_model"))
import physical_parcel_yam as backend  # noqa: E402
import record_physical_parcel as recorder  # noqa: E402
from bag_sim import physical_parcel  # noqa: E402

LEVELS = 3
HALF_PATCH_M = np.array([.016, .034])


def split_face(face, midpoint):
    marked = [tuple(sorted(edge)) in midpoint for edge in zip(face, np.roll(face, -1))]
    count = sum(marked)
    if count == 0:
        return [face]
    if count == 3:
        a, b, c = face
        ab, bc, ca = (midpoint[tuple(sorted(edge))] for edge in ((a, b), (b, c), (c, a)))
        return [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    start = next(i for i in range(3) if marked[i] and not marked[(i - 1) % 3])
    a, b, c = np.roll(face, -start)
    ab = midpoint[tuple(sorted((a, b)))]
    if count == 1:
        return [(a, ab, c), (ab, b, c)]
    bc = midpoint[tuple(sorted((b, c)))]
    return [(ab, b, bc), (a, ab, c), (ab, bc, c)]


def refine(raw, reference, current, faces, panels, center):
    for _ in range(LEVELS):
        triangles = current[faces, :2] - center
        selected = (panels == 0) & (triangles.max(axis=1) >= -HALF_PATCH_M).all(axis=1)
        selected &= (triangles.min(axis=1) <= HALF_PATCH_M).all(axis=1)
        edges = {tuple(sorted(edge)) for a, b, c in faces[selected] for edge in ((a, b), (b, c), (c, a))}
        edge_list = np.asarray(sorted(edges), dtype=int)
        midpoint = {tuple(pair): i for i, pair in enumerate(edge_list, len(raw))}
        raw = np.vstack((raw, raw[edge_list].mean(axis=1)))
        reference = np.vstack((reference, reference[edge_list].mean(axis=1)))
        current = np.vstack((current, current[edge_list].mean(axis=1)))
        new_faces, new_panels = [], []
        for face, panel in zip(faces, panels, strict=True):
            parts = split_face(face, midpoint)
            new_faces.extend(parts)
            new_panels.extend([panel] * len(parts))
        faces, panels = np.asarray(new_faces), np.asarray(new_panels)
    return raw, reference, current, faces, panels


def areas(points, faces):
    tri = points[faces]
    return .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def test_topology(reference, faces):
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    if np.any(counts != 2) or len(directed) != len(np.unique(directed, axis=0)) or np.any(areas(reference, faces) <= 0.):
        raise ValueError("Refinement must preserve a closed oriented triangular surface")


def prepared_wrap(asset):
    archive = physical_parcel.HERE / asset
    with np.load(archive) as data:
        saved = {key: data[key] for key in ("reference_m", "current_m", "payload_origin_m", "payload_size_m",
                                          "payload_offset_xy", "faces", "panel_ids")}

    def wrap(rest, start, faces, panel_ids, front_ids, back_ids, item, radius, *, box_face_clearance,
             content_offset_xy):
        np.testing.assert_allclose(rest, saved["reference_m"], atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(faces, saved["faces"])
        np.testing.assert_array_equal(panel_ids, saved["panel_ids"])
        np.testing.assert_allclose(item["half_size"] * 2, saved["payload_size_m"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(content_offset_xy, saved["payload_offset_xy"], atol=1e-12, rtol=0.)
        current = saved["current_m"].copy()
        item["position"] = saved["payload_origin_m"].copy()
        edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
        a, b = edges.T
        stretch = np.linalg.norm(current[b] - current[a], axis=1) / np.linalg.norm(rest[b] - rest[a], axis=1)
        metadata = dict(maximum_edge_strain=float(np.max(abs(stretch - 1.))),
                        method="linear subdivision of the original enclosed loaded surface")
        return current, metadata

    return wrap


def prepare():
    inputs = json.loads((ORIGINAL / "config.json").read_bytes())
    cfg = backend.Config(**inputs)
    parcel, _ = backend.parcel_world(cfg)
    backend.initialize_state(parcel, parcel.m, parcel.d, cfg)
    model, data = parcel.m, parcel.d
    if np.any(data.qvel):
        raise ValueError("The original fitted bag preparation must be at rest")
    center = np.array([*parcel.cfg.center, 0.])
    source = physical_parcel.HERE / cfg.geometry
    with np.load(source) as asset:
        raw, faces, panels = [asset[key] for key in ("vertices", "faces", "panel_ids")]
    reference = parcel.envelope["reference"] - center
    current = data.flexvert_xpos - center
    original_area = [areas(points, faces).sum() for points in (reference, current)]
    original_faces = len(faces)
    coarse = (raw, reference, current, faces, panels)
    refined = refine(*coarse, np.zeros(2))
    _, reference_fine, current_fine, faces_fine, panels_fine = refined
    test_topology(reference_fine, faces_fine)
    np.testing.assert_allclose([areas(points, faces_fine).sum() for points in (reference_fine, current_fine)],
                               original_area, atol=1e-12, rtol=0.)
    payload_origin = data.xpos[model.body("payload_0").id] - center
    asset_paths = {}
    for label, (vertices, rest, loaded, elements, element_panels) in (("coarse", coarse), ("refined", refined)):
        case = RUN / label
        case.mkdir(exist_ok=False)
        geometry = case / "prepared_mesh.npz"
        np.savez_compressed(geometry, vertices=vertices, faces=elements, panel_ids=element_panels, reference_m=rest,
                            current_m=loaded, payload_origin_m=payload_origin, payload_size_m=cfg.payload_size,
                            payload_offset_xy=cfg.payload_offset_xy)
        asset_paths[label] = f"../{geometry.relative_to(ROOT).as_posix()}"
    fine_cfg = replace(cfg, geometry=asset_paths["refined"], initial_state=None)
    wrap = prepared_wrap(asset_paths["refined"])
    upper_bound = None
    with patch.object(physical_parcel, "wrap_generic", wrap):
        try:
            fine, _ = backend.parcel_world(fine_cfg)
        except ValueError as exc:
            match = re.fullmatch(r"timestep does not resolve explicit film bending; use <= (\S+) s", str(exc))
            if match is None:
                raise
            upper_bound = float(match[1])
            timestep = fine_cfg.dt
            while timestep > upper_bound:
                timestep *= .5
            fine_cfg = replace(fine_cfg, dt=timestep)
            fine, _ = backend.parcel_world(fine_cfg)
    np.testing.assert_allclose(fine.envelope["masses"].sum(), parcel.envelope["masses"].sum(), atol=1e-12, rtol=0.)
    for label in ("coarse", "refined"):
        case = RUN / label
        config = inputs | {"geometry": asset_paths[label], "dt": fine_cfg.dt, "initial_state": None}
        for name, settings in (("config.json", config), ("smoke.json", config | {"smoke_seconds": .02})):
            (case / name).write_text(json.dumps(settings, indent=2) + "\n")
    info = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_vertices=parcel.m.nflexvert, original_faces=original_faces, refined_vertices=fine.m.nflexvert,
                refined_faces=len(faces_fine), panels=dict(Counter(map(int, panels_fine))), levels=LEVELS,
                half_patch_m=HALF_PATCH_M.tolist(), reference_and_loaded_areas_m2=original_area,
                film_mass_kg=float(fine.envelope["masses"].sum()), peak_initial_strain=fine.envelope["metadata"]["preparation"],
                explicit_step_upper_bound_s=upper_bound, compared_step_s=fine_cfg.dt)
    (RUN / "preparation.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def main():
    if sys.argv[1:] == ["--prepare"]:
        prepare()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(args.config.read_bytes())
    if config["geometry"].endswith("/prepared_mesh.npz"):
        with patch.object(physical_parcel, "wrap_generic", prepared_wrap(config["geometry"])):
            recorder.main()
    else:
        recorder.main()


if __name__ == "__main__":
    main()
