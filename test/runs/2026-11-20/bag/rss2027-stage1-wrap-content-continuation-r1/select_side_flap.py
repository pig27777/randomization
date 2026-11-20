"""Aim a native YAM grasp at the 60% package's measured free film."""

import argparse
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from screen_side_directions import CASE, backend, physical_parcel, prepared_wrap, screen, shifted_target
from refine_parcel import main as record  # noqa: E402
from rss2027_yam_capture_audit import aperture_box, aperture_material_areas  # noqa: E402
from bag_sim.wrap import triangle_box_intersections  # noqa: E402

REFERENCE = CASE / "side-wrist-clearance"
TARGET = CASE / "side-film-margin"
BOX_CORNERS = np.array(list(itertools.product((-1., 1.), repeat=3)))
BOX_FACES = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                      (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
DEPTHS_M = (.078, .089, .100, .111)


def box(model, data, name):
    geom = model.geom(name)
    pos = data.geom_xpos[geom.id].copy()
    rotation = data.geom_xmat[geom.id].reshape(3, 3).copy()
    half = geom.size.copy()
    corners = (BOX_CORNERS * half) @ rotation.T + pos
    return pos, half, rotation, corners


def intersects_box(volume, center, rotation, extent):
    pos, half, axes, corners = volume
    if np.all(abs((center - pos) @ axes) < half):
        return True
    return bool(triangle_box_intersections(corners, BOX_FACES, extent, center, rotation).any())


def candidates(config, settled_qpos):
    cfg = backend.Config(**config)
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    model, data = robot.sim.m, robot.sim.d
    center, axes, extent = aperture_box(data, data.geom_xpos[robot.pad_geom],
                                        model.geom_bodyid[robot.pad_geom[0]], .008)
    center, axes = center.copy(), axes.copy()
    data.qpos[:] = settled_qpos
    mujoco.mj_fwdPosition(model, data)
    points = data.flexvert_xpos.copy()
    reference = robot.sim.parcel.envelope["reference"]
    faces = robot.sim.parcel.envelope["faces"]
    obstacles = [box(model, data, name) for name in ("payload_geom_0", "table")]
    up = np.array([0., 0., 1.])
    lateral = np.cross(up, robot.approach)
    lateral /= np.linalg.norm(lateral)
    for across, height in itertools.product(np.linspace(-.045, .045, 7), np.linspace(-.025, .025, 7)):
        shift = across * lateral + height * up
        span_lengths = []
        for depth in DEPTHS_M:
            at = center + shift + depth * robot.approach
            if any(intersects_box(volume, at, axes, extent) for volume in obstacles):
                span_lengths.append(0.)
                continue
            clipped = faces[triangle_box_intersections(points, faces, extent, at, axes)]
            local = (points[clipped] - at) @ axes
            _, rest_area = aperture_material_areas(local, reference[clipped], extent)
            span_lengths.append(float(rest_area / .020))
        score = max(min(a, b) for a, b in zip(span_lengths[:-1], span_lengths[1:], strict=True))
        yield dict(yaw_deg=cfg.yaw, tilt_deg=cfg.tilt, shift_m=shift.tolist(),
                   span_equivalent_m=span_lengths, paired_depth_score_m=score)


def prepare():
    paths = json.loads((REFERENCE / "collision-screen.json").read_bytes())["paths"]
    config = json.loads((REFERENCE / "config.json").read_bytes())
    with np.load(REFERENCE / "recording/trial/trajectory.npz") as archive:
        index = int(np.argmin(abs(archive["time"] - .3)))
        settled_qpos = archive["qpos"][index]
        time = float(archive["time"][index])
    np.testing.assert_allclose(time, .3, atol=3e-5, rtol=0.)
    scored = []
    for path in paths:
        if path["status"] == "reachable":
            cfg = config | {"yaw": path["yaw_deg"], "tilt": path["tilt_deg"]}
            scored.extend(candidates(cfg, settled_qpos))
    scored.sort(key=lambda row: row["paired_depth_score_m"], reverse=True)
    selected = None
    for row in scored:
        if row["paired_depth_score_m"] <= 0.:
            break
        candidate = config | {"yaw": row["yaw_deg"], "tilt": row["tilt_deg"]}
        try:
            path = screen(candidate, np.asarray(row["shift_m"]))
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            row["path"] = str(error)
            continue
        row["path"] = path["status"]
        if path["status"] == "reachable":
            selected = row | {"path_check": {k: v for k, v in path.items() if k != "trace"}}
            break
    TARGET.mkdir(exist_ok=False)
    report = dict(source_time_s=time, predicted_gap_m=.008, depth_m=DEPTHS_M, selected=selected, candidates=scored)
    (TARGET / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if selected:
        settings = config | {"yaw": selected["yaw_deg"], "tilt": selected["tilt_deg"]}
        for name, seconds in (("config", 0.), ("smoke", .02)):
            (TARGET / f"{name}.json").write_text(json.dumps(settings | {"smoke_seconds": seconds}, indent=2) + "\n")


def advance_case():
    original = json.loads((TARGET / "selection.json").read_bytes())
    chosen = original["selected"]
    if chosen is None:
        raise ValueError("No selected film margin has an admissible insertion interval")
    retained = chosen["span_equivalent_m"]
    first = max(range(len(retained) - 1), key=lambda i: min(retained[i], retained[i + 1]))
    interval = DEPTHS_M[first:first + 2]
    case = CASE / "side-film-advance"
    case.mkdir(exist_ok=False)
    chosen = chosen | {"close_advance_m": sum(interval) / 2.}
    proof = dict(source_selection=str(TARGET / "selection.json"), close_advance_interval_m=interval, selected=chosen)
    (case / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    for filename in ("config.json", "smoke.json"):
        (case / filename).write_bytes((TARGET / filename).read_bytes())


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == ["--prepare-advance"]:
        advance_case()
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, required=True)
        args, _ = parser.parse_known_args()
        chosen = json.loads((args.config.parent / "selection.json").read_bytes())["selected"]
        if chosen is None:
            raise ValueError("No physical target passed the side-flap geometry and YAM path checks")
        with shifted_target(np.asarray(chosen["shift_m"]), chosen.get("close_advance_m")):
            record()
