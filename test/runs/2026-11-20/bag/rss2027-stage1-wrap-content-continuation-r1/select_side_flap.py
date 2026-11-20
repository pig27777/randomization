"""Aim a native YAM grasp at the 60% package's measured free film."""

import argparse
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from screen_side_directions import CASE, backend, physical_parcel, prepared_wrap, screen, shifted_target
from refine_parcel import main as record  # noqa: E402
from rss2027_yam_capture_audit import aperture_box, aperture_material_areas  # noqa: E402
from bag_sim.wrap import triangle_box_intersections  # noqa: E402

REFERENCE = CASE / "side-wrist-clearance"
TARGET = CASE / "side-film-margin"
ADVANCE = CASE / "side-film-advance"
BOX_CORNERS = np.array(list(itertools.product((-1., 1.), repeat=3)))
BOX_FACES = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                      (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
DEPTHS_M = (.078, .089, .100, .111)
OPEN_ROUTE = CASE / "side-film-open-route"


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


def capture_length(points, reference, faces, obstacles, center, axes, extent):
    if any(intersects_box(volume, center, axes, extent) for volume in obstacles):
        return 0.
    clipped = faces[triangle_box_intersections(points, faces, extent, center, axes)]
    local = (points[clipped] - center) @ axes
    return float(aperture_material_areas(local, reference[clipped], extent)[1] / .020)


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
            span_lengths.append(capture_length(points, reference, faces, obstacles, at, axes, extent))
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
    ADVANCE.mkdir(exist_ok=False)
    chosen = chosen | {"close_advance_m": sum(interval) / 2.}
    proof = dict(source_selection=str(TARGET / "selection.json"), close_advance_interval_m=interval, selected=chosen)
    (ADVANCE / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    for filename in ("config.json", "smoke.json"):
        (ADVANCE / filename).write_bytes((TARGET / filename).read_bytes())


def swept_fingers(pads, shift, motion, points, faces, obstacles):
    hit = []
    for name, origin, axes in pads:
        # This envelope contains each rounded face over the entire straight open approach.
        half = np.array([.010, .004, .020]) + .00025 + .5 * abs(motion @ axes)
        center = origin + shift + .5 * motion
        film = np.flatnonzero(triangle_box_intersections(points, faces, half, center, axes))
        rigid = [i for i, box in enumerate(obstacles) if intersects_box(box, center, axes, half)]
        if len(film) or rigid:
            hit.append(dict(finger=name, film_faces=film.tolist(), rigid_boxes=rigid))
    return hit


def finger_frames(model, data):
    return [(name, data.geom_xpos[model.geom(name).id].copy(),
             data.xmat[model.body(name).id].reshape(3, 3).copy()) for name in ("left_pad", "right_pad")]


def old_open_contact():
    trial = ADVANCE.with_name("side-film-advance-corner-half-step") / "recording/trial"
    saved = json.loads((trial / "result.json").read_bytes())
    cfg = backend.Config(**saved["config"])
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    model, data = robot.sim.m, robot.sim.d
    with np.load(trial / "trajectory.npz") as archive:
        data.qpos[:] = archive["final_qpos"]
    mujoco.mj_fwdPosition(model, data)
    obstacles = [box(model, data, name) for name in ("payload_geom_0", "table")]
    return swept_fingers(finger_frames(model, data), np.zeros(3), np.zeros(3),
                         data.flexvert_xpos, model.flex_elem.reshape(-1, 3), obstacles)


def open_route_candidates(config, settled_qpos, prior):
    cfg = backend.Config(**config)
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(cfg)
    model, data = robot.sim.m, robot.sim.d
    mujoco.mj_fwdPosition(model, data)
    pads = finger_frames(model, data)
    center = np.mean([row[1] for row in pads], axis=0)
    axes, extent = pads[0][2], np.array([.010, .004, .020])
    data.qpos[:] = settled_qpos
    mujoco.mj_fwdPosition(model, data)
    points = data.flexvert_xpos.copy()
    reference = robot.sim.parcel.envelope["reference"]
    faces = robot.sim.parcel.envelope["faces"]
    obstacles = [box(model, data, name) for name in ("payload_geom_0", "table")]
    check = prior["yaw_deg"] == cfg.yaw and prior["tilt_deg"] == cfg.tilt
    original = (swept_fingers(pads, prior["shift_m"], prior["close_advance_m"] * robot.approach,
                              points, faces, obstacles) if check else None)
    up = np.array([0., 0., 1.])
    lateral = np.cross(up, robot.approach)
    lateral /= np.linalg.norm(lateral)
    rows = []
    for across, height, depth in itertools.product(np.arange(-.060, .061, .005),
                                                  np.arange(-.030, .041, .005),
                                                  np.arange(.050, .111, .005)):
        shift = across * lateral + height * up
        motion = depth * robot.approach
        target = center + shift + motion
        length = capture_length(points, reference, faces, obstacles, target, axes, extent)
        if length <= 0. or swept_fingers(pads, shift, motion + .002 * robot.approach, points, faces, obstacles):
            continue
        paired = min(length, *[capture_length(points, reference, faces, obstacles,
                                               target + delta * robot.approach, axes, extent)
                               for delta in (-.002, .002)])
        if paired > 0.:
            rows.append(dict(yaw_deg=cfg.yaw, tilt_deg=cfg.tilt, shift_m=shift.tolist(),
                             close_advance_m=float(depth), paired_depth_score_m=paired))
    return rows, original


def prepare_open_route():
    configs = json.loads((REFERENCE / "collision-screen.json").read_bytes())["paths"]
    config = json.loads((ADVANCE / "config.json").read_bytes())
    prior = json.loads((ADVANCE / "selection.json").read_bytes())["selected"]
    trial = ADVANCE / "recording/trial/trajectory.npz"
    with np.load(trial) as archive:
        index = int(np.argmin(abs(archive["time"] - .3)))
        time = float(archive["time"][index])
        settled = archive["qpos"][index]
    np.testing.assert_allclose(time, .3, atol=3e-5, rtol=0.)
    scored = []
    baseline = None
    for row in configs:
        if row["status"] == "reachable":
            cfg = config | {"yaw": row["yaw_deg"], "tilt": row["tilt_deg"]}
            choices, control = open_route_candidates(cfg, settled, prior)
            scored.extend(choices)
            if control is not None:
                baseline = control
    actual = old_open_contact()
    if (baseline is None or not any(row["finger"] == "right_pad" and row["film_faces"] for row in baseline)
            or not any(row["finger"] == "right_pad" and 0 in row["film_faces"] for row in actual)):
        raise ValueError("The old static sweep and recorded stop must identify the right-finger collision")
    scored.sort(key=lambda row: row["paired_depth_score_m"], reverse=True)
    selected = None
    for row in scored:
        cfg = config | {"yaw": row["yaw_deg"], "tilt": row["tilt_deg"]}
        try:
            path = screen(cfg, np.asarray(row["shift_m"]), row["close_advance_m"])
        except RuntimeError as error:
            if not str(error).startswith(("YAM initial pose unreachable", "initial film contact penetration")):
                raise
            row["path"] = str(error)
            continue
        row["path"] = path["status"]
        if path["status"] == "reachable":
            selected = row | {"path_check": {key: value for key, value in path.items() if key != "trace"}}
            break
    OPEN_ROUTE.mkdir(exist_ok=False)
    proof = dict(source_time_s=time, predicted_gap_m=.008, old_open_sweep_conflicts=baseline,
                 old_actual_stop_conflicts=actual,
                 selected=selected, candidates=scored)
    (OPEN_ROUTE / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    if selected:
        for filename, seconds in (("config.json", 0.), ("smoke.json", .02)):
            (OPEN_ROUTE / filename).write_text(json.dumps(cfg | {"smoke_seconds": seconds}, indent=2) + "\n")


def contact_response_case():
    case = CASE / "side-film-advance-corner-response"
    case.mkdir(exist_ok=False)
    proof = json.loads((ADVANCE / "selection.json").read_bytes())
    proof["contact_response_s"] = {"right_round_1": .0005, "right_round_2": .0005}
    (case / "selection.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    for filename in ("config.json", "smoke.json"):
        (case / filename).write_bytes((ADVANCE / filename).read_bytes())


def corner_contact_response(times):
    original = backend.compile_yam_model

    def compile_model(parcel, built, xml):
        model = original(parcel, built, xml)
        root = ET.fromstring(built.xml)
        for name, seconds in times.items():
            geom = model.geom(name).id
            np.testing.assert_array_equal(model.geom_solref[geom], [built.cfg.contact_time, 1.])
            model.geom_solref[geom, 0] = seconds
            root.find(f".//geom[@name='{name}']").set("solref", f"{seconds:.17g} 1")
        built.xml = ET.tostring(root, encoding="unicode")
        return model

    return patch.object(backend, "compile_yam_model", compile_model)


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == ["--prepare-advance"]:
        advance_case()
    elif sys.argv[1:] == ["--prepare-open-route"]:
        prepare_open_route()
    elif sys.argv[1:] == ["--prepare-corner-response"]:
        contact_response_case()
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=Path, required=True)
        args, _ = parser.parse_known_args()
        proof = json.loads((args.config.parent / "selection.json").read_bytes())
        chosen = proof["selected"]
        if chosen is None:
            raise ValueError("No physical target passed the side-flap geometry and YAM path checks")
        response = corner_contact_response(proof["contact_response_s"]) if "contact_response_s" in proof else nullcontext()
        with response, shifted_target(np.asarray(chosen["shift_m"]), chosen.get("close_advance_m")):
            record()
