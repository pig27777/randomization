"""Repeat the suspended Kraft loading on a conforming jaw-local mesh."""

from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
sys.path[:0] = [str(RUN.with_name(name)) for name in ("rss2027-stage1-kraft-held-load-step-r1",
                                                     "rss2027-stage1-randomized-top-local-mesh-r1")]
import replay_held_load as replay  # noqa: E402
from refine_parcel import areas, physical_parcel, prepared_wrap, split_face, test_topology  # noqa: E402
from record_support_withdrawal import integration_state  # noqa: E402
from rss2027_yam_capture_audit import aperture_box  # noqa: E402

ASSET = RUN / "prepared_mesh.npz"
GEOMETRY = "../" + ASSET.relative_to(ROOT).as_posix()
COMPILED_DT = .00005
original_scene = replay.held_scene
backend = replay.backend


def midpoints(points, edges):
    return np.vstack((points, points[edges].mean(axis=1)))


def get_config():
    original = json.loads((replay.SOURCE / "trial/result.json").read_bytes())
    return backend.Config(**original["config"])


def film_qpos_addresses(sim):
    start = np.array([sim.m.body_jntadr[sim.m.body(f"bag_{i}").id] for i in range(sim.m.nflexvert)])
    return sim.m.jnt_qposadr[start[:, None] + np.arange(3)]


def prepare():
    robot, frame, fingerprint = original_scene()
    sim, cfg = robot.sim, get_config()
    mujoco.mj_fwdPosition(sim.m, sim.d)
    world = sim.d.flexvert_xpos.copy()
    pad = sim.m.geom_bodyid[robot.pad_geom[0]]
    center, basis, bounds = aperture_box(sim.d, sim.d.geom_xpos[robot.pad_geom], pad, robot.gap())
    bounds += [.010, .010, .012]
    with np.load(physical_parcel.HERE / sim.parcel.cfg.geometry) as asset:
        raw, faces, panels = [asset[name] for name in ("vertices", "faces", "panel_ids")]
    triangles = ((world - center) @ basis)[faces]
    selected = np.all(triangles.max(axis=1) >= -bounds, axis=1)
    selected &= np.all(triangles.min(axis=1) <= bounds, axis=1)
    if not selected.any():
        raise ValueError("No film intersects the refined jaw neighborhood")
    edges = np.unique(np.sort(np.concatenate((faces[selected][:, [0, 1]], faces[selected][:, [1, 2]],
                                              faces[selected][:, [2, 0]])), axis=1), axis=0)
    ids = {tuple(edge): index for index, edge in enumerate(edges, len(raw))}
    refined, refined_panels = [], []
    for face, panel in zip(faces, panels, strict=True):
        children = split_face(face, ids)
        refined.extend(children)
        refined_panels.extend([panel] * len(children))
    refined = np.asarray(refined)
    p = sim.parcel
    shift = np.array([*p.cfg.center, 0.])
    reference, current = (p.envelope[name] - shift for name in ("reference", "current"))
    test_topology(midpoints(reference, edges), refined)
    area_checks = {}
    for name, points in (("reference", reference), ("assembled", current), ("held", world)):
        base_area = float(areas(points, faces).sum())
        np.testing.assert_allclose(areas(midpoints(points, edges), refined).sum(), base_area, atol=1e-12, rtol=0.)
        area_checks[name] = base_area
    before = np.linalg.norm(reference[edges[:, 1]] - reference[edges[:, 0]], axis=1)
    with ASSET.open("xb") as stream:
        np.savez_compressed(stream, vertices=midpoints(raw, edges), faces=refined, panel_ids=refined_panels,
                            reference_m=midpoints(reference, edges), current_m=midpoints(current, edges),
                            payload_origin_m=p.envelope["payload_origin"] - shift, payload_size_m=cfg.payload_size,
                            payload_offset_xy=cfg.payload_offset_xy, old_edges=edges)
    prepared = dict(source_frame=frame, source_state_sha256=fingerprint, original_vertices=len(raw),
                    refined_vertices=len(raw) + len(edges), refined_faces=len(refined),
                    selected_faces=int(selected.sum()), film_mass_kg=float(p.envelope["masses"].sum()),
                    areas_m2=area_checks, split_edge_median_m=float(np.median(before)),
                    refined_edge_median_m=float(np.median(before) / 2))
    robot, _frame, digest = refined_scene()
    prepared["refined_initial_state_sha256"] = digest
    prepared["initial_depth_mm"] = backend.measure(robot)["penetration_mm"]
    (RUN / "preparation.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(json.dumps(prepared, indent=2))


def copy_joint_state(old, new, edges):
    for field, address in (("qpos", "jnt_qposadr"), ("qvel", "jnt_dofadr"),
                           ("qacc_warmstart", "jnt_dofadr")):
        source, target = [getattr(sim.d, field) for sim in (old, new)]
        old_addresses, new_addresses = [getattr(sim.m, address) for sim in (old, new)]
        if field == "qpos":
            nodes = [film_qpos_addresses(sim) for sim in (old, new)]
        else:
            nodes = [sim.film_dofs for sim in (old, new)]
        target[nodes[1]] = midpoints(source[nodes[0]], edges)
        for j in range(old.m.njnt):
            if old.m.body(old.m.jnt_bodyid[j]).name.startswith("bag_"):
                continue
            next_j = new.m.joint(old.m.joint(j).name).id
            old_start, new_start = old_addresses[j], new_addresses[next_j]
            old_end = old_addresses[j + 1] if j + 1 < old.m.njnt else len(source)
            target[new_start:new_start + old_end - old_start] = source[old_start:old_end]
    for field in ("ctrl", "act", "mocap_pos", "mocap_quat", "eq_active"):
        getattr(new.d, field)[:] = getattr(old.d, field)
    new.d.time = old.d.time


def refined_scene(source=original_scene, *, movable_support=True):
    coarse, frame, _ = source()
    old = coarse.sim
    cfg = replace(get_config(), geometry=GEOMETRY, initial_state=None, dt=COMPILED_DT)
    support = (patch.object(backend, "parcel_world", partial(replay.world_with_movable_support, lower_floor=True))
               if movable_support else nullcontext())
    with support, patch.object(physical_parcel, "wrap_generic", prepared_wrap(GEOMETRY)):
        robot = backend.scene(cfg)
    new = robot.sim
    with np.load(ASSET) as asset:
        edges = asset["old_edges"]
    copy_joint_state(old, new, edges)
    mujoco.mj_fwdPosition(old.m, old.d)
    backend.refresh(new)
    np.testing.assert_allclose(new.d.flexvert_xpos, midpoints(old.d.flexvert_xpos, edges), atol=1e-12, rtol=0.)
    np.testing.assert_allclose(robot.gap(), coarse.gap(), atol=1e-12, rtol=0.)
    np.testing.assert_allclose(new.com(), old.com(), atol=1e-12, rtol=0.)
    np.testing.assert_allclose(new.parcel.envelope["masses"].sum(), old.parcel.envelope["masses"].sum(), atol=1e-12, rtol=0.)
    np.testing.assert_array_equal(new.d.ctrl, old.d.ctrl)
    failure, health = backend.check_state(new)
    if failure or health["payload_geometry"]["intersecting_triangles"] or backend.measure(robot)["penetration_mm"] > 1.:
        raise ValueError(f"Refined checkpoint violates the original numerical or geometry criterion: {failure}, {health}")
    return robot, frame, hashlib.sha256(integration_state(new).tobytes()).hexdigest()


def main():
    if sys.argv[1:] == ["--prepare"]:
        prepare()
        return
    with patch.object(replay, "held_scene", refined_scene):
        replay.main()


if __name__ == "__main__":
    main()
