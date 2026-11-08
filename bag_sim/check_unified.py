"""Checks for the unified bag model; prints one JSON line per check.

Usage: python -B check_unified.py regression|profiles|cushion [args]
  regression NAME      unified profile vs. the original runtime.py (reference/) on the same asset (pe, kraft, paperboard)
  profiles NAME SECS   load a profile on the table and step it for SECS simulated seconds
  cushion H_MM GSM KPA a box pressing a flat cushioned sheet on the table, vs. the predicted law
"""
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bag_model  # noqa: E402

ASSETS = {"pe": "01_pe_film", "kraft": "02_kraft_paper", "paperboard": "03_paperboard_envelope"}


def regression(name, substeps=40):
    sys.path.insert(0, str(HERE / "reference"))
    import runtime
    old = runtime.load_paper(HERE / "reference" / ASSETS[name])
    old.step_substeps(substeps)
    reference = old.positions.copy()
    old.close()
    new = bag_model.load_profile(name)
    new.step_substeps(substeps)
    difference = float(np.abs(new.positions - reference).max())
    report = new.state_report()
    new.close()
    return dict(check="regression", profile=name, substeps=substeps, max_position_difference_m=difference,
                warnings=report["warnings"], finite=report["finite"])


def profiles(name, seconds):
    bag = bag_model.load_profile(name)
    start = time.perf_counter()
    frames = int(round(seconds / bag.params["frame_dt_s"]))
    for _ in range(frames):
        bag.step()
    wall = time.perf_counter() - start
    report = bag.state_report()
    bag.close()
    keys = ("mass_kg", "membrane_model", "cushion", "finite", "max_edge_stretch", "max_edge_compression", "min_vertex_z_m",
            "max_vertex_z_m", "max_speed_m_s", "max_crease_softening", "warnings", "contacts", "solver_iterations")
    return dict(check="profiles", profile=name, sim_s=seconds, realtime_factor=round(seconds / wall, 5),
                **{k: report[k] for k in keys})


def cushion(h_mm, gsm, kpa, box=0.06, seconds=0.6):
    h = h_mm * 1e-3
    params = dict(json.loads((HERE / "profiles/bubble.json").read_text())["params"])
    params.update(mass_kg=None, areal_density_gsm=gsm, self_contact=False, crease_softening=False,
                  self_contact_front_panel_ids=[], self_contact_back_panel_ids=[], velocity_damping_per_s=20.)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n = int(round(0.16 / h))
        xs = np.linspace(-0.08, 0.08, n + 1)
        vertices = np.array([(x, y, 0.) for x in xs for y in xs])
        idx = lambda i, j: i * (n + 1) + j
        faces = []
        for i in range(n):
            for j in range(n):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                faces += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
        np.savez(tmp / "sheet.npz", vertices=vertices, faces=np.array(faces, np.int32), uv=vertices[:, :2] / 0.16 + .5)
        half = box / 2
        corners = np.array([(x, y, z) for x in (-half, half) for y in (-half, half) for z in (-.005, .005)])
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        (tmp / "box.obj").write_text("".join(f"v {x} {y} {z}\n" for x, y, z in corners)
                                     + "".join(f"f {a+1} {b+1} {c+1} {d+1}\n" for a, b, c, d in quads))
        radius = params["collision_radius_m"]
        top = radius + 0.00015 + radius + 0.005 + 0.0002
        mass = kpa * 1e3 * box * box / 9.81
        bag = bag_model.BagModel(tmp / "sheet.npz", params, name="cushion_check", base_dir=tmp,
                                 contents=[dict(name="box", mesh_path="box.obj", mass_kg=mass, position_m=[0, 0, top])])
        for _ in range(int(round(seconds / params["frame_dt_s"]))):
            bag.step()
        body = bag.content_body_ids[0]
        bottom = float(bag.data.xpos[body][2]) - 0.005
        under = np.abs(bag.positions[:, :2]).max(axis=1) < half
        mid = float(bag.positions[under, 2].mean())
        speed = float(np.abs(bag.data.cvel[body]).max())
        bag.close()
    depth_top, depth_bottom = radius - (bottom - mid), radius - mid
    grid = np.linspace(0, params["cushion_range_m"] * 1.5, 4001)
    predicted = float(np.interp(kpa * 1e3, bag_model.cushion_pressure(params, grid), grid))
    return dict(check="cushion", h_mm=h_mm, gsm=gsm, pressure_kpa=kpa, node_mass_mg=round(gsm * 1e-3 * h * h * 1e6, 2),
                predicted_depth_mm=round(predicted * 1e3, 3), top_face_depth_mm=round(depth_top * 1e3, 3),
                table_face_depth_mm=round(depth_bottom * 1e3, 3), box_speed=round(speed, 5))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "regression":
        result = regression(sys.argv[2])
    elif mode == "profiles":
        result = profiles(sys.argv[2], float(sys.argv[3]))
    else:
        result = cushion(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    print(json.dumps(result, ensure_ascii=False), flush=True)
