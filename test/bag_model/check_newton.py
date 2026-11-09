"""Stage-1 checks for the Newton backend; prints one JSON line per check.

Usage: <newton python> -B test/bag_model/check_newton.py cantilever MESH H_MM | press H_MM GSM KPA | profile NAME SECS
  cantilever  PE-like strip (D=2e-6, 46 g/m2) clamped at one end, vs. the inextensible elastica
  press       a box pressing a flat cushioned sheet on the ground, vs. the linear cushion law
  profile     load a profile on the ground and step it
"""
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parents[2] / "bag_sim"
sys.path.insert(0, str(HERE))
import newton  # noqa: E402
from bag_model import parameters  # noqa: E402
from newton_bag import NewtonBag, content_shape_config, load_profile_newton  # noqa: E402

# Elastica reference (droop, reach) in mm for L-h free length, D=2e-6 N m, 46 g/m2, from a converged Newton solve.
ELASTICA = {20.0: (96.15, 9.84), 10.0: (106.38, 9.25), 5.0: (111.49, 8.99)}
L, W = .12, .06


def strip(kind, h):
    nx, ny = int(round(L / h)), int(round(W / h))
    if kind == "grid":
        X, Y = np.meshgrid(np.linspace(0, L, nx + 1), np.linspace(-W / 2, W / 2, ny + 1), indexing="ij")
        idx = lambda i, j: i * (ny + 1) + j
        faces = []
        for i in range(nx):
            for j in range(ny):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                faces += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
        points = np.c_[X.ravel(), Y.ravel()]
    else:
        # Equilateral lattice with edges at 0 and +-60 deg: no edge lies on the fold axis.
        rows, nc = int(round(W / (h * np.sqrt(3) / 2))), nx + 1
        points = np.array([(i * h + (j % 2) * h / 2, j * h * np.sqrt(3) / 2 - rows * h * np.sqrt(3) / 4)
                           for j in range(rows + 1) for i in range(nc)])
        points[:, 0] *= L / (L + h / 2)
        idx = lambda i, j: j * nc + i
        faces = []
        for j in range(rows):
            for i in range(nc - 1):
                if j % 2 == 0:
                    faces += [(idx(i, j), idx(i + 1, j), idx(i, j + 1)), (idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1))]
                else:
                    faces += [(idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)), (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1))]
    vertices = np.c_[points, np.full(len(points), .5)]
    return vertices, np.array(faces, np.int32)


def cantilever(kind, h_mm, seconds=3.0):
    h = h_mm * 1e-3
    vertices, faces = strip(kind, h)
    params = dict(areal_density_gsm=46., membrane_model="elastic", young_pa=2e8, membrane_thickness_m=5e-5,
                  bending_rigidity_nm=2e-6, plasticity=False, memory_ratio=0., bending_damping_nms=1e-9,
                  collision_radius_m=.00025, self_contact=False, velocity_damping_per_s=12., precreased_panel_boundaries=False)
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(Path(tmp) / "strip.npz", vertices=vertices, faces=faces, uv=vertices[:, :2])
        bag = NewtonBag(Path(tmp) / "strip.npz", params, name="strip", support=False)
    bag.fix(np.flatnonzero(vertices[:, 0] < 1.01 * h))
    tip = np.flatnonzero(vertices[:, 0] > L - .75 * h)
    start = time.perf_counter()
    history = []
    for frame in range(int(seconds / .01)):
        bag.step()
        if frame % 30 == 29:
            history.append(.5 - bag.positions[tip, 2].mean())
    q = bag.positions
    report = bag.state_report()
    ref = ELASTICA[float(h_mm)]
    return dict(check="cantilever", mesh=kind, h_mm=h_mm, droop_mm=round((.5 - q[tip, 2].mean()) * 1e3, 2),
                reach_mm=round((q[tip, 0].mean() - h) * 1e3, 2), ref_droop_mm=ref[0], ref_reach_mm=ref[1],
                last_change_mm=round((history[-1] - history[-2]) * 1e3, 3), max_edge_stretch=round(report["max_edge_stretch"], 4),
                max_edge_compression=round(report["max_edge_compression"], 4), wall_s=round(time.perf_counter() - start, 1))


def press(h_mm, gsm, kpa, box=.06, seconds=1.0):
    h = h_mm * 1e-3
    params = dict(json.loads((HERE / "profiles/bubble.json").read_text())["params"])
    params.update(mass_kg=None, areal_density_gsm=gsm, self_contact=False, crease_softening=False, memory_ratio=0.,
                  bending_knee_deg=None, velocity_damping_per_s=20.)
    n = int(round(.16 / h))
    xs = np.linspace(-.08, .08, n + 1)
    vertices = np.array([(x, y, 0.) for x in xs for y in xs])
    idx = lambda i, j: i * (n + 1) + j
    faces = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    radius = params["collision_radius_m"]
    mass = kpa * 1e3 * box * box / 9.81
    half_z = .005

    def add_box(builder, contact):
        z = radius + .00015 + radius + half_z + .0002
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, z), wp.quat_identity()), label="box")
        cfg = content_shape_config(contact, parameters(params), mu=.5, density=mass / (box * box * 2 * half_z))
        builder.add_shape_box(body, hx=box / 2, hy=box / 2, hz=half_z, cfg=cfg)
        return body

    with tempfile.TemporaryDirectory() as tmp:
        np.savez(Path(tmp) / "sheet.npz", vertices=vertices, faces=np.array(faces, np.int32), uv=vertices[:, :2] / .16 + .5)
        bag = NewtonBag(Path(tmp) / "sheet.npz", params, name="press", extra=add_box)
    for _ in range(int(seconds / .01)):
        bag.step()
    q = bag.positions
    bottom = float(bag.s0.body_q.numpy()[bag.extra][2]) - half_z
    under = np.abs(q[:, :2]).max(axis=1) < box / 2
    mid = float(q[under, 2].mean())
    predicted = kpa * 1e3 / params["cushion_soft_modulus_pa_per_m"]
    return dict(check="press", h_mm=h_mm, gsm=gsm, pressure_kpa=kpa, node_mass_mg=round(gsm * 1e-3 * h * h * 1e6, 2),
                predicted_depth_mm=round(predicted * 1e3, 4), top_face_depth_mm=round((radius - (bottom - mid)) * 1e3, 4),
                ground_face_depth_mm=round((radius - mid) * 1e3, 4), contact=bag.contact)


def profile(name, seconds):
    bag = load_profile_newton(name)
    bag.step()
    start = time.perf_counter()
    frames = int(round(seconds / bag.params["frame_dt_s"]))
    for _ in range(frames):
        bag.step()
    wall = time.perf_counter() - start
    return dict(check="profile", sim_s=seconds, realtime_factor=round(seconds / wall, 4), **bag.state_report())


if __name__ == "__main__":
    wp.config.quiet = True
    mode = sys.argv[1]
    if mode == "cantilever":
        result = cantilever(sys.argv[2], float(sys.argv[3]))
    elif mode == "press":
        result = press(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    else:
        result = profile(sys.argv[2], float(sys.argv[3]))
    print(json.dumps(result, ensure_ascii=False), flush=True)
