"""Put one rigid mesh inside a bag in Newton, let it settle on the table, record it.

Usage: <newton python> -B demo_contents.py CONTENT [--profile bubble] [--seconds 1.5] [--fps 30] [--out outputs]
CONTENT: book | bottle | ball | torus | document   (procedural meshes; torus checks a non-convex shape)

The walls start wrapped around the content: the bubble mailer uses the legacy distance/contact
projection (initialization.py); other profiles use wrap_generic on their front/back panel groups.
Both work against the unchanged empty-bag reference. The content is an independent
free body: it touches the bag only through contact and friction.
Writes outputs/<profile>_<content>.npz with bag frames, content mesh and poses, and checks.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
LEGACY = HERE / "legacy"
sys.path[:0] = [str(HERE), str(LEGACY)]
import newton  # noqa: E402
from asset_mesh import tabletop_mesh  # noqa: E402
from contents import normalize_contents, place_contents  # noqa: E402
from initialization import prepare_loaded_positions  # noqa: E402
from bag_model import parameters  # noqa: E402
from newton_bag import content_mesh, content_shape_config, load_profile_newton  # noqa: E402
from wrap import open_pairs, panel_pairs, wrap_generic  # noqa: E402,F401  (re-exported for proxy_eval)


def box_mesh(hx, hy, hz):
    v = np.array([(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def ball_mesh(r, subdivisions=2):
    t = (1 + 5 ** .5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, faces = {}, []

        def mid(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = faces
    return np.array(v) * r, np.array(f)


def bottle_mesh(r, half_length, n=24):
    """Closed cylinder lying along y."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(angles), r * np.sin(angles)]
    v = [(x, -half_length, z) for x, z in ring] + [(x, half_length, z) for x, z in ring] + [(0, -half_length, 0), (0, half_length, 0)]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, n + i, n + j), (i, n + j, j), (2 * n, i, j), (2 * n + 1, n + j, n + i)]
    return np.array(v, float), np.array(f)


def torus_mesh(major, minor, n_major=36, n_minor=16):
    v, f = [], []
    for i in range(n_major):
        u = 2 * np.pi * i / n_major
        for j in range(n_minor):
            w = 2 * np.pi * j / n_minor
            v.append(((major + minor * np.cos(w)) * np.cos(u), (major + minor * np.cos(w)) * np.sin(u), minor * np.sin(w)))
    idx = lambda i, j: (i % n_major) * n_minor + j % n_minor
    for i in range(n_major):
        for j in range(n_minor):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            f += [(a, b, c), (a, c, d)]
    return np.array(v), np.array(f)


CONTENTS = {
    "book": dict(mesh=lambda: box_mesh(.05, .075, .015), mass=.25, title="Book 100x150x30 mm, 250 g"),
    "bottle": dict(mesh=lambda: bottle_mesh(.03, .07), mass=.30, title="Bottle d60 x 140 mm, 300 g"),
    "ball": dict(mesh=lambda: ball_mesh(.035), mass=.15, title="Ball d70 mm, 150 g"),
    "torus": dict(mesh=lambda: torus_mesh(.04, .012), mass=.12, title="Torus 104 mm (non-convex), 120 g"),
    "document": dict(mesh=lambda: box_mesh(.1, .075, .004), mass=.25, title="Document stack 200x150x8 mm, 250 g"),
}


def volume(v, f):
    return abs(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)


def orient_outward(v, f):
    """Flip triangles so the signed volume is positive (outward normals)."""
    signed = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()
    return f if signed > 0 else f[:, ::-1].copy()


@wp.kernel
def _mesh_distance(mesh: wp.uint64, points: wp.array[wp.vec3], max_dist: float, out: wp.array[float]):
    i = wp.tid()
    query = wp.mesh_query_point_sign_normal(mesh, points[i], max_dist)
    d = max_dist
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        d = wp.length(points[i] - closest) * query.sign
    out[i] = d


def surface_distance(vertices, faces, pose, points):
    """Signed distance of world points to the content surface at pose (px,py,pz,qx,qy,qz,qw)."""
    q = pose[3:]
    x, y, z, w = q
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    local = (points - pose[:3]) @ rotation
    mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3), indices=wp.array(faces.ravel(), dtype=wp.int32))
    out = wp.zeros(len(points), dtype=float)
    wp.launch(_mesh_distance, dim=len(points), inputs=[mesh.id, wp.array(local, dtype=wp.vec3), 1.0, out])
    return out.numpy()


def poke_through(bag_q, bag_faces, panel_ids, content_world, front_ids=(0,), back_ids=(1,)):
    """Largest distance a content vertex sits outside the bag walls, in m (<= 0 means fully inside).

    Panels are near horizontal, so each content vertex is located in the xy projection of the
    upper (front group) and lower (back group) wall triangles and compared with the wall height there.
    """
    worst = -np.inf
    for group, sign in ((front_ids, 1.0), (back_ids, -1.0)):
        tris = bag_q[bag_faces[np.isin(panel_ids, group)]]
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        v0, v1 = (b - a)[:, :2], (c - a)[:, :2]
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = np.abs(det) > 1e-12
        for point in content_world:
            w = point[:2] - a[:, :2]
            beta = np.where(ok, (w[:, 0] * v1[:, 1] - w[:, 1] * v1[:, 0]) / np.where(ok, det, 1), -1)
            gamma = np.where(ok, (v0[:, 0] * w[:, 1] - v0[:, 1] * w[:, 0]) / np.where(ok, det, 1), -1)
            inside = (beta >= 0) & (gamma >= 0) & (beta + gamma <= 1)
            if not inside.any():
                continue
            height = (a[:, 2] + beta * (b - a)[:, 2] + gamma * (c - a)[:, 2])[inside]
            # Upper wall: the point pokes out when above the wall; lower wall: when below it.
            excess = sign * (point[2] - (height.max() if sign > 0 else height.min()))
            worst = max(worst, float(excess))
    return worst


def world_vertices(vertices, pose):
    x, y, z, w = pose[3:]
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return vertices @ rotation.T + pose[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content", choices=sorted(CONTENTS))
    parser.add_argument("--profile", default="bubble")
    parser.add_argument("--set", default="{}", help='JSON overrides of profile params, e.g. {"newton_self_contact_scale": 0.5}')
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=str(HERE / "outputs"))
    args = parser.parse_args()
    wp.config.quiet = True
    spec = CONTENTS[args.content]
    vertices, faces = spec["mesh"]()
    faces = orient_outward(vertices, faces).astype(np.int32)
    # Absolute, because the legacy loader resolves relative mesh paths against its own folder.
    out = Path(args.out).resolve()
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    obj = out / "meshes" / f"{args.content}.obj"
    obj.write_text("".join(f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in vertices)
                   + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces))

    # Wrap the walls around the content's bounding shape.
    profile = json.loads((HERE / "profiles" / f"{args.profile}.json").read_text())
    p = {**profile["params"], **json.loads(args.set)}
    radius = p["collision_radius_m"]
    with np.load(HERE / profile["geometry"]) as g:
        rest = np.array(g["vertices"], float)
        bag_faces = np.array(g["faces"])
        panel_ids = np.array(g["panel_ids"])
        start = np.array(g["initial_vertices"], float) if "initial_vertices" in g else rest.copy()
    shape = (dict(shape="sphere", radius_m=float(np.linalg.norm(vertices, axis=1).max()), mass_kg=spec["mass"])
             if args.content == "ball" else dict(shape="mesh", mesh_file=str(obj), mass_kg=spec["mass"]))
    item = normalize_contents([shape])
    if args.profile == "bubble":
        # The validated legacy loader for this exact two-panel topology.
        gap = float(rest[:, 2].max() - rest[:, 2].min())
        _, _, grid, _ = tabletop_mesh(14, 18, .18, .24, gap, .015)
        place_contents(item, 0.0, radius, (0., 0.))
        initial, init_report = prepare_loaded_positions(
            rest + [0, 0, radius + .0001], bag_faces, grid,
            dict(width_mm=180., height_mm=240., radius_mm=radius * 1e3, rest_gap_mm=gap * 1e3), item, 0.0, (0., 0.))
    else:
        initial, init_report = wrap_generic(rest, start, bag_faces, panel_ids, p["self_contact_front_panel_ids"],
                                            p["self_contact_back_panel_ids"], item[0], radius)
    if init_report["maximum_edge_strain"] > .05:
        raise ValueError(f"Wrapped initial state strains edges by {init_report['maximum_edge_strain']:.3f}")
    center = item[0]["position"]

    def add_content(builder, contact):
        params = parameters(p)
        cfg = content_shape_config(contact, params, density=spec["mass"] / volume(vertices, faces), primitive=False)
        body = builder.add_body(xform=wp.transform(wp.vec3(*map(float, center)), wp.quat_identity()), label=args.content)
        builder.add_shape_mesh(body, mesh=content_mesh(vertices, faces, params), cfg=cfg)
        return body

    bag = load_profile_newton(args.profile, params=json.loads(args.set), initial_vertices=initial, extra=add_content)
    body = bag.extra
    every = max(1, int(round(1 / (args.fps * bag.params["frame_dt_s"]))))
    frames, poses = [bag.positions.astype(np.float32)], [bag.s0.body_q.numpy()[body].copy()]
    initial_clearance = float(surface_distance(vertices, faces, poses[0], bag.positions).min())
    start = time.perf_counter()
    for frame in range(int(round(args.seconds / bag.params["frame_dt_s"]))):
        bag.step()
        if (frame + 1) % every == 0:
            frames.append(bag.positions.astype(np.float32))
            poses.append(bag.s0.body_q.numpy()[body].copy())
    wall = time.perf_counter() - start

    q, pose = bag.positions, poses[-1]
    distance = surface_distance(vertices, faces, pose, q)
    front = np.unique(bag.faces[np.isin(bag.panel_ids, p["self_contact_front_panel_ids"])])
    lo, hi = q[front, :2].min(axis=0), q[front, :2].max(axis=0)
    inside = bool(q[:, 2].min() < pose[2] < q[:, 2].max() and np.all(pose[:2] > lo) and np.all(pose[:2] < hi))
    report = bag.state_report()
    poke = poke_through(q, bag.faces, bag.panel_ids, world_vertices(vertices, pose), p["self_contact_front_panel_ids"], p["self_contact_back_panel_ids"])
    checks = dict(profile=args.profile, content=args.content, title=spec["title"], wall_s=round(wall, 1), realtime_factor=round(args.seconds / wall, 4),
                  init_max_edge_strain=round(init_report["maximum_edge_strain"], 4),
                  initial_wall_clearance_mm=round(initial_clearance * 1e3, 3), overrides=json.loads(args.set),
                  content_inside_bag=inside, content_shift_mm=round(float(np.linalg.norm(pose[:2] - center[:2])) * 1e3, 2),
                  content_drop_mm=round(float(center[2] - pose[2]) * 1e3, 2),
                  min_wall_clearance_mm=round(float(distance.min()) * 1e3, 3), particle_radius_mm=radius * 1e3,
                  wall_penetration_mm=round(max(0., radius - float(distance.min())) * 1e3, 3),
                  content_poke_through_mm=round(poke * 1e3, 3), full_surface_contact=bag.params["newton_full_surface_contact"],
                  content_speed_m_s=round(float(np.linalg.norm(bag.s0.body_qd.numpy()[body][:3])), 5),
                  bag_max_speed_m_s=round(report["max_speed_m_s"], 5), max_edge_stretch=round(report["max_edge_stretch"], 5),
                  finite=report["finite"])
    np.savez_compressed(out / f"{args.profile}_{args.content}.npz", frames=np.array(frames), faces=bag.faces,
                        content_vertices=vertices, content_faces=faces, content_poses=np.array(poses),
                        seconds=args.seconds, checks=json.dumps(checks))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    main()
