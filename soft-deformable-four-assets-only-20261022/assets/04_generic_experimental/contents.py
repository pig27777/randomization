"""Independent rigid contents and signed distances used only for initialization."""
from __future__ import annotations
from copy import deepcopy
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

PRESETS = {
    "book": dict(shape="box", size_m=[.10, .15, .03], mass_kg=.25, appearance="book"),
    "small_box": dict(shape="box", size_m=[.06, .08, .016], mass_kg=.12),
    "sphere": dict(shape="sphere", radius_m=.025, mass_kg=.15),
    "cylinder": dict(shape="cylinder", radius_m=.025, height_m=.035, mass_kg=.15),
    "mesh": dict(shape="mesh", mesh_file="example_objects/octahedron.obj", mass_kg=.15),
}


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in np.asarray(values).ravel())


def normalize_contents(items):
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        items = [items]
    result = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            if item not in PRESETS:
                raise ValueError(f"Unknown contents preset {item!r}; choose {list(PRESETS)}")
            item = deepcopy(PRESETS[item])
        else:
            item = deepcopy(item)
        item.setdefault("name", f"item_{i}")
        item.setdefault("xy_m", [0., 0.])
        item.setdefault("yaw_deg", 0.)
        item.setdefault("friction", .6)
        item.setdefault("rgba", [.60, .24, .12, 1.])
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", item["name"]):
            raise ValueError("Contents names must be simple identifiers")
        shape = item.get("shape")
        if shape == "book":
            item["shape"] = shape = "box"
            item.setdefault("appearance", "book")
        if "half_size_m" in item:
            item.setdefault("size_m", (2 * np.asarray(item.pop("half_size_m"))).tolist())
        if "mass_g" in item:
            item.setdefault("mass_kg", item.pop("mass_g") * .001)
        if "center_xy_m" in item:
            item["xy_m"] = item.pop("center_xy_m")
        if shape == "box":
            size = np.asarray(item.get("size_m"), dtype=float)
            if size.shape != (3,):
                raise ValueError("A box needs size_m=[length, width, height]")
            half = size / 2
        elif shape == "sphere":
            half = np.repeat(float(item["radius_m"]), 3)
        elif shape == "cylinder":
            half = np.array([item["radius_m"], item["radius_m"], item["height_m"] / 2], float)
        elif shape == 'mesh':
            path = Path(item['mesh_file'])
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if path.suffix.lower() != '.obj':
                raise ValueError('Mesh contents currently require an OBJ file')
            scale = np.asarray(item.get('scale', [1., 1., 1.]), float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('Mesh scale must contain three positive finite values')
            vertices = np.asarray([list(map(float, line.split()[1:4])) for line in path.read_text().splitlines()
                                   if line.startswith('v ')])
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError('Mesh must contain finite 3D vertices')
            vertices *= scale
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            half = (upper - lower) / 2
            item.update(mesh_file=str(path.resolve()), mesh_center=(upper + lower) / 2, scale=scale)
        else:
            raise ValueError("Supported collision shapes: box, sphere, cylinder, mesh")
        if not np.isfinite(half).all() or np.any(half <= 0):
            raise ValueError("Contents dimensions must be positive and finite")
        if not np.isfinite(item.get("mass_kg", np.nan)) or item["mass_kg"] <= 0:
            raise ValueError("Contents mass_kg must be positive and finite")
        xy = np.asarray(item["xy_m"], float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("xy_m must contain two finite coordinates")
        if not np.isfinite(item["yaw_deg"]) or not np.isfinite(item["friction"]) or item["friction"] < 0:
            raise ValueError("Invalid rotation or friction")
        item["half_size"] = half
        angle = math.radians(item["yaw_deg"])
        c, s = math.cos(angle), math.sin(angle)
        item["rotation"] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        result.append(item)
    if len({item['name'] for item in result}) != len(result):
        raise ValueError("Contents names must be unique")
    return result


def place_contents(items, support_z, wall_radius, origin_xy):
    for i, item in enumerate(items):
        item["position"] = np.array([*(np.asarray(origin_xy) + item['xy_m']),
                                     support_z + 2 * wall_radius + .00075 + item['half_size'][2]])
        item["body_name"] = "bag_contents" if i == 0 else f"bag_contents_{i}"
        item["geom_name"] = "contents_geom" if i == 0 else f"contents_geom_{i}"


def add_contents(world, items, assets):
    for item in items:
        half, shape = item['half_size'], item['shape']
        body = ET.SubElement(world, 'body', name=item['body_name'], pos=numbers(item['position']),
                             euler=numbers([0, 0, math.radians(item['yaw_deg'])]))
        ET.SubElement(body, 'freejoint', name=f"{item['body_name']}_free")
        if shape == 'mesh':
            mesh_name = f"contents_mesh_{item['name']}"
            ET.SubElement(assets, 'mesh', name=mesh_name, file=item['mesh_file'], scale=numbers(item['scale']))
            geometry = dict(mesh=mesh_name, pos=numbers(-item['mesh_center']))
        else:
            size = half if shape == 'box' else half[:1] if shape == 'sphere' else half[[0, 2]]
            geometry = dict(size=numbers(size))
        ET.SubElement(body, 'geom', name=item['geom_name'], type=shape, **geometry,
                      mass=str(item['mass_kg']), rgba='0 0 0 0' if item.get('appearance') == 'book' else numbers(item['rgba']),
                      friction=f"{item['friction']} .005 .0001", condim='3',
                      solref='.004 1', solimp='.99 .999 .0001')
        if item.get('appearance') == 'book':
            def visual(name, size, pos, color):
                ET.SubElement(body, 'geom', name=f"{item['body_name']}_{name}", type='box',
                              size=numbers(size), pos=numbers(pos), rgba=color, mass='0',
                              contype='0', conaffinity='0', group='1')
            h = min(.0008, half[2] / 8)
            visual('pages', half * [.97, .98, .90], [0, 0, 0], '.92 .88 .74 1')
            for sign, name in [(1, 'top'), (-1, 'bottom')]:
                visual(name, [half[0], half[1], h], [0, 0, sign * (half[2] - h)], '.55 .075 .045 1')
            visual('spine', [h, half[1], half[2]], [-half[0] + h, 0, 0], '.32 .035 .025 1')
            # A small cover stripe gives a clear visual orientation.
            visual('title', [half[0] * .45, half[1] * .025, .00003],
                   [0, half[1] * .35, half[2] + .00003], '.9 .65 .26 1')


def signed_distance(points, item):
    """Distance and outward normal in world coordinates, for analytic colliders."""
    local = (np.asarray(points) - item['position']) @ item['rotation']
    half, shape = item['half_size'], item['shape']
    if shape == 'sphere':
        length = np.linalg.norm(local, axis=1)
        normal = local / np.maximum(length[:, None], 1e-12)
        normal[length < 1e-12] = [0, 0, 1]
        distance = length - half[0]
    elif shape in ('box', 'mesh'):
        q = np.abs(local) - half
        outside = np.maximum(q, 0)
        length = np.linalg.norm(outside, axis=1)
        distance = length + np.minimum(q.max(axis=1), 0)
        normal = np.sign(local) * outside / np.maximum(length[:, None], 1e-12)
        inside = length < 1e-12
        ids = np.flatnonzero(inside)
        axes = q[inside].argmax(axis=1)
        normal[inside] = 0
        normal[ids, axes] = np.where(local[ids, axes] >= 0, 1., -1.)
    else:
        radial = np.linalg.norm(local[:, :2], axis=1)
        q = np.stack([radial - half[0], np.abs(local[:, 2]) - half[2]], axis=1)
        outside = np.maximum(q, 0)
        length = np.linalg.norm(outside, axis=1)
        distance = length + np.minimum(q.max(axis=1), 0)
        weights = outside / np.maximum(length[:, None], 1e-12)
        inside = length < 1e-12
        weights[inside] = np.eye(2)[q[inside].argmax(axis=1)]
        normal = np.zeros_like(local)
        normal[:, :2] = local[:, :2] / np.maximum(radial[:, None], 1e-12) * weights[:, :1]
        normal[radial < 1e-12, 0] = weights[radial < 1e-12, 0]
        normal[:, 2] = np.where(local[:, 2] >= 0, 1., -1.) * weights[:, 1]
    return distance, normal @ item['rotation'].T
