"""Write the unified-format geometry of the two-panel bubble mailer (the former legacy bag).

Usage: python -B make_bubble_geometry.py [--nx 14] [--ny 18] [--width-mm 180] [--height-mm 240]
                                         [--radius-mm 1.75] [--seam-mm 15] [--out geometry/bubble_180x240.npz]

Reuses the legacy two-panel topology: panels joined by three side strips, open mouth.
The panel gap follows the contact thickness, so thick cushioned walls start apart.
Panel ids: 0 upper panel, 1 lower panel, 2 side strips. Region ids: 0 padded body, 1 welded seam.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy"))
from asset_mesh import tabletop_mesh  # noqa: E402


def build(nx, ny, width, height, radius, seam):
    gap = 2 * radius + 0.0004
    points, faces, grid, regions = tabletop_mesh(nx, ny, width, height, gap, seam)
    points = points.astype(float)
    # Legacy material frame has length along z and thickness along y; lay it flat with length along y.
    vertices = np.column_stack([points[:, 0], height / 2 - (points[:, 2] - 0.065), points[:, 1] + gap / 2])
    panel_faces = 2 * nx * ny
    panel_ids = np.full(len(faces), 2, np.int32)
    panel_ids[:panel_faces] = 0
    panel_ids[panel_faces:2 * panel_faces] = 1
    uv = np.asarray([(i / nx, j / ny) for _ in (0, 1) for j in range(ny + 1) for i in range(nx + 1)], float)
    return dict(vertices=vertices, faces=faces.astype(np.int32), uv=uv, panel_ids=panel_ids,
                region_ids=regions.astype(np.int32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=14)
    parser.add_argument("--ny", type=int, default=18)
    parser.add_argument("--width-mm", type=float, default=180.)
    parser.add_argument("--height-mm", type=float, default=240.)
    parser.add_argument("--radius-mm", type=float, default=1.75)
    parser.add_argument("--seam-mm", type=float, default=15.)
    parser.add_argument("--out", default=str(HERE / "geometry/bubble_180x240.npz"))
    args = parser.parse_args()
    geometry = build(args.nx, args.ny, args.width_mm * 1e-3, args.height_mm * 1e-3, args.radius_mm * 1e-3, args.seam_mm * 1e-3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **geometry)
    print(args.out, {k: v.shape for k, v in geometry.items()})


if __name__ == "__main__":
    main()
