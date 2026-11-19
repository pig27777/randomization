"""Check which archived Stage 1 geometries the existing two-sheet mesh can enclose."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
sys.path.insert(0, str(ROOT))
from bag_sim.physical_parcel import Config, load_envelope, parcel_profile  # noqa: E402

DESIGN = RUN.parents[1].parent / "2026-11-19/bag/rss2027-stage1-r1/geometry/geometry.csv"
MESH = "geometry/kraft_edge_sealed_35.npz"
GEOMETRY_ERRORS = {
    "Content does not fit inside": "initializer_seam_allowance",
    "initial film strain exceeds": "initial_strain",
    "initial film does not enclose": "initial_enclosure",
}


def check(sample_id, row, shell, *, mesh=MESH, vertices=142, include_cut=True):
    cut = np.array([row["cut_x_m"], row["cut_y_m"]], dtype=float)
    seams = np.array([[row[f"seam_{axis}_{side}_m"] for side in ("minus", "plus")]
                      for axis in ("x", "y")], dtype=float)
    inner = cut - seams.sum(axis=1)
    shift = (seams[:, 0] - seams[:, 1]) / 2
    offset = np.array([row["offset_x_m"], row["offset_y_m"]], dtype=float)
    size = tuple(float(row[f"payload_{axis}_m"]) for axis in "xyz")
    fill = float(row["fill_fraction"])
    np.testing.assert_allclose(inner, [float(row[f"inside_{ax}_m"]) for ax in "xy"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.prod(size) / float(row["capacity_m3"]), fill, rtol=0, atol=1e-12)

    contours = [("inside_seams", inner, shift, offset)]
    if include_cut:
        contours.insert(0, ("cut", cut, np.zeros(2), offset + shift))
    for contour, width, center, relative_offset in contours:
        cfg = Config(geometry=mesh, bag_size=tuple(width), center=tuple(center),
                     payload_size=size, payload_offset_xy=tuple(relative_offset), **shell)
        record = dict(sample=sample_id, group=row["group"], fill_fraction=fill, contour=contour,
                      status="enclosed", maximum_edge_strain=None, minimum_wall_gap_m=None,
                      error="")
        try:
            envelope = load_envelope(cfg)
        except ValueError as exc:
            label = next((kind for prefix, kind in GEOMETRY_ERRORS.items() if str(exc).startswith(prefix)), None)
            if label is None:
                raise
            record.update(status=label, error=str(exc))
        else:
            np.testing.assert_allclose(envelope["payload_origin"][:2], offset + shift, rtol=0, atol=1e-12)
            meta = envelope["metadata"]
            prep = meta["preparation"]
            if (meta["vertices"] != vertices or prep["maximum_edge_strain"] > .05
                    or prep["payload_intersecting_triangles"]
                    or np.any(np.abs(prep["payload_corner_winding"]) < .5)):
                raise RuntimeError(f"Sample {sample_id} violated the existing geometry check: {meta}")
            record.update(maximum_edge_strain=prep["maximum_edge_strain"],
                          minimum_wall_gap_m=prep["minimum_wall_gap_m"])
        yield record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    shell = parcel_profile("pe")["shell"]
    with DESIGN.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for sample_id, row in enumerate(rows, start=1):
        records.extend(check(sample_id, row, shell))
        if sample_id % 8 == 0:
            print(f"{sample_id}/{len(rows)}", flush=True)
    counts = {group: {contour: Counter(row["status"] for row in records
                                       if row["group"] == group and row["contour"] == contour)
                      for contour in ("cut", "inside_seams")}
              for group in dict.fromkeys(row["group"] for row in rows)}
    for group, selection in counts.items():
        expected = sum(row["group"] == group for row in rows)
        if any(sum(contour.values()) != expected for contour in selection.values()):
            raise RuntimeError(f"Incomplete wrapping checks for group {group}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "wrapping.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = dict(hostname=platform.node(), samples=len(rows), attempts=len(records),
                   input_sha256=hashlib.sha256(DESIGN.read_bytes()).hexdigest(),
                   mesh=MESH, mesh_sha256=hashlib.sha256((ROOT / "bag_sim" / MESH).read_bytes()).hexdigest(),
                   shell=shell, groups=counts)
    with (args.output / "summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
