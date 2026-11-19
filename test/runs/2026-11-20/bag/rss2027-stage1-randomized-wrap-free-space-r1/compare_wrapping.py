"""Measure how the interim floor and sealed rim constrain wrapping."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-volume-projection-r1")))
from project_volume import ASSET_VERTICES, DESIGN, FACES, PANELS, RADIUS, measurements  # noqa: E402
from bag_sim.legacy.contents import normalize_contents  # noqa: E402
from bag_sim import wrap  # noqa: E402

CASES = (4, 127, 41, 46, 74, 78)


def alternate_wrap(mode):
    original = inspect.getsource(wrap.wrap_generic)
    floor = """        if not seam_clearance:
            # Box initialization is translated above the table only after both
            # sheets have cleared the solid; an early floor blocks the bottom.
            q[:, 2] = np.maximum(q[:, 2], floor)
"""
    sealed_rim = """                local[shared, :2] = seam_reference[:, :2]
                local[shared, 2] = 0.
"""
    mobile_rim = """                local[shared, 2] = 0.
                side = local[shared, :2]
                rim = half[:2] + radius + .0003
                deficit = rim - abs(side)
                inside = np.flatnonzero(np.all(deficit > 0., axis=1))
                axis = np.argmin(deficit[inside], axis=1)
                direction = np.sign(side[inside, axis])
                direction = np.where(direction == 0., np.where(seam_reference[inside, axis] >= 0., 1., -1.), direction)
                side[inside, axis] = direction * rim[axis]
                local[shared, :2] = side
"""
    bounded_rim = """                local[shared, 2] = 0.
                side = local[shared, :2]
                base = seam_reference[:, :2]
                outer = half[:2] + radius + .0003
                on_negative = np.isclose(base, base.min(axis=0), atol=1e-9, rtol=0.)
                on_positive = np.isclose(base, base.max(axis=0), atol=1e-9, rtol=0.)
                if not (on_negative | on_positive).any(axis=1).all():
                    raise ValueError("Unexpected interior sealed node")
                side = np.where(on_negative, np.minimum(side, -outer), side)
                side = np.where(on_positive, np.maximum(side, outer), side)
                local[shared, :2] = side
"""
    substitutions = {"free_space": (floor, ""), "mobile_seam": (sealed_rim, mobile_rim),
                     "bounded_seam": (sealed_rim, bounded_rim), "bounded_step": (sealed_rim, bounded_rim)}
    old, new = substitutions[mode]
    if original.count(old) != 1:
        raise ValueError(f"Expected exactly one {mode} projection")
    namespace = vars(wrap).copy()
    tested = original.replace(old, new)
    if mode == "bounded_step":
        clearance = inspect.getsource(wrap._clear_box_sheet)
        previous, capped = "local[:, 2] += sign * dz", "local[:, 2] += sign * np.minimum(dz, .002)"
        if clearance.count(previous) != 1:
            raise ValueError("Expected exactly one face-height correction")
        tested = clearance.replace(previous, capped) + "\n\n" + tested
        original = clearance + "\n\n" + original
    exec(compile(tested, str(Path(wrap.__file__)), "exec"), namespace)
    return namespace["wrap_generic"], hashlib.sha256(original.encode()).hexdigest()


def check(row, algorithm, edges):
    reference = np.column_stack((-ASSET_VERTICES[:, 1], ASSET_VERTICES[:, 0], ASSET_VERTICES[:, 2]))
    reference[:, :2] *= [float(row[f"inside_{axis}_m"]) / original for axis, original in zip("xy", (.35, .25))]
    size = [float(row[f"payload_{axis}_m"]) for axis in "xyz"]
    offset = [float(row[f"offset_{axis}_m"]) for axis in "xy"]
    item = normalize_contents([dict(shape="box", size_m=size, mass_kg=.20)])[0]
    coords, prep = algorithm(reference, reference.copy(), FACES, PANELS, [0], [3], item, RADIUS,
                             box_face_clearance=True, content_offset_xy=offset)
    np.testing.assert_allclose(coords[:, 2].min(), RADIUS + .00015, rtol=0., atol=1e-12)
    np.testing.assert_allclose(item["position"][:2], offset, rtol=0., atol=1e-12)
    stats = measurements(reference, coords, edges, item)
    np.testing.assert_allclose(stats["maximum_edge_strain"], prep["maximum_edge_strain"], rtol=0., atol=1e-12)
    return stats | dict(wall_gap_m=prep["minimum_wall_gap_m"], seam_clearance=prep["box_seam_clearance"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("free_space", "mobile_seam", "bounded_seam", "bounded_step"),
                        default="free_space")
    args = parser.parse_args()
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    edges = np.unique(np.sort(np.concatenate((FACES[:, [0, 1]], FACES[:, [1, 2]], FACES[:, [2, 0]])), axis=1), axis=0)
    variant, source_sha256 = alternate_wrap(args.mode)
    results = []
    for sample in CASES:
        row = design[sample - 1]
        result = dict(sample=sample, fill_fraction=float(row["fill_fraction"]),
                      original=check(row, wrap.wrap_generic, edges))
        result[args.mode] = check(row, variant, edges)
        results.append(result)
        print(json.dumps(result), flush=True)
    output = "summary.json" if args.mode == "free_space" else args.mode.replace("_", "-") + ".json"
    with (RUN / output).open("x") as stream:
        stream.write(json.dumps(dict(original_wrap_function_sha256=source_sha256, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
