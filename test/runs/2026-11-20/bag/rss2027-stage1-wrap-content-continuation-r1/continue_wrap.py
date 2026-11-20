"""Check whether a fixed random parcel can be wrapped through feasible intermediate sizes."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-wrap-tangent-r1")))
from relax_tangent import edge_targets, incident_faces, MIN_GAP, relax_once, separating_planes  # noqa: E402
from project_volume import ASSET_VERTICES, DESIGN, FACES, PANELS, RADIUS, measurements  # noqa: E402
from bag_sim.legacy.contents import normalize_contents  # noqa: E402
from bag_sim.wrap import panel_pairs, wrap_generic  # noqa: E402

CASES = (41, 46, 74, 78)
GROWTH_LEVELS = 32
STEPS_PER_LEVEL = 80


def grow_near_contents(local, previous_half, new_half, adjacent):
    normals, bounds = separating_planes(local, previous_half)
    bounds += abs(normals) @ (new_half - previous_half)
    directions = normals[adjacent]
    limits = np.where(adjacent >= 0, bounds[adjacent], -np.inf)
    proposed = local.copy()
    for _ in range(32):
        for direction, limit in zip(directions.transpose(1, 0, 2), limits.T, strict=True):
            violation = np.maximum(limit - np.einsum("ij,ij->i", proposed, direction), 0.)
            proposed += violation[:, None] * direction
        distances = np.einsum("nki,ni->nk", directions, proposed)
        if float(np.max(limits - distances)) <= 1e-10:
            return proposed
    return None


def evaluate(sample, design, destination, strain_band, extra_final_steps, clearance_m, save_highest, local_growth=False):
    row = design[sample - 1]
    width = np.array([float(row[f"inside_{axis}_m"]) for axis in "xy"])
    offset = np.array([float(row[f"offset_{axis}_m"]) for axis in "xy"])
    target = np.array([float(row[f"payload_{axis}_m"]) for axis in "xyz"])
    target_fill = float(row["fill_fraction"])
    base = (0.30 / target_fill) ** (1. / 3.)
    ref = np.column_stack((-ASSET_VERTICES[:, 1], ASSET_VERTICES[:, 0], ASSET_VERTICES[:, 2]))
    ref[:, :2] *= width / [.35, .25]
    item = normalize_contents([dict(shape="box", size_m=target * base + 2. * clearance_m, mass_kg=.20)])[0]
    current, _ = wrap_generic(ref, ref.copy(), FACES, PANELS, [0], [3], item, RADIUS,
                              box_face_clearance=True, content_offset_xy=offset)
    edges = np.unique(np.sort(np.concatenate((FACES[:, [0, 1]], FACES[:, [1, 2]], FACES[:, [2, 0]])), axis=1), axis=0)
    groups = edge_targets(ref, edges)
    pairs = panel_pairs(ref, FACES, PANELS, [0], [3])
    adjacent = incident_faces()
    seed = measurements(ref, current, edges, item)
    result = dict(sample=sample, target_fill=target_fill, seed=seed, stages=[])
    if seed["outside_corners"] or seed["intersecting_triangles"]:
        return result | dict(status="initial_enclosure_failed")
    center = item["position"]
    local = current - center
    previous_scale = base
    best = best_pose = None
    for level, scale in enumerate(np.linspace(base, 1., GROWTH_LEVELS + 1)):
        half = target * scale / 2. + clearance_m
        if local_growth and level:
            proposed = grow_near_contents(local, item["half_size"], half, adjacent)
            if proposed is None:
                result.setdefault("levels_needing_global_scale", []).append(level)
                proposed = local * (scale / previous_scale)
            local = proposed
        else:
            local = local * (scale / previous_scale)
        previous_scale = scale
        item["half_size"] = half
        steps = STEPS_PER_LEVEL + (extra_final_steps if level == GROWTH_LEVELS else 0)
        for _ in range(steps):
            local = relax_once(local, half, groups, pairs, adjacent, strain_band)
        gap = float(np.min(local[pairs[:, 0], 2] - local[pairs[:, 1], 2]))
        measurements_at_level = measurements(ref, local + center, edges, item)
        if measurements_at_level["outside_corners"] or measurements_at_level["intersecting_triangles"]:
            raise ValueError(f"Sample {sample} lost enclosure at level {level}")
        step = dict(level=level, fill_fraction=float(target_fill * scale**3), wall_gap_m=gap,
                    accepted=measurements_at_level["enclosed"] and gap >= MIN_GAP - 1e-9,
                    **measurements_at_level)
        result["stages"].append(step)
        if step["accepted"]:
            best = step
            best_pose = (local, scale)
    if best_pose and (best["level"] == GROWTH_LEVELS or save_highest):
        local, scale = best_pose
        loaded = local + center
        drop = RADIUS + .00015 - loaded[:, 2].min()
        loaded[:, 2] += drop
        origin = center + [0., 0., drop]
        case = destination / f"sample-{sample:03d}"
        case.mkdir(exist_ok=False)
        with (case / "prepared_mesh.npz").open("xb") as stream:
            np.savez_compressed(stream, vertices=ASSET_VERTICES, reference_m=ref, current_m=loaded,
                                faces=FACES, panel_ids=PANELS, payload_origin_m=origin,
                                payload_size_m=target * scale, payload_offset_xy=offset)
    return result | dict(status="target_passed" if result["stages"][-1]["accepted"] else "target_not_accepted",
                         highest_accepted=best)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--strain-band", type=float, default=0.)
    parser.add_argument("--extra-final-steps", type=int, default=0)
    parser.add_argument("--box-clearance-mm", type=float, default=0.)
    parser.add_argument("--save-highest", type=int, nargs="+", choices=CASES, default=[])
    parser.add_argument("--groups", choices=("0.60", "0.90"), nargs="+")
    parser.add_argument("--local-growth", action="store_true")
    args = parser.parse_args()
    if not 0. <= args.strain_band < .05:
        parser.error("--strain-band must be within [0, 0.05)")
    if args.extra_final_steps < 0:
        parser.error("--extra-final-steps must be nonnegative")
    if not np.isfinite(args.box_clearance_mm) or args.box_clearance_mm < 0.:
        parser.error("--box-clearance-mm must be finite and nonnegative")
    with DESIGN.open(newline="") as stream:
        design = list(csv.DictReader(stream))
    samples = CASES
    if args.groups:
        samples = [index for index, row in enumerate(design, 1) if row["group"] in args.groups]
    results = []
    for sample in samples:
        try:
            case = evaluate(sample, design, args.output, args.strain_band, args.extra_final_steps,
                            args.box_clearance_mm * 1e-3, sample in args.save_highest, args.local_growth)
        except ValueError as error:
            if not args.groups or not str(error).startswith(("Content does not fit inside",
                    "Starting triangle has no exterior halfspace", f"Sample {sample} lost enclosure at level")):
                raise
            case = dict(sample=sample, target_fill=float(design[sample - 1]["fill_fraction"]),
                        status="enclosure_failed", reason=str(error), stages=[])
        results.append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "stages"}), flush=True)
    with (args.output / "summary.json").open("x") as stream:
        result = dict(growth_levels=GROWTH_LEVELS, relaxation_steps=STEPS_PER_LEVEL, results=results)
        if args.strain_band:
            result["solver_strain_band"] = args.strain_band
        if args.extra_final_steps:
            result["extra_final_steps"] = args.extra_final_steps
        if args.box_clearance_mm:
            result["box_clearance_mm"] = args.box_clearance_mm
        if args.groups:
            result["groups"] = args.groups
        if args.local_growth:
            result["local_growth"] = True
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
