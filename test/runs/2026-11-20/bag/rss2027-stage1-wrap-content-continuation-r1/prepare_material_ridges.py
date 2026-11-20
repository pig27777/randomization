"""Prepare three repository films around the same archived 60% content geometry."""

import csv
import json
from unittest.mock import patch

from equilibrate_wrap import RUN, backend, physical_parcel, prepare, prepared_wrap
from prepare_aligned_fold import DESIGN, record_case, sample_fold

SOURCE = RUN / "fold-aligned-60-side/sample-058/smoke.json"
PRESSURE = RUN.with_name("rss2027-stage1-whole-bag-pressure-pilots-r1")
MATERIAL_FIELDS = ("bag_profile", "material", "dt", "friction", "contact_time", "film_contact_time_s",
                   "bend_damping_ratio", "rayleigh", "ccd_tolerance_m")


def main():
    geometry = json.loads(SOURCE.read_bytes())
    with DESIGN.open(newline="") as stream:
        shape = list(csv.DictReader(stream))[57]
    target = RUN / "fold-aligned-60-materials-clearance"
    target.mkdir(exist_ok=False)
    results = {}
    for material in ("bubble", "kraft", "paperboard"):
        reference = PRESSURE / material / "recording/trial/result.json"
        original = json.loads(reference.read_bytes())
        if original["warnings"] or original["peaks"]["penetration_mm"] > 1.:
            raise ValueError(f"The archived pressure case failed for {material}")
        inputs = {name: original["config"][name] for name in MATERIAL_FIELDS}
        if material == "paperboard":
            # The aligned reference mesh has an explicit bending limit of 43.36 µs.
            inputs["dt"] = .000025
        config = geometry | inputs
        cfg = backend.Config(**config)
        with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
            parcel, _xml = backend.parcel_world(cfg)
        radius = parcel.cfg.radius
        wall_gap = 2. * radius + .0002
        aligned, best, *shape_data = sample_fold(58, shape, radius_m=radius, wall_gap_m=wall_gap)
        result = dict(material_settings=inputs, collision_radius_m=radius, wall_gap_m=wall_gap,
                      alignment=aligned, relaxed=None)
        if best is not None and best[0] <= .05:
            seeds = target / "inputs" / material
            seeds.mkdir(parents=True)
            record_case(seeds, aligned, best, *shape_data, radius_m=radius, config=config)
            relaxed = target / material
            relaxed.mkdir()
            result["relaxed"] = prepare(seeds / "sample-058", relaxed, 1500, True, wall_gap_m=wall_gap)
        results[material] = result
        print(material, radius, wall_gap, bool(result["relaxed"] and result["relaxed"]["exported"]), flush=True)
    (target / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
