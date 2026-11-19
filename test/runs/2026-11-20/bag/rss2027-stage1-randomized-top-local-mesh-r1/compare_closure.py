"""Compare membrane height and closure loads in the matched mesh experiments."""

import csv
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
sys.path[:0] = [str(RUN.with_name("rss2027-stage1-reference-curves-r1")),
               str(RUN.with_name("rss2027-stage1-top-aperture-height-r1"))]
from export_curves import COLUMNS, export_case  # noqa: E402
from measure_height import replay  # noqa: E402


def main():
    columns = {name: index for index, name in enumerate(COLUMNS)}
    result, records = {}, []
    for variant in ("coarse", "refined"):
        run = (RUN / variant / "recording/trial").relative_to(ROOT / "test/runs")
        case = export_case(variant, "pe", run, {})
        rows = replay(case, columns)
        closure = [row for row in rows if row["phase"] == "close"]
        frames = {row[0]: row for row in case["samples"]}
        for row in rows:
            membrane_height = row["membrane_z_max_m"]
            row["highest_membrane_above_tip_m"] = membrane_height + .020 if membrane_height is not None else None
            row["downforce_N"] = frames[row["t_s"]][columns["downforce_N"]]
            row["left_inner_candidates"] = frames[row["t_s"]][columns["left_inner_candidates"]]
            row["right_inner_candidates"] = frames[row["t_s"]][columns["right_inner_candidates"]]
        present = [row for row in closure if row["aperture_reference_length_m"] > 1e-9]
        first_lift = next((row for row in rows if row["phase"] == "lift"), None)
        result[variant] = dict(closure_frames=len(closure), material_present_frames=len(present),
                               peak_material=max(present, key=lambda row: row["aperture_reference_length_m"])
                               if present else None,
                               peak_over_tip=max((row["highest_membrane_above_tip_m"] for row in closure
                                                  if row["highest_membrane_above_tip_m"] is not None), default=None),
                               first_lift=first_lift, normal_press=case["normal_press"], outcome=case["outcome"])
        records.extend(rows)
    with (RUN / "closure-comparison.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    with (RUN / "closure-geometry.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
