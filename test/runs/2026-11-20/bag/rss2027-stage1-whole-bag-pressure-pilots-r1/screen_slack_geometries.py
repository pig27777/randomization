"""Find additional valid randomized bag and contents geometries for each material."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-wrap-r1")))
from check_wrapping import DESIGN, check, parcel_profile  # noqa: E402


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with DESIGN.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    results = {}
    for material in ("pe", "paperboard", "kraft", "bubble"):
        indices = range(1, len(rows) + 1) if material == "bubble" else (2, 4, 27, 127)
        shell = parcel_profile(material)["shell"]
        records = []
        for index in indices:
            shape = rows[index - 1]
            cavity = [float(shape[f"inside_{axis}_m"]) for axis in "xy"]
            payload = [float(shape[f"payload_{axis}_m"]) for axis in "xyz"]
            l, w, h = payload
            maximum_slack_area = 2. * (np.prod(cavity) - (l * w + l * h + w * h))
            record = next(check(index, shape, shell, include_cut=False))
            records.append(dict(record, cavity_m=cavity, payload_m=payload, slack_area_upper_m2=float(maximum_slack_area),
                                payload_offset_m=[float(shape[f"offset_{axis}_m"]) for axis in "xy"]))
        results[material] = records
        accepted = [r for r in records if r["status"] == "enclosed"]
        print(material, len(accepted), [(r["sample"], r["fill_fraction"], r["slack_area_upper_m2"]) for r in accepted],
              flush=True)
    data = dict(design=DESIGN.relative_to(RUN.parents[4]).as_posix(), geometries=results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
