"""Check fixed 60% and 90% randomized inner contours on existing seam assets."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-wrap-r1")))
from check_wrapping import DESIGN, ROOT, check, parcel_profile  # noqa: E402


def main():
    assets = {162: "geometry/kraft_coarse_40.npz", 200: "geometry/kraft_coarse_35.npz"}
    with DESIGN.open(newline="") as stream:
        samples = [(i, row) for i, row in enumerate(csv.DictReader(stream), 1)
                   if row["group"] in ("0.60", "0.90")]
    shell = parcel_profile("pe")["shell"]
    records = []
    for i, row in samples:
        for vertices, mesh in assets.items():
            [result] = check(i, row, shell, mesh=mesh, vertices=vertices, include_cut=False)
            records.append({"mesh": mesh, "vertices": vertices} | result)
        if i % 8 == 0:
            print(f"Processed through sample {i}", flush=True)
    with (RUN / "wrapping.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {asset: {"sha256": hashlib.sha256((ROOT / "bag_sim" / mesh).read_bytes()).hexdigest(),
                       "groups": {group: Counter(row["status"] for row in records
                                                 if row["vertices"] == asset and row["group"] == group)
                                  for group in ("0.60", "0.90")}}
               for asset, mesh in assets.items()}
    with (RUN / "summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
