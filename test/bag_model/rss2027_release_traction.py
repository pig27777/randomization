"""Measure simultaneous loaded inner-surface sliding during a whole-bag release."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rss2027_micropatch import sustained_starts


def inner_loads(contacts):
    normal = np.zeros(2)
    sliding = np.zeros(2)
    for contact in contacts:
        force, *shear = contact["force_contact_N"]
        if not contact["inner"] or force <= 0.:
            continue
        pad = contact["pad"]
        normal[pad] += force
        velocity = contact["tangent_constraint_velocity_m_s"]
        if velocity and min(contact["friction"]) > 0.:
            ratio = np.linalg.norm(np.asarray(shear) / contact["friction"]) / force
            if ratio >= .95 and np.linalg.norm(velocity) > .002:
                sliding[pad] += force
    return normal, sliding


def audit(run):
    frames = json.loads((run / "recording/release-contacts.json").read_text())
    with (run / "capture/capture-audit.csv").open() as stream:
        curve = {float(row["t_s"]): row for row in csv.DictReader(stream)}
    rows = []
    for frame in frames:
        time = frame["t_s"]
        sample = curve[time]
        normal, sliding = inner_loads(frame["contacts"])
        positive_load = float(sample["tension_N"]) > .001 and float(sample["support_N"]) <= .05
        dual = bool(min(normal) > .001)
        classified = frame["phase"] == "open" and positive_load and dual and bool((sliding >= .5 * normal).all())
        rows.append(dict(t_s=time, phase=frame["phase"], actual_gap_m=float(sample["actual_gap_m"]),
                         pull_N=float(sample["tension_N"]), left_inner_normal_N=float(normal[0]),
                         right_inner_normal_N=float(normal[1]), left_sliding_normal_N=float(sliding[0]),
                         right_sliding_normal_N=float(sliding[1]), bilateral_loaded_inner=dual and positive_load,
                         bilateral_sliding=classified))
    if not rows:
        raise ValueError("No raw contact observations")
    events = sustained_starts(np.array([row["t_s"] for row in rows]),
                              [row["bilateral_sliding"] for row in rows])
    result = dict(frames=len(rows), hold_bilateral=sum(row["phase"] == "lift_hold" and row["bilateral_loaded_inner"]
                                                     for row in rows),
                  release_bilateral=sum(row["phase"] == "open" and row["bilateral_loaded_inner"] for row in rows),
                  first_sustained_bilateral_sliding=rows[events[0]] if events else None)
    with (run / "release-traction.csv").open("x", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        output.writeheader()
        output.writerows(rows)
    with (run / "release-traction.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audit(args.run)


if __name__ == "__main__":
    main()
