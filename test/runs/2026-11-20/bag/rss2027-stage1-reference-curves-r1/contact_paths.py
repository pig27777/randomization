"""Integrate the load-weighted local sliding speeds in archived whole-bag releases."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
DATA = RUN / "stage1_whole_bag_data.json"
ROOT = RUN.parents[4]


def observations(case, columns):
    trial = ROOT / case["trial"]
    path = trial.parent / "release-contacts.json"
    frames = json.loads(path.read_bytes())
    source = ROOT / case["contact_source"]
    with source.open(newline="") as stream:
        native = {float(row["t_s"]): row for row in csv.DictReader(stream)}
    trace = {row[columns["t_s"]]: row for row in case["samples"]}
    rows = []
    for frame in frames:
        if frame["phase"] != "open":
            continue
        t = frame["t_s"]
        reference = native[t]
        trace_row = trace[t]
        load, sliding, weighted_speed = np.zeros((3, 2))
        for contact in frame["contacts"]:
            normal, *shear = contact["force_contact_N"]
            if not contact["inner"] or normal <= 0.:
                continue
            pad = contact["pad"]
            load[pad] += normal
            friction = contact["friction"]
            velocity = contact["tangent_constraint_velocity_m_s"]
            if not velocity or min(friction) <= 0.:
                continue
            speed = np.linalg.norm(velocity)
            if np.linalg.norm(np.asarray(shear) / friction) >= .95 * normal and speed > .002:
                sliding[pad] += normal
                weighted_speed[pad] += normal * speed
        np.testing.assert_allclose(load, [float(reference[f"{side}_inner_normal_N"]) for side in ("left", "right")],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(sliding, [float(reference[f"{side}_sliding_normal_N"]) for side in ("left", "right")],
                                   rtol=0, atol=1e-12)
        positive_pull = trace_row[columns["signed_pull_N"]] > .001
        off_support = trace_row[columns["support_N"]] <= .05
        eligible = (load > .001) & (sliding >= .5 * load) & positive_pull & off_support
        speed = np.divide(weighted_speed, sliding, out=np.zeros(2), where=sliding > 0)
        assert bool(eligible.all()) == (reference["bilateral_sliding"] == "True"), (case["id"], t)
        rows.append((t, *eligible, *speed))
    return rows


def integrate(rows):
    t, *fields = np.asarray(rows).T
    dt = np.diff(t, prepend=t[0])
    if len(rows) < 2 or not np.allclose(dt[1:], .002, rtol=0, atol=5e-5):
        raise ValueError("A contiguous 2 ms release trace is required")
    active = np.array(fields[:2]).T.astype(bool)
    speed = np.array(fields[2:]).T
    interval = active[1:] & active[:-1]
    steps = interval * (speed[1:] + speed[:-1]) * (dt[1:, None] / 2)
    path = np.vstack((np.zeros((1, 2)), np.cumsum(steps, axis=0)))
    summaries = {}
    for name, mask in zip(("left", "right", "bilateral"), (*active.T, active.all(axis=1)), strict=True):
        padded = np.r_[False, mask, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
        lengths = t[ends] - t[starts]
        summaries[name] = dict(active_samples=int(mask.sum()), longest_continuous_s=float(lengths.max(initial=0.)),
                               intervals_at_least_10_ms=int(np.sum(lengths >= .010 - 1e-9)))
    summaries["total_qualified_local_path_by_pad_m"] = path[-1].tolist()
    return path, summaries


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    output = parser.parse_args().output
    data = json.loads(DATA.read_bytes())
    columns = {name: i for i, name in enumerate(data["sample_columns"])}
    summaries, records = {}, []
    for case in data["cases"]:
        if case["contact_source"] is None:
            continue
        rows = observations(case, columns)
        path, summary = integrate(rows)
        summaries[case["id"]] = summary
        for row, distance in zip(rows, path, strict=True):
            records.append((case["id"], *row, *distance))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "contact-paths.csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case", "t_s", "left_qualified", "right_qualified", "left_weighted_speed_m_s",
                         "right_weighted_speed_m_s", "left_accumulated_local_m", "right_accumulated_local_m"))
        writer.writerows(records)
    with (output / "contact-paths.json").open("x") as stream:
        stream.write(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
