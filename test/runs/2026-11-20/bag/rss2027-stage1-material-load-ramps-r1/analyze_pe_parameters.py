"""Check which parameter windows are identifiable from the completed PE ramp."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
RECORDING = RUN / "pe/recording"
FIELDS = ("time_s", "applied_N", "tension_N", "support_N", "separation_m", "left_normal_N", "right_normal_N",
          "left_sliding", "right_sliding", "left_speed_m_s", "right_speed_m_s")


def intervals(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def local_contact_motion(contacts):
    normal, slides, weighted_velocity = np.zeros((3, 2))
    for contact in contacts:
        load, *tangent = contact["force_contact_N"]
        if load <= 0. or not contact["inner"]:
            continue
        pad = contact["pad"]
        normal[pad] += load
        friction = np.asarray(contact["friction"])
        speed = np.linalg.norm(contact["tangent_constraint_velocity_m_s"])
        if (friction > 0.).all() and speed > .002 and np.linalg.norm(tangent / friction) >= .95 * load:
            slides[pad] += load
            weighted_velocity[pad] += load * speed
    speed = np.divide(weighted_velocity, slides, out=np.zeros(2), where=slides > 0.)
    return normal, slides, speed


def measured_rows():
    frames = json.loads((RECORDING / "load-contacts.json").read_bytes())
    with (RECORDING / "load-curve.csv").open(newline="") as stream:
        recorded = list(csv.DictReader(stream))
    rows = []
    for frame, row in zip(frames, recorded, strict=True):
        t = frame["t_s"]
        assert t == float(row["t_s"])
        normal, slides, speed = local_contact_motion(frame["contacts"])
        for side, label in enumerate(("left", "right")):
            np.testing.assert_allclose([normal[side], slides[side]],
                                       [float(row[f"{label}_inner_N"]), float(row[f"{label}_sliding_N"])],
                                       rtol=0., atol=1e-12)
        tension, support, force = map(float, (row["tension_N"], row["support_N"], row["applied_load_N"]))
        supported = normal.min() > .001 and tension > .001 and support <= .05
        moving = (slides >= .5 * normal) & supported & (force > 0.)
        assert bool(moving.all()) == (row["native_bilateral_sliding"] == "True")
        separation = float(row["tip_z_m"]) - float(row["payload_com_z_m"])
        rows.append((t, force, tension, support, separation, *normal, *moving, *speed))
    return np.asarray(rows)


def fit_windows(table, elastic, displacement_radius):
    eligible = []
    for a, b in intervals(elastic):
        window = table[a:b]
        times, force, displacement = window[:, 0], window[:, 2], window[:, 4]
        if times[-1] - times[0] < .02 - 1e-9 or np.ptp(displacement) < displacement_radius:
            continue
        coefficient, offset = np.polyfit(displacement - displacement[0], force, 1)
        residual = force - (offset + coefficient * (displacement - displacement[0]))
        variance = np.square(force - force.mean()).sum()
        score = float(1. - (residual @ residual) / variance) if variance > 0. else None
        eligible.append(dict(start_s=float(times[0]), end_s=float(times[-1]), samples=len(times),
                             displacement_span_m=float(np.ptp(displacement)), stiffness_N_m=float(coefficient),
                             r_squared=score, rms_residual_N=float(np.sqrt(np.mean(residual**2)))))
    return sorted(eligible, key=lambda row: row["end_s"] - row["start_s"], reverse=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=RUN / "pe/parameter-observations.json")
    args = parser.parse_args()
    evidence = json.loads((RECORDING / "load-check.json").read_bytes())
    if not evidence["numerical_ok"]:
        raise ValueError("PE source did not pass its numerical gate")
    table = measured_rows()
    time = table[:, 0]
    np.testing.assert_allclose(np.diff(time), .002, atol=1e-9, rtol=0.)
    slip = table[:, 7:9].astype(bool)
    combined = slip.all(axis=1)
    events = [(a, b) for a, b in intervals(combined) if time[b - 1] - time[a] >= .010 - 1e-9]
    first, end = events[0]
    np.testing.assert_equal(time[first], evidence["first_sustained_bilateral_sliding"]["t_s"])
    total_normal = table[:, 5:7].sum(axis=1)
    ratio = table[first, 2] / total_normal[first]
    # Make the critical sample exactly zero despite rounding in the ratio.
    overload = table[:, 2] - ratio * total_normal
    overload[first] = 0.
    active_overload = combined & (overload > 0.) & (time >= time[first])
    duration = max((time[b - 1] - time[a] for a, b in intervals(active_overload)), default=0.)
    motion = table[first:end, 9:]
    distance = np.sum((motion[1:] + motion[:-1]) * (np.diff(time[first:end])[:, None] / 2.), axis=0)
    before_slip = ((time < time[first]) & (table[:, 1] > 0.) & (table[:, 2] > .001) & (table[:, 3] <= .05)
                   & (table[:, 5:7].min(axis=1) > .001) & ~slip.any(axis=1))
    summary = dict(
        samples=len(table), onset=dict(time_s=float(time[first]), tension_N=float(table[first, 2]),
                                      both_inner_N=float(total_normal[first]), effective_force_ratio=float(ratio)),
        first_bilateral_sampled_duration_s=float(time[end - 1] - time[first]),
        first_bilateral_local_path_by_pad_m=distance.tolist(),
        longest_positive_overload_s=float(duration),
        slip_rate_window_qualified=bool(duration >= .010 - 1e-9),
        pre_slip_stiffness_candidates=fit_windows(table, before_slip, .00025),
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow((*FIELDS, "overload_N"))
        writer.writerows(np.column_stack((table, overload)))


if __name__ == "__main__":
    main()
