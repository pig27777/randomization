"""Measure signed whole-bag membrane flow during the archived PE slip window."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from replay_pe import RUN, replay
from rss2027_yam_capture_audit import aperture_box, aperture_material_areas

FEED_SOURCE = RUN.with_name("rss2027-stage1-material-feed-r1") / "replay.py"
spec = importlib.util.spec_from_file_location("aperture_material_feed", FEED_SOURCE)
feed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feed)


def record_flow(dt, duration, direction, begin_s=0.):
    stride = round(.002 / dt)
    previous = None
    transitions = []

    def sample(robot, index, values):
        nonlocal previous
        time = values[0]
        if index % stride or not begin_s <= time <= duration + dt / 2:
            return
        row = dict(zip(replay.FIELDS, values, strict=True), t=time, phase="load", step=index)
        sim = robot.sim
        data, model = sim.d, sim.m
        faces = sim.parcel.envelope["faces"]
        reference = sim.parcel.envelope["reference"][faces]
        centers = data.geom_xpos[robot.pad_geom]
        origin, basis, extents = aperture_box(data, centers, model.geom_bodyid[robot.pad_geom[0]], row["actual_gap_m"])
        local = ((data.flexvert_xpos - origin) @ basis)[faces]
        area = aperture_material_areas(local, reference, extents)[1] if extents[1] > 0. else 0.
        current = local, extents, area, row
        if previous is not None:
            flow = feed.material_exchange(previous, current, reference)
            transitions.append(dict(flow, from_step=previous[-1]["step"], to_step=index,
                                    tension_N=row["tension_N"], normal_N=row["left_inner_N"] + row["right_inner_N"]))
        previous = current

    return (*replay.run(dt, duration + dt, direction, readout=sample), transitions)


def summarize(result, rows, transitions):
    net = sum(row["entered_m"] - row["left_m"] for row in transitions)
    if transitions:
        np.testing.assert_allclose(net, transitions[-1]["final_m"] - transitions[0]["initial_m"], atol=1e-12, rtol=0.)
    start = result["sliding_onset"]
    ratio = None
    qualifies = np.zeros(len(rows), dtype=bool)
    if start is not None:
        ratio = start["tension_N"] / (start["left_inner_N"] + start["right_inner_N"])
        positive = rows[:, 3] - ratio * rows[:, 5:7].sum(axis=1)
        positive[np.argmin(abs(rows[:, 0] - start["time_s"]))] = 0.
        qualifies = (rows[:, 9] > 0.) & (positive > 0.) & (rows[:, 0] > start["time_s"])
    starts_ends = np.flatnonzero(np.diff(np.r_[False, qualifies, False])).reshape(-1, 2)
    longest = max((float(rows[b - 1, 0] - rows[a, 0]) for a, b in starts_ends), default=0.)
    for row in transitions:
        a, b = row["from_step"], row["to_step"]
        row["positive_overload_with_bilateral_slip"] = bool(qualifies[a:b + 1].all())
        row["native_bilateral_sliding_through_interval"] = bool(rows[a:b + 1, 9].all())
        row["net_material_withdrawal_m"] = row["left_m"] - row["entered_m"]
        if row["positive_overload_with_bilateral_slip"]:
            row["overload_integral_Ns"] = float(np.trapezoid(positive[a:b + 1], rows[a:b + 1, 0]))
    slip = [row for row in transitions if row["native_bilateral_sliding_through_interval"]]
    return dict(dt_s=result["dt_s"], compared_native_steps=len(rows), source_sha256=result["source_state_sha256"],
                onset=start, candidate_force_ratio=ratio, longest_bilateral_positive_overload_s=longest,
                sample_period_s=.002, total_signed_withdrawal_m=float(-net),
                sliding_interval_count=len(slip), signed_withdrawal_during_bilateral_slip_m=float(
                    sum(row["net_material_withdrawal_m"] for row in slip)), transitions=transitions)


def measure(dt, original):
    result, rows, transitions = record_flow(dt, .56, "vertical", begin_s=.46)
    with np.load(original.with_suffix(".npz")) as archive:
        archived = archive["rows"]
    np.testing.assert_array_equal(rows, archived[:len(rows)])
    assert result["outcome"] == "no_loss_before_limit" and result["end_health"]["payload_geometry"]["outside_corners"] == 0
    return summarize(result, rows, transitions)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay.SOURCE = RUN.with_name("rss2027-stage1-material-load-ramps-r1") / "pe/recording"
    replay.CHECKPOINT = RUN / "held-state.npz"
    args.output.mkdir(exist_ok=False)
    for name, dt in (("25", .000025), ("12p5", .0000125)):
        result = measure(dt, RUN / f"recording/dt-{name}us.json")
        (args.output / f"dt-{name}us.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print({key: value for key, value in result.items() if key != "transitions"}, flush=True)


if __name__ == "__main__":
    main()
