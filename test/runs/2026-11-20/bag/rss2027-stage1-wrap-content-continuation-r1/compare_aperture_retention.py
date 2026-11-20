"""Track the membrane from the first lift in a fixed-width moving YAM aperture."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from screen_side_directions import backend, physical_parcel, prepared_wrap

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-material-feed-r1")))
from replay import aperture_box, aperture_material_areas, shared_material_area  # noqa: E402


def replay_lift(case):
    trial = case.resolve() / "recording/trial"
    result = json.loads((trial / "result.json").read_bytes())
    config = backend.Config(**result["config"])
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(config.geometry)):
        model = backend.scene(config).sim.m
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    faces = model.flex_elem.reshape(-1, 3)
    reference = data.flexvert_xpos[faces]
    pads = [model.geom(f"{side}_pad").id for side in ("left", "right")]
    left_pad = model.geom_bodyid[pads[0]]
    trace = json.loads((trial / "trace.json").read_bytes())
    first = next(i for i, row in enumerate(trace) if row["phase"] == "lift")
    if trace[first]["gap"] <= 0.:
        raise ValueError(f"The first lift has no positive aperture in {case}")
    width = .020
    rows = []
    with np.load(trial / "trajectory.npz") as trajectory:
        np.testing.assert_array_equal(trajectory["time"], [row["t"] for row in trace])
        qpos = trajectory["qpos"]
        for i in range(first, len(trace)):
            row = trace[i]
            data.qpos[:] = qpos[i]
            mujoco.mj_fwdPosition(model, data)
            center, basis, actual = aperture_box(data, data.geom_xpos[pads], left_pad, row["gap"])
            points = ((data.flexvert_xpos - center) @ basis)[faces]
            if i == first:
                fixed = actual.copy()
                initial = points.copy()
                initial_area = float(aperture_material_areas(initial, reference, fixed)[1])
                if initial_area <= 0.:
                    raise ValueError(f"The first lift contains no membrane in {case}")
            area = float(aperture_material_areas(points, reference, actual)[1]) if actual[1] > 0. else 0.
            remaining = shared_material_area(initial, points, reference, fixed, fixed)
            if not -1e-12 <= remaining <= initial_area + 1e-12:
                raise ValueError(f"Aperture retention is inconsistent at {row['t']} in {case}")
            window = float(aperture_material_areas(points, reference, fixed)[1])
            rows.append(dict(t_s=row["t"], actual_gap_m=row["gap"], actual_reference_length_m=area / width,
                             fixed_reference_length_m=window / width, initial_material_retained=remaining / initial_area,
                             support_N=row["support_N"], fingertip_contact_N=max(row["contact_magnitude_each_N"])))
    start = rows[0]
    np.testing.assert_allclose(start["initial_material_retained"], 1., atol=1e-10, rtol=0.)
    np.testing.assert_allclose(start["actual_reference_length_m"], start["fixed_reference_length_m"], atol=1e-12, rtol=0.)
    first_low = next(row for row in rows if row["fingertip_contact_N"] <= .05 and row["t_s"] > start["t_s"])
    return dict(trial=trial.relative_to(RUN).as_posix(), first_lift=start, first_low_fingertip=first_low,
                final=rows[-1], samples=rows)


def main():
    side = RUN / "equilibrated-entry/sample-061"
    cases = (side / "side-film-open-route", side / "side-film-outward-lift",
             RUN / "fold-aligned-90-energy/sample-066/side-open-route")
    result = {case.name: replay_lift(case) for case in cases}
    with (RUN / "aperture-retention.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
