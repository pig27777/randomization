"""Compare measured membrane capture and sustained support-free loads."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from record_local_top import RUN, backend
from replay_ridge_window import aperture_box, aperture_material_areas, areas
from summarize_capture_forces import resolve

BASE = RUN / "pe-capture-travel"
ORIGINAL = RUN.with_name("rss2027-stage1-pickup-baseline-cpu-r4") / "recording/trial"


def read(folder, name):
    return json.loads((folder / name).read_bytes())


def film_at_lift(trial, lift):
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    faces = model.flex_elem.reshape(-1, 3)
    reference = data.flexvert_xpos[faces]
    with np.load(trial / "trajectory.npz") as archive:
        index, = np.flatnonzero(archive["time"] == lift["t_s"])
        data.qpos[:] = archive["qpos"][index]
    mujoco.mj_fwdPosition(model, data)
    pads = [model.geom(f"{side}_pad").id for side in ("left", "right")]
    center, axes, bounds = aperture_box(data, data.geom_xpos[pads], model.geom_bodyid[pads[0]], lift["actual_gap_m"])
    material = ((data.flexvert_xpos - center) @ axes)[faces]
    _, projection = areas(material, bounds)
    reference_area = aperture_material_areas(material, reference, bounds)[1]
    width = 2. * bounds[0]
    np.testing.assert_allclose(reference_area / width, lift["aperture_material_length_m"], atol=1e-12, rtol=0.)
    return dict(first_lift_reference_length_m=float(reference_area / width),
                first_lift_reference_excess_m=float((reference_area - projection) / width))


def hold_observation(case):
    records = read(case, "recording/capture-contacts.json")
    hold = [resolve(row) for row in records if row["phase"] == "lift_hold"]
    if not hold:
        return None
    unsupported = [row for row in hold if row["support_N"] <= .05]
    positive = [row for row in unsupported if row["tension_N"] > .001]
    normals = np.asarray([row["standard_inner_normal_N"] for row in positive]).reshape(-1, 2)
    with (case / "capture/capture-audit.csv").open(newline="") as stream:
        retained = [float(row["aperture_material_length_m"]) for row in csv.DictReader(stream)
                    if row["phase"] == "lift_hold"]
    return dict(samples=len(hold), unsupported_samples=len(unsupported), positive_pull_samples=len(positive),
                bilateral_standard_samples=int((normals > .001).all(axis=1).sum()),
                standard_contact_missing_each=(normals <= .001).sum(axis=0).tolist(),
                actual_gap_range_m=[min(row["gap_m"] for row in hold), max(row["gap_m"] for row in hold)],
                reference_membrane_range_m=[min(retained), max(retained)] if retained else None)


def observation(case):
    trial = case / "recording/trial"
    result = read(trial, "result.json")
    trace = read(trial, "trace.json")
    summary = dict(case=case.name, samples=len(trace), outcome=result["outcome"],
                   peak_penetration_mm=result["peaks"]["penetration_mm"], accepted=False)
    if not (case / "capture/capture-summary.json").is_file():
        return trace, summary
    capture = read(case, "capture/capture-summary.json")
    loads = read(case, "force-summary.json")
    first_close = next(row for row in trace if row["phase"] == "close")
    robot = backend.scene(backend.Config(**result["config"]))
    advance = (np.asarray(first_close["tip_mid"]) - trace[0]["tip_mid"]) @ robot.approach
    held = loads["longest_positive_unsupported_bilateral_s"]
    intact = all(not row["payload_geometry"]["outside_corners"] and not row["payload_geometry"]["intersecting_triangles"]
                 for row in trace)
    summary.update(actual_initial_close_advance_m=float(advance), sustained_positive_unsupported_bilateral_s=held,
                   hold_observation=hold_observation(case),
                   intact=intact, accepted=bool(result["outcome"] == "pickup_ok" and held >= .20 - 1e-9
                                                and intact and not result["warnings"] and summary["peak_penetration_mm"] <= 1.))
    lift = capture["first_lift"]
    if lift is not None:
        summary.update(first_lift_gap_m=lift["actual_gap_m"], first_lift_tension_N=lift["tension_N"], **film_at_lift(trial, lift))
    return trace, summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("variants", nargs="+", help="completed variant subdirectories")
    parser.add_argument("--output", type=Path, default=BASE / "comparison.json")
    args = parser.parse_args()
    baseline = BASE / "original-advance"
    trace, reference = observation(baseline)
    with (np.load(baseline / "recording/trial/trajectory.npz") as new,
          np.load(ORIGINAL / "trajectory.npz") as old):
        np.testing.assert_array_equal(new["time"], old["time"])
        np.testing.assert_array_equal(new["qpos"], old["qpos"])
    comparisons = [reference]
    for name in args.variants:
        variant, data = observation(BASE / name)
        prefix = next((i for i, row in enumerate(variant) if row["phase"] == "close"), len(variant))
        ignored = {"side_advance_m", "side_close_advance_m"}
        expected, actual = [[{key: value for key, value in row.items() if key not in ignored} for row in items[:prefix]]
                            for items in (trace, variant)]
        if expected != actual:
            raise ValueError(f"The original and {name} differed before its explicit closure")
        data["identical_approach_samples"] = prefix
        comparisons.append(data)
    with args.output.open("x") as stream:
        stream.write(json.dumps(dict(bitwise_original_geometry=True, trials=comparisons), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
