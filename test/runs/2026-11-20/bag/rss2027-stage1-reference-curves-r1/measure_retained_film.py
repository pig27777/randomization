"""Measure the reference film retained in five qualified whole-bag suspensions."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from export_curves import ROOT, RUN

sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
from measure_surface_excess import aperture_box, areas  # noqa: E402
from rss2027_yam_capture_audit import aperture_material_areas  # noqa: E402

CASES = ("pe-matched-load", "kraft-clear-slow", "paperboard-rounded-pad",
         "bubble-fast-lift", "bubble-fast-lift-20g")


def measure(case, columns):
    trial = ROOT / case["trial"]
    health = case["measurement"]
    if not health["numerical_ok"] or health["preloading_s"] < .20 or health["load_start_s"] is None:
        raise ValueError(f'{case["id"]} is missing the original 0.20 s preload')
    start = health["load_start_s"]
    force_samples = [dict(zip(columns, reading, strict=True)) for reading in case["samples"]
                     if start - .20 - 1e-9 <= reading[0] < start]
    if not all(row["bilateral_held"] and row["applied_load_N"] == 0. and row["support_N"] <= .05
               and row["tension_N"] > .001 and min(row["left_inner_N"], row["right_inner_N"]) > .001
               for row in force_samples):
        raise ValueError(f'{case["id"]} preload is not continuously qualified')
    times = np.asarray([row["t_s"] for row in force_samples])
    if len(times) < 100:
        raise ValueError(f'{case["id"]} preload is missing force samples')
    np.testing.assert_allclose(np.diff(times), .002, atol=1e-9, rtol=0.)
    trace = json.loads((trial / "trace.json").read_bytes())
    model = mujoco.MjModel.from_binary_path(str(trial / "model.mjb"))
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)
    faces = model.flex_elem.reshape(-1, 3)
    reference = data.flexvert_xpos[faces].copy()
    pads = [model.geom(f"{side}_pad").id for side in ("left", "right")]
    fingers = [int(model.joint(f"{side}_finger").qposadr[0]) for side in ("left", "right")]
    samples = []
    with np.load(trial / "trajectory.npz") as archive:
        np.testing.assert_array_equal(archive["time"], [row["t"] for row in trace])
        for time, qpos, recorded in zip(archive["time"], archive["qpos"], trace, strict=True):
            if not times[0] - 1e-9 <= time <= times[-1] + 1e-9:
                continue
            index = int(np.argmin(abs(times - time)))
            np.testing.assert_allclose(time, times[index], atol=1e-9, rtol=0.)
            forces = force_samples[index]
            geometry = recorded["payload_geometry"]
            if not recorded["finite"] or geometry["outside_corners"] or geometry["intersecting_triangles"]:
                raise ValueError(f'{case["id"]} contains an invalid preload geometry')
            gap = float(qpos[fingers].sum())
            np.testing.assert_allclose((gap, recorded["support_N"]), (forces["actual_gap_m"], forces["support_N"]),
                                       atol=1e-9, rtol=0.)
            data.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, data)
            center, axes, bounds = aperture_box(data, data.geom_xpos[pads], model.geom_bodyid[pads[0]], gap)
            triangles = ((data.flexvert_xpos - center) @ axes)[faces]
            spatial, projection = areas(triangles, bounds)
            reference_area = aperture_material_areas(triangles, reference, bounds)[1]
            width = 2. * bounds[0]
            samples.append(dict(t_s=float(time), actual_gap_m=gap, tension_N=forces["tension_N"],
                                support_N=forces["support_N"], normal_N=[forces["left_inner_N"], forces["right_inner_N"]],
                                reference_length_m=float(reference_area / width),
                                projected_length_m=float(projection / width),
                                spatial_excess_m=float((spatial - projection) / width),
                                reference_excess_m=float((reference_area - projection) / width)))
    if not samples:
        raise ValueError(f'{case["id"]} has no synchronized preload geometry')
    ranges = {key: [min(row[key] for row in samples), max(row[key] for row in samples)]
              for key in ("actual_gap_m", "reference_length_m", "reference_excess_m", "tension_N")}
    return dict(id=case["id"], material=case["material"], payload_mass_kg=case["config"]["payload_mass"],
                original_preloading_s=health["preloading_s"], ranges=ranges, samples=samples)


if __name__ == "__main__":
    source = json.loads((RUN / "stage1_whole_bag_data.json").read_bytes())["load_ramps"]
    cases = {row["id"]: row for row in source["cases"]}
    measurements = [measure(cases[key], source["sample_columns"]) for key in CASES]
    with (RUN / "retained-material-windows.json").open("x") as stream:
        stream.write(json.dumps(measurements, indent=2, allow_nan=False) + "\n")
