"""Halve the mesh in the active jaws of the measured PE withdrawal state."""

import argparse
from functools import partial
import hashlib
import json
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined_pe import RUN, configure, experiment

SLIDE = RUN.with_name("rss2027-stage1-pe-held-load-step-r1")
sys.path.insert(0, str(SLIDE))
from measure_aperture_flow import record_flow, summarize  # noqa: E402
from measure_tangent_flow import estimate_beta  # noqa: E402

SOURCE = SLIDE / "local-tangent-exact"
START_REF = SLIDE / "tangent-prefix/dt-25us/summary.json"
BASELINE = SOURCE / "dt-25us.json"
CASE = RUN / "mid-slide-mesh"


def set_case():
    configure()
    native = json.loads(START_REF.read_bytes())["native_load"]
    experiment.RUN = CASE
    experiment.ASSET = CASE / "prepared_mesh.npz"
    experiment.GEOMETRY = "../" + experiment.ASSET.relative_to(experiment.ROOT).as_posix()
    experiment.COMPILED_DT = native["dt_s"]
    return native


def original_mid_slide():
    robot, _, _ = _ORIGINAL_SCENE()
    with np.load(SOURCE / "mid-slide-state.npz") as archive:
        state, spec = archive["integration_state"], int(archive["spec"])
        count = int(archive["matched_prefix_steps"])
    mujoco.mj_setState(robot.sim.m, robot.sim.d, state, spec)
    return robot, count, hashlib.sha256(state.tobytes()).hexdigest()


_ORIGINAL_SCENE = experiment.original_scene


def replay(dt, original):
    direction = np.asarray(original["force_direction"])
    baseline = json.loads(BASELINE.read_bytes())
    reference = json.loads((SOURCE / "comparison.json").read_bytes())
    ratio = reference["common_onset_force_ratio"]
    duration = reference["original_end_s"] - reference["original_start_s"]
    scene = partial(experiment.refined_scene, source=original_mid_slide)
    for name, horizon in (("smoke", .020), ("window", duration)):
        output = CASE / name
        output.mkdir(exist_ok=False)
        with patch.object(experiment.replay, "held_scene", scene):
            native, values, transitions = record_flow(dt, horizon, "jaw-tangent", force_vector=direction,
                                                        load_origin_s=original["source_time_s"])
        flow = summarize(native, values, transitions, force_ratio=ratio)
        beta = estimate_beta(native, flow)
        geometry = native["end_health"]["payload_geometry"]
        admissible = (native["outcome"] == "no_loss_before_limit" and native["peak_penetration_mm"] <= 1.
                      and native["end_health"]["warnings"] == 0 and native["supported_force_peak_N"] <= .05
                      and geometry["outside_corners"] == 0 and geometry["intersecting_triangles"] == 0)
        report = dict(native_load=native, material_flow=flow, beta_candidate=beta, admissible=admissible)
        np.savez_compressed(output / "native-load.npz", columns=np.asarray(experiment.replay.FIELDS), rows=values)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, native["outcome"], native["peak_penetration_mm"], beta, flush=True)
        if not admissible:
            break
    coarse = baseline["beta_candidate"]
    has_same_window = beta is not None and abs(beta["from_s"] - coarse["from_s"]) < 1e-8 and (
        abs(beta["to_s"] - coarse["to_s"]) < 1e-8)
    changes = {}
    if has_same_window:
        for field in ("beta_m_per_N_s", "displacement_m"):
            changes[field] = beta[field] / coarse[field] - 1.
    summary = dict(completed_stage=name, coarse=coarse, refined=beta, last_run_admissible=admissible,
                   relative_change=changes, equal_fitted_window=has_same_window,
                   passed=admissible and has_same_window and all(abs(value) <= .10 for value in changes.values()))
    (CASE / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    native = set_case()
    if args.prepare:
        CASE.mkdir(exist_ok=False)
        scene = partial(experiment.refined_scene, source=original_mid_slide)
        with patch.object(experiment, "original_scene", original_mid_slide), patch.object(experiment, "refined_scene", scene):
            experiment.prepare()
    else:
        replay(native["dt_s"], native)


if __name__ == "__main__":
    main()
