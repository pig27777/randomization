"""Halve the mesh in the active jaws of a measured whole-bag withdrawal state."""

import argparse
from functools import partial
import hashlib
import json
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined_pe import RUN, configure, experiment
from replay_settled import settled_scene

SLIDE = RUN.with_name("rss2027-stage1-pe-held-load-step-r1")
sys.path.insert(0, str(SLIDE))
from measure_aperture_flow import record_flow, summarize  # noqa: E402
from measure_tangent_flow import estimate_beta  # noqa: E402

KRAFT = RUN.with_name("rss2027-stage1-kraft-held-local-mesh-r1")


def settled_kraft():
    mesh = KRAFT / "prepared_mesh.npz"
    geometry = "../" + mesh.relative_to(experiment.ROOT).as_posix()
    with patch.multiple(experiment, RUN=KRAFT, ASSET=mesh, GEOMETRY=geometry, COMPILED_DT=.00005):
        return settled_scene()


def original_mid_slide(setup, source):
    robot, _, _ = setup()
    with np.load(source / "mid-slide-state.npz") as archive:
        state, spec = archive["integration_state"], int(archive["spec"])
        count = int(archive["matched_prefix_steps"])
    mujoco.mj_setState(robot.sim.m, robot.sim.d, state, spec)
    return robot, count, hashlib.sha256(state.tobytes()).hexdigest()


_ORIGINAL_SCENE = experiment.original_scene


def measure(output, scene, dt, horizon, original, ratio):
    output.mkdir(exist_ok=False)
    direction = np.asarray(original["force_direction"])
    with patch.object(experiment.replay, "held_scene", scene):
        native, values, transitions = record_flow(dt, horizon, original["load_mode"], force_vector=direction,
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
    print(output.name, native["outcome"], native["peak_penetration_mm"], beta, flush=True)
    return report


def replay(case, source, original, setup, baseline, smoke_duration, dt):
    reference = json.loads((source / "comparison.json").read_bytes())
    ratio = reference["common_onset_force_ratio"]
    duration = reference["original_end_s"] - reference["original_start_s"]
    if dt != baseline["native_load"]["dt_s"]:
        baseline = measure(case / "coarse", setup, dt, duration, original, ratio)
    scene = partial(experiment.refined_scene, source=setup)
    for name, horizon in (("smoke", smoke_duration), ("window", duration)):
        refined = measure(case / name, scene, dt, horizon, original, ratio)
        if not refined["admissible"]:
            break
    coarse = baseline["beta_candidate"]
    beta = refined["beta_candidate"]
    admissible = refined["admissible"] and baseline.get("admissible", True)
    has_same_window = bool(coarse and beta and all(abs(beta[key] - coarse[key]) < 1e-8 for key in ("from_s", "to_s")))
    changes = {}
    if has_same_window:
        for field in ("beta_m_per_N_s", "displacement_m"):
            changes[field] = beta[field] / coarse[field] - 1.
    summary = dict(completed_stage=name, dt_s=dt, coarse=coarse, refined=beta, last_run_admissible=admissible,
                   relative_change=changes, equal_fitted_window=has_same_window,
                   passed=admissible and has_same_window and all(abs(value) <= .10 for value in changes.values()))
    (case / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--material", choices=("pe", "kraft"), default="pe")
    parser.add_argument("--step-us", type=float, choices=(12.5, 25.))
    args = parser.parse_args()
    if args.material == "kraft":
        case, source = KRAFT / "mid-slide-mesh", KRAFT / "local-tangent"
        original = json.loads((KRAFT / "settled-recording/dt-50us.json").read_bytes())
        physical_scene, smoke = settled_kraft, .008
    else:
        configure()
        case, source = RUN / "mid-slide-mesh", SLIDE / "local-tangent-exact"
        original = json.loads((SLIDE / "tangent-prefix/dt-25us/summary.json").read_bytes())["native_load"]
        physical_scene, smoke = _ORIGINAL_SCENE, .020
    baseline = json.loads((source / "dt-25us.json").read_bytes())
    dt = baseline["native_load"]["dt_s"]
    if args.step_us is not None:
        dt = args.step_us / 1e6
        tag = f"{args.step_us:g}".replace(".", "p")
        case = case.with_name(f"{case.name}-{tag}us")
    setup = partial(original_mid_slide, physical_scene, source)
    mesh = case / "prepared_mesh.npz"
    geometry = "../" + mesh.relative_to(experiment.ROOT).as_posix()
    with patch.multiple(experiment, RUN=case, ASSET=mesh, GEOMETRY=geometry,
                        COMPILED_DT=dt):
        if args.prepare:
            case.mkdir(exist_ok=False)
            scene = partial(experiment.refined_scene, source=setup)
            with patch.object(experiment, "original_scene", setup), patch.object(experiment, "refined_scene", scene):
                experiment.prepare()
        else:
            replay(case, source, original, setup, baseline, smoke, dt)


if __name__ == "__main__":
    main()
