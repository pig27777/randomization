"""Pull an approved paperboard suspension along its native jaw tangent."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
SOURCE = RUN / "paperboard-rounded-pad/recording"
CASE = RUN / "paperboard-tangent"
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-pe-held-load-step-r1")))
from measure_aperture_flow import record_flow, replay, summarize  # noqa: E402
from measure_tangent_flow import estimate_beta  # noqa: E402
from replay_local_tangent import replay_window  # noqa: E402


def setup():
    replay.SOURCE = SOURCE
    replay.CHECKPOINT = CASE / "held-state.npz"


def scene():
    backend = replay.backend
    assemble = backend.scene
    original_masks = json.loads((SOURCE / "disabled-legacy-pad-edges.json").read_bytes())

    def same_contact_surfaces(config):
        robot = assemble(config)
        for name, masks in original_masks.items():
            g = robot.sim.m.geom(name).id
            np.testing.assert_equal([robot.sim.m.geom_contype[g], robot.sim.m.geom_conaffinity[g]], masks)
            robot.sim.m.geom_contype[g] = robot.sim.m.geom_conaffinity[g] = 0
        return robot

    with patch.object(backend, "scene", same_contact_surfaces):
        return _HELD_SCENE()


_HELD_SCENE = replay.held_scene


def prepare():
    evidence = json.loads((SOURCE / "load-check.json").read_bytes())
    if not evidence["numerical_ok"] or evidence["preloading_s"] < .2 or evidence["load_start_s"] is None:
        raise ValueError("Paperboard source has no qualified original preload")
    CASE.mkdir(exist_ok=False)
    replay.extract_state()
    robot, index, digest = scene()
    sim = robot.sim
    backend = replay.backend
    backend.refresh(sim)
    row = json.loads((SOURCE / "trial/trace.json").read_bytes())[index]
    force = backend.measure(robot)
    np.testing.assert_allclose(force["force_on_fingers_N"], row["force_on_fingers_N"], atol=1e-8, rtol=0.)
    normal, _ = replay.inner_loads(replay.release_contacts(robot))
    failure, health = backend.check_state(sim)
    if (failure or health["payload_geometry"]["intersecting_triangles"] or force["support_N"]
            or min(normal) <= .001 or force["penetration_mm"] > 1.):
        raise ValueError(f"Paperboard checkpoint failed its suspension check at index {index}")
    proof = dict(source_index=index, time_s=sim.d.time, state_sha256=digest, native_inner_N=normal.tolist(),
                 support_N=force["support_N"], initial_penetration_mm=force["penetration_mm"])
    (CASE / "preparation.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")


def execute():
    for name, duration in (("smoke", .020), ("recording", 2.40)):
        output = CASE / name
        output.mkdir(exist_ok=False)
        with patch.object(replay, "held_scene", scene):
            result, rows, transitions = record_flow(.0001, duration, "jaw-tangent")
        flow = summarize(result, rows, transitions)
        beta = estimate_beta(result, flow)
        report = dict(native_load=result, material_flow=flow, beta_candidate=beta)
        np.savez_compressed(output / "native-load.npz", columns=np.asarray(replay.FIELDS), rows=rows)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, result["outcome"], result["peak_penetration_mm"], result["longest_bilateral_slide_s"], beta, flush=True)
        if name == "smoke" and (result["outcome"] != "no_loss_before_limit"
                                or result["supported_force_peak_N"] > .05 or min(rows[-1, 5:7]) <= .001):
            break


def compare_steps():
    full = CASE / "recording"
    reference = json.loads((full / "summary.json").read_bytes())
    if reference["beta_candidate"] is None:
        raise ValueError("Paperboard comparison requires the archived positive withdrawal window")
    ratio = reference["material_flow"]["candidate_force_ratio"]
    with patch.object(replay, "held_scene", scene):
        replay_window(CASE / "half-step", reference["native_load"], full / "native-load.npz",
                      1.590, 1.604, ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--prepare", action="store_true")
    command.add_argument("--compare-steps", action="store_true")
    args = parser.parse_args()
    setup()
    if args.prepare:
        prepare()
    elif args.compare_steps:
        compare_steps()
    else:
        execute()
