"""Pull a verified paperboard or bubble-film suspension along its native jaw tangent."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-pe-held-load-step-r1")))
from measure_aperture_flow import record_flow, replay, summarize  # noqa: E402
from measure_tangent_flow import estimate_beta  # noqa: E402
from replay_local_tangent import replay_window  # noqa: E402

MATERIAL_SOURCES = {"paperboard": "paperboard-rounded-pad", "bubble": "bubble-fast-lift",
                    "bubble-20g": "bubble-20g-prepared"}


def setup(material):
    replay.SOURCE = RUN / MATERIAL_SOURCES[material] / "recording"
    case = RUN / f"{material}-tangent"
    replay.CHECKPOINT = case / "held-state.npz"
    return case


def scene():
    if replay.SOURCE.parent.name != MATERIAL_SOURCES["paperboard"]:
        return _HELD_SCENE()
    backend = replay.backend
    assemble = backend.scene
    masks = replay.SOURCE / "disabled-legacy-pad-edges.json"
    original_masks = json.loads(masks.read_bytes())

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


def prepare(case):
    evidence = json.loads((replay.SOURCE / "load-check.json").read_bytes())
    if not evidence["numerical_ok"] or evidence["preloading_s"] < .2 or evidence["load_start_s"] is None:
        raise ValueError(f"{replay.SOURCE} has no qualified original preload")
    case.mkdir(exist_ok=False)
    replay.extract_state()
    robot, index, digest = scene()
    sim = robot.sim
    backend = replay.backend
    backend.refresh(sim)
    row = json.loads((replay.SOURCE / "trial/trace.json").read_bytes())[index]
    force = backend.measure(robot)
    np.testing.assert_allclose(force["force_on_fingers_N"], row["force_on_fingers_N"], atol=1e-8, rtol=0.)
    normal, _ = replay.inner_loads(replay.release_contacts(robot))
    failure, health = backend.check_state(sim)
    if (failure or health["payload_geometry"]["intersecting_triangles"] or force["support_N"]
            or min(normal) <= .001 or force["penetration_mm"] > 1.):
        raise ValueError(f"{replay.SOURCE} failed its suspension check at index {index}")
    proof = dict(source_index=index, time_s=sim.d.time, timestep_s=sim.m.opt.timestep,
                 state_sha256=digest, native_inner_N=normal.tolist(),
                 support_N=force["support_N"], initial_penetration_mm=force["penetration_mm"])
    (case / "preparation.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")


def execute(case, *, half_step=False, alternate_rate=None):
    robot, _, _ = scene()
    dt = float(robot.sim.m.opt.timestep)
    direction, rate = None, replay.RATE
    if half_step or alternate_rate is not None:
        native = json.loads((case / "recording/summary.json").read_bytes())["native_load"]
        np.testing.assert_equal(dt, native["dt_s"])
        direction = np.asarray(native["force_direction"])
        if half_step:
            dt /= 2
            folder = "half-step"
        else:
            rate, folder = alternate_rate
        case = case / folder
        case.mkdir(exist_ok=False)
    for name, duration in (("smoke", .020), ("recording", 12. / rate)):
        output = case / name
        output.mkdir(exist_ok=False)
        with patch.multiple(replay, held_scene=scene, RATE=rate):
            result, rows, transitions = record_flow(dt, duration, "jaw-tangent", force_vector=direction)
        flow = summarize(result, rows, transitions)
        beta = estimate_beta(result, flow)
        report = dict(native_load=result, material_flow=flow, beta_candidate=beta)
        np.savez_compressed(output / "native-load.npz", columns=np.asarray(replay.FIELDS), rows=rows)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(name, result["outcome"], result["peak_penetration_mm"], result["longest_bilateral_slide_s"], beta, flush=True)
        if name == "smoke" and (result["outcome"] != "no_loss_before_limit"
                                or result["supported_force_peak_N"] > .05
                                or np.min(rows[:, 5:7]) <= .001):
            break


def compare_steps(case):
    full = case / "recording"
    reference = json.loads((full / "summary.json").read_bytes())
    if reference["beta_candidate"] is None:
        raise ValueError("Paperboard comparison requires the archived positive withdrawal window")
    ratio = reference["material_flow"]["candidate_force_ratio"]
    with patch.object(replay, "held_scene", scene):
        replay_window(case / "half-step", reference["native_load"], full / "native-load.npz",
                      1.590, 1.604, ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--material", choices=tuple(MATERIAL_SOURCES), default="paperboard")
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--prepare", action="store_true")
    command.add_argument("--compare-steps", action="store_true")
    command.add_argument("--half-step", action="store_true")
    command.add_argument("--fast-ramp", action="store_true")
    command.add_argument("--slow-ramp", action="store_true")
    args = parser.parse_args()
    if args.compare_steps and args.material != "paperboard":
        parser.error("The archived comparison window is for paperboard")
    if (args.half_step or args.fast_ramp) and args.material != "bubble":
        parser.error("The half-step and fast-ramp whole-history controls are for bubble film")
    if args.slow_ramp and args.material != "paperboard":
        parser.error("The 2.5 N/s control uses the archived paperboard source")
    case = setup(args.material)
    if args.prepare:
        prepare(case)
    elif args.compare_steps:
        compare_steps(case)
    else:
        alternate_rate = (20., "fast-ramp") if args.fast_ramp else (2.5, "slow-ramp") if args.slow_ramp else None
        execute(case, half_step=args.half_step, alternate_rate=alternate_rate)
