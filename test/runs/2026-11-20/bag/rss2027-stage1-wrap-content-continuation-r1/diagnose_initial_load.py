"""Resolve the elastic and contact forces during the high-fill entry transient."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
from refine_parcel import backend, prepared_wrap  # noqa: E402
from bag_sim import physical_parcel  # noqa: E402
from record_physical_parcel import penetration_peak  # noqa: E402


def maximum_node_norm(values):
    return float(np.linalg.norm(values, axis=1).max())


def run_probe(cfg, damping):
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        robot = backend.scene(replace(cfg, rayleigh=damping))
    sim = robot.sim
    model, data = sim.m, sim.d
    dofs = sim.film_dofs
    _, initial = backend.check_state(sim)
    samples = []
    outcome = "entry_complete"
    for step in range(round(.02 / cfg.dt)):
        t = float(data.time)
        robot.command(robot.initial_target, cfg.grasp_opening_m)
        mujoco.mj_step1(model, data)
        membrane = data.qfrc_passive[dofs].copy()
        sim.parcel.apply_internal(model, data, dofs)
        bending = data.qfrc_passive[dofs] - membrane
        mujoco.mj_step2(model, data)
        measured = backend.measure(robot)
        sample = dict(t_s=t, penetration_mm=measured["penetration_mm"], support_N=measured["support_N"],
                      max_node_stretch_passive_N=maximum_node_norm(membrane),
                      max_node_hinge_passive_N=maximum_node_norm(bending),
                      max_node_contact_N=maximum_node_norm(data.qfrc_constraint[dofs]),
                      max_node_acceleration_m_s2=maximum_node_norm(data.qacc[dofs]),
                      max_node_speed_m_s=maximum_node_norm(data.qvel[dofs]))
        samples.append(sample)
        if sample["penetration_mm"] > 1. or data.warning.number.any():
            outcome = "native_penetration_or_warning"
            if sample["penetration_mm"] > 1.:
                sample["contact"] = penetration_peak(robot, "settle")
            break
        if step % round(.0005 / cfg.dt) == 0:
            failure, _ = backend.check_state(sim)
            if failure:
                outcome = failure
                break
    backend.refresh(sim)
    error, final = backend.check_state(sim)
    if outcome == "entry_complete" and error:
        outcome = error
    return dict(outcome=outcome, rayleigh_s=damping, integration_time_s=float(data.time),
                initial_payload_geometry=initial["payload_geometry"], final_payload_geometry=final["payload_geometry"],
                peak_penetration_mm=max(row["penetration_mm"] for row in samples), samples=samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    for case in sorted((RUN / "randomized-entry").glob("sample-???")):
        cfg = backend.Config(**json.loads((case / "smoke.json").read_bytes()))
        results = [run_probe(cfg, damping) for damping in (cfg.rayleigh, .0004, .001)]
        with (args.output / f"{case.name}.json").open("x") as stream:
            stream.write(json.dumps(results, allow_nan=False, separators=(",", ":")) + "\n")
        print(case.name, [{key: result[key] for key in ("outcome", "rayleigh_s", "integration_time_s",
                                                        "peak_penetration_mm")} for result in results], flush=True)


if __name__ == "__main__":
    main()
