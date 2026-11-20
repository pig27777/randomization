"""Prepare the high-fill parcel gradually, then restore the original model."""

import argparse
import json
from pathlib import Path
import platform
from unittest.mock import patch

import mujoco
import numpy as np

from diagnose_initial_load import RUN, backend, physical_parcel, prepared_wrap
from physical_parcel_checks import film_payload_geometry


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = RUN / "strain-constrained-entry/sample-090/smoke.json"
    params = json.loads(source.read_bytes())
    cfg = backend.Config(**params)
    assert cfg.support_height == 0. and cfg.initial_state is None
    with patch.object(physical_parcel, "wrap_generic", prepared_wrap(cfg.geometry)):
        parcel, _ = backend.parcel_world(cfg)
    backend.initialize_state(parcel, parcel.m, parcel.d, cfg)
    model, data = parcel.m, parcel.d
    faces = parcel.envelope["faces"]
    stiffness, gravity = model.flex_stiffness.copy(), model.opt.gravity.copy()
    bending, damping = parcel.hinge_stiffness.copy(), parcel.hinge_damping.copy()
    initial = film_payload_geometry(model, data, faces)
    assert initial["outside_corners"] == 0 and initial["intersecting_triangles"] == 0
    peak = 0.
    outcome = "prepared"
    rows = []
    for step in range(round(.30 / cfg.dt)):
        time = float(data.time)
        u = min(1., time / .12)
        fraction = u * u * (3. - 2. * u)
        amplitude = .01 + .99 * fraction
        model.flex_stiffness[:] = amplitude * stiffness
        parcel.hinge_stiffness[:] = amplitude * bending
        parcel.hinge_damping[:] = np.sqrt(amplitude) * damping
        model.opt.gravity[:] = fraction * gravity
        parcel.step()
        contacts = parcel.contacts()
        peak = max(peak, contacts["penetration_m"] * 1000.)
        if peak > 1. or data.warning.number.any():
            outcome = "native_penetration_or_warning"
            break
        if step % round(.002 / cfg.dt) == 0:
            geometry = film_payload_geometry(model, data, faces)
            rows.append(dict(t_s=time, material_fraction=amplitude, peak_penetration_mm=peak,
                             peak_node_speed_m_s=float(np.linalg.norm(data.qvel[parcel.film_dofs], axis=1).max()),
                             payload_geometry=geometry))
            if geometry["outside_corners"] or geometry["sampled_depth_m"] > .001:
                outcome = "payload_geometry"
                break
    model.flex_stiffness[:] = stiffness
    parcel.hinge_stiffness[:] = bending
    parcel.hinge_damping[:] = damping
    model.opt.gravity[:] = gravity
    parcel.forward()
    final = film_payload_geometry(model, data, faces)
    final_depth = parcel.contacts()["penetration_m"] * 1000.
    if outcome == "prepared" and (final_depth > 1. or final["outside_corners"] or final["intersecting_triangles"]):
        outcome = "terminal_geometry"
    archive = args.output / "sample-090"
    archive.mkdir(parents=True, exist_ok=False)
    root = RUN.parents[4]
    if outcome == "prepared":
        state = archive / "initial_state.npz"
        backend.save_initial_state(parcel, data, cfg, state)
        params["initial_state"] = state.resolve().relative_to(root).as_posix()
        for name, seconds in (("smoke", .02), ("settle", .30)):
            (archive / f"{name}.json").write_text(json.dumps(params | {"smoke_seconds": seconds}, indent=2) + "\n")
    report = dict(outcome=outcome, hostname=platform.node(), mujoco=mujoco.__version__, dt_s=cfg.dt,
                  restored_after_s=.12, simulated_s=float(data.time), peak_penetration_mm=peak,
                  final_penetration_mm=final_depth, final_payload_geometry=final, samples=rows)
    (archive / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(outcome, report["simulated_s"], peak, final_depth, flush=True)


if __name__ == "__main__":
    main()
