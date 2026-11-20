"""Run the jaw-local refinement from the established pre-grasp preparation."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np

from replay_refined_pe import RUN, configure, experiment

RAMP = RUN.with_name("rss2027-stage1-kraft-load-ramp-r1")
sys.path.insert(0, str(RAMP))
from record_load_ramp import main as record_load  # noqa: E402
from physical_parcel_checks import film_payload_geometry  # noqa: E402


def prepare():
    backend = experiment.backend
    cfg = experiment.get_config()
    coarse, _ = backend.parcel_world(cfg)
    backend.initialize_state(coarse, coarse.m, coarse.d, cfg)
    fine_cfg = replace(cfg, geometry=experiment.GEOMETRY, initial_state=None, wall_limit=2200.)
    with patch.object(experiment.physical_parcel, "wrap_generic", experiment.prepared_wrap(experiment.GEOMETRY)):
        fine, _ = backend.parcel_world(fine_cfg)
    backend.initialize_state(fine, fine.m, fine.d, fine_cfg)
    with np.load(experiment.ASSET) as asset:
        edges = asset["old_edges"]
    experiment.copy_joint_state(coarse, fine, edges)
    for parcel in (coarse, fine):
        mujoco.mj_fwdPosition(parcel.m, parcel.d)
    np.testing.assert_allclose(fine.d.flexvert_xpos, experiment.midpoints(coarse.d.flexvert_xpos, edges),
                               atol=1e-12, rtol=0.)
    np.testing.assert_allclose(fine.envelope["masses"].sum(), coarse.envelope["masses"].sum(), atol=1e-12, rtol=0.)
    geometry = film_payload_geometry(fine.m, fine.d, fine.envelope["faces"])
    if geometry["intersecting_triangles"] or geometry["outside_corners"]:
        raise ValueError(f"The refined pre-grasp state fails the original payload geometry: {geometry}")
    case = RUN / "full-action"
    case.mkdir(exist_ok=False)
    state = case / "initial_state.npz"
    backend.save_initial_state(fine, fine.d, fine_cfg, state)
    config = vars(fine_cfg) | {"initial_state": state.relative_to(experiment.ROOT).as_posix()}
    (case / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = dict(coarse=coarse.m.nflexvert, refined=fine.m.nflexvert, original_state=cfg.initial_state,
                  source_mesh=experiment.GEOMETRY, refined_initial_payload_geometry=geometry,
                  initial_reference_mass_kg=float(fine.envelope["masses"].sum()))
    (case / "preparation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    configure()
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    else:
        with patch.object(experiment.physical_parcel, "wrap_generic", experiment.prepared_wrap(experiment.GEOMETRY)):
            record_load()
