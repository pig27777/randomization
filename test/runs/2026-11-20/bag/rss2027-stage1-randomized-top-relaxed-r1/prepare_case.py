"""Prepare the same edge-loaded randomized bag with its enclosed, relaxed pose."""

import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
PROJECTION = RUN.with_name("rss2027-stage1-randomized-volume-projection-r1")
ORIGINAL = RUN.with_name("rss2027-stage1-randomized-top-r1") / "sample-004"
sys.path[:0] = [str(PROJECTION), str(ROOT / "test/bag_model")]
from project_volume import RADIUS, measurements, project  # noqa: E402
from bag_sim.legacy.contents import normalize_contents  # noqa: E402
from bag_sim.wrap import panel_pairs  # noqa: E402
from physical_parcel_yam import Config, parcel_world, save_initial_state  # noqa: E402


def main():
    inputs = json.loads((ORIGINAL / "config.json").read_bytes())
    cfg = Config(**inputs)
    parcel, _ = parcel_world(cfg)
    env = parcel.envelope
    ref, current, faces, edges = (env[key] for key in ("reference", "current", "faces", "edges"))
    pairs = panel_pairs(ref, faces, env["panels"], [0], [3])
    cargo = normalize_contents([dict(shape="box", size_m=cfg.payload_size, mass_kg=cfg.payload_mass)])[0]
    cargo["position"] = env["payload_origin"].copy()
    assert parcel.cfg.radius == RADIUS
    before = measurements(ref, current, edges, cargo)
    shape, shortened, rejected = project(ref, current, faces, pairs, edges, cargo, keep_enclosed=True)
    after = measurements(ref, shape, edges, cargo)
    if not after["enclosed"] or after["maximum_edge_strain"] >= before["maximum_edge_strain"]:
        raise ValueError(f"The sample 4 preparation did not preserve enclosure and lower strain: {after}")
    # Restore the original lowest film height while translating the payload by
    # the same amount. World pose changes preserve the projected fit.
    shift = float(current[:, 2].min() - shape[:, 2].min())
    shape[:, 2] += shift
    m, d = parcel.m, parcel.d
    positions = m.jnt_qposadr[m.body_jntadr[m.flex_vertbodyid, None] + np.arange(3)]
    d.qpos[positions] = shape - ref
    d.qpos[m.joint("payload_free").qposadr[0] + 2] += shift
    case = RUN / "sample-004"
    case.mkdir(exist_ok=False)
    state = case / "initial_state.npz"
    save_initial_state(parcel, d, cfg, state)
    inputs["initial_state"] = state.relative_to(ROOT).as_posix()
    for name, data in (("config.json", inputs), ("smoke.json", inputs | {"smoke_seconds": .02}),
                       ("relaxation.json", dict(original=before, fitted=after, shortened_steps=shortened,
                                                rejected_steps=rejected, floor_alignment_shift_m=shift))):
        (case / name).write_text(json.dumps(data, indent=2) + "\n")
    print(case / "relaxation.json", after)


if __name__ == "__main__":
    main()
