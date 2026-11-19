"""Check vertical reaction against the solver's independent generalized force."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
from rss2027_micropatch import PAD_BOTTOM, build, pad_reactions  # noqa: E402


def probe(slack):
    _, model, data, _, _, _ = build(12, slack, .00005, .5, False, .0001)
    bodies = [model.body(name).id for name in ("left", "right")]
    z_joints = model.body_jntadr[bodies] + 2
    z_positions = model.jnt_qposadr[z_joints]
    z_forces = model.jnt_dofadr[z_joints]
    pad_origins = model.body_pos[bodies, 2]
    pads = {
        int(geom): side
        for side, body in enumerate(bodies)
        for geom in np.flatnonzero(model.geom_bodyid == body)
    }
    for depth in (600, 200, 150, 100, 75, 50, 20):
        gap = depth * 1e-6
        data.qpos[z_positions] = PAD_BOTTOM + gap - pad_origins
        data.qacc_warmstart[:] = 0
        mujoco.mj_forward(model, data)
        force, _ = pad_reactions(model, data, pads)
        hits = sum(c.efc_address >= 0 and sum(int(geom) in pads for geom in c.geom) == 1 for c in data.contact)
        yield dict(
            gap_m=gap,
            pad_contact_count=hits,
            force_N=force[:, 2].tolist(),
            generalized_force_N=data.qfrc_constraint[z_forces].tolist(),
            reaction_error_N=float(np.max(np.abs(force[:, 2] - data.qfrc_constraint[z_forces]))),
            max_penetration_m=max(0., -float(data.contact.dist.min())) if data.ncon else 0.,
        )


if __name__ == "__main__":
    results = {"flat": list(probe(0.)), "slack_2mm_initial_tent": list(probe(.002))}
    (RUN / "normal-probe.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, rows in results.items():
        print(name)
        for row in rows:
            print(row["gap_m"], sum(row["force_N"]), row["pad_contact_count"], row["reaction_error_N"])
