"""Measure supported flat-film load at a fixed 20 micrometre clearance."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parents[3] / "bag_model"))
from rss2027_micropatch import PAD_BOTTOM, build, pad_reactions  # noqa: E402


def dwell(dt, pad_time, support_time):
    _, model, data, _, _, _ = build(12, 0., dt, .5, False, pad_time)
    support = model.geom("support").id
    model.geom_solref[support, 0] = support_time
    bodies = [model.body(name).id for name in ("left", "right")]
    joints = model.body_jntadr[bodies, None] + np.arange(3)
    pad_qpos = model.jnt_qposadr[joints]
    pad_dofs = model.jnt_dofadr[joints]
    positions = np.zeros((2, 3))
    positions[:, 2] = PAD_BOTTOM + 20e-6 - model.body_pos[bodies, 2]
    pads = {
        int(geom): side
        for side, body in enumerate(bodies)
        for geom in np.flatnonzero(model.geom_bodyid == body)
    }
    observed = []
    peak_penetration = 0.
    for frame in range(round(.030 / dt)):
        data.qpos[pad_qpos] = positions
        data.qvel[pad_dofs] = 0
        mujoco.mj_step(model, data)
        depth = -float(data.contact.dist.min()) if data.ncon else 0.
        peak_penetration = max(peak_penetration, depth)
        if frame % round(.00025 / dt) == 0:
            force, _ = pad_reactions(model, data, pads)
            vertical = force[:, 2]
            reaction = data.qfrc_constraint[pad_dofs[:, 2]]
            observed.append((data.time - dt, float(vertical.sum()), float(np.max(np.abs(vertical - reaction)))))
    force = [p for t, p, _ in observed if t >= .020]
    return dict(
        dt_s=dt, pad_contact_time_s=pad_time, support_contact_time_s=support_time, bottom_gap_m=20e-6,
        final_band_load_N=[min(force), max(force)], final_band_mean_N=float(np.mean(force)),
        peak_penetration_m=peak_penetration, max_reaction_error_N=max(e for _, _, e in observed),
        warnings=data.warning.number.tolist(), force_trace=observed
    )


if __name__ == "__main__":
    configs_us = ((50, 100, 500), (50, 100, 100), (25, 50, 50), (12.5, 25, 25))
    trials = [dwell(*(value * 1e-6 for value in config)) for config in configs_us]
    (RUN / "normal-dwell.json").write_text(json.dumps(trials, indent=2) + "\n")
    for result in trials:
        print({k: v for k, v in result.items() if k != "force_trace"}, flush=True)
