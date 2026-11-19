"""Reuse the numerical starting state to isolate the timestep in load-ramp validation."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
ORIGINAL = RUN.with_name("rss2027-stage1-kraft-load-ramp-clear-r1")
sys.path.insert(0, str(ROOT / "test/bag_model"))
from physical_parcel_yam import Config, parcel_world, prepared_signature  # noqa: E402


def main():
    settings = json.loads((ORIGINAL / "config.json").read_bytes())
    original = Config(**settings)
    source = ROOT / original.initial_state
    same_pose, _ = parcel_world(original)
    half_step = replace(original, dt=original.dt / 2)
    fine, _ = parcel_world(half_step)
    with np.load(source) as state:
        positions, velocities = state["qpos"], state["qvel"]
        if state["config_json"].item() != prepared_signature(same_pose, original):
            raise ValueError("The original preparation signature changed")
    assert positions.shape == (same_pose.m.nq,) == (fine.m.nq,)
    assert velocities.shape == (same_pose.m.nv,) == (fine.m.nv,)
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    destination = RUN / "initial_state.npz"
    with destination.open("xb") as stream:
        np.savez_compressed(stream, qpos=positions, qvel=velocities,
                            config_json=np.asarray(prepared_signature(fine, half_step)))
    config = settings | {"dt": half_step.dt, "initial_state": destination.relative_to(ROOT).as_posix()}
    (RUN / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = dict(original_state=source.relative_to(ROOT).as_posix(), original_dt_s=original.dt,
                   compared_dt_s=half_step.dt, original_state_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   copied_positions=len(positions), copied_velocities=len(velocities))
    (RUN / "same-state.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
