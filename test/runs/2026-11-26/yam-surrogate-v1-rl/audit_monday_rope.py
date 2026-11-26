"""Reproduce the Monday proxy and read its load path without modifying its dynamics."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np

from src.low_order_parcel.lumped_contact_proxy import LumpedContactProxy


def replay(name, script, output, archived, trajectory, parameters, history_key, mass):
    spec = spec_from_file_location(name, script)
    proxy = module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(proxy)
    finally:
        sys.path.pop(0)

    rope_forces = []

    class MeasuredProxy(LumpedContactProxy):
        def step_with_reference(self, state, inputs, reference, inverse_mass=None):
            contact, length = super().step_with_reference(state, inputs, reference, inverse_mass)
            rope_forces.append(contact.force[0].detach().numpy().copy())
            return contact, length

    proxy.LumpedContactProxy = MeasuredProxy
    replay_file = output / f"{name}-replay.npz"
    if replay_file.exists():
        raise FileExistsError(replay_file)
    proxy.run(trajectory, replay_file, **parameters)

    with np.load(replay_file) as replayed, np.load(archived) as previous:
        keys = replayed.files
        identity = {key: bool(np.array_equal(replayed[key], previous[key])) for key in keys
                    if key not in {"wall_s", "steps"}}
        history = replayed[history_key]
        forces = np.stack(rope_forces)
        if len(history) != len(forces):
            raise RuntimeError(f"History and force samples disagree for {name}")

    # Both Monday exporters store the integration-step clearance and support
    # in columns 1 and 2. Their respective public replay is only a moving boundary.
    airborne = (history[:, 1] > .001) & (history[:, 2] <= 1e-6)
    hover = airborne & (history[:, 0] >= 7.5) & (history[:, 0] < 8.5)
    if not hover.any():
        raise RuntimeError(f"Monday trace {name} had no off-table window near 8 s")
    vertical_pull = forces[..., 2].sum(axis=1)
    mean_pull = float(vertical_pull[hover].mean())
    np.savez_compressed(output / f"{name}-rope-forces.npz", time_s=history[:, 0], rope_force_world_N=forces,
                        vertical_pull_N=vertical_pull, clearance_m=history[:, 1], support_N=history[:, 2])
    return dict(
        mass_kg=mass, step_s=.002, steps=len(history), finite=bool(np.isfinite(history).all()
        and np.isfinite(forces).all()), archived_trajectory_equal=all(identity.values()),
        unequal_fields=[key for key, matches in identity.items() if not matches],
        airborne_steps=int(airborne.sum()), peak_clearance_m=float(history[:, 1].max()),
        hover_window_s=[7.5, 8.5], hover_steps=int(hover.sum()), hover_mean_vertical_rope_N=mean_pull,
        hover_vertical_rope_fraction_of_weight=mean_pull / (9.81 * mass),
        peak_vertical_rope_N=float(vertical_pull.max()), native_robot_payload_force_measured=False,
        replay_robot_boundary="recorded pad kinematics", rope_endpoint="weighted positions on payload blocks",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    monday = Path(__file__).resolve().parents[2] / "2026-11-23" / "bag"
    fitting = monday / "umi-pe-yam-hold-r1"
    dynamic = monday / "real-grasp-skirt-c-r2"
    results = {
        "neck_98mm": replay(
            "neck-98mm", monday / "pe-yam-film-comparison-r1" / "proxy.py", args.output,
            fitting / "proxy_episode.npz", fitting / "q_ref.npz",
            dict(offset_y=0., cloth_xy=(.22, .31), neck_length=.098), "audit_2ms", .065,
        ),
        "center_250g": replay(
            "center-250g", dynamic / "proxy_contact.py", args.output,
            dynamic / "proxy_probe.npz", dynamic / "q_ref_top_center.npz",
            dict(mass=.25, stiffness=92.5), "history", .25,
        ),
    }
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    (args.output / "monday-rope-audit.json").write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
