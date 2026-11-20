"""Attribute the native YAM top-grasp load to each actual colliding surface."""

import argparse
import json
from pathlib import Path

import numpy as np


def resolve(sample):
    world = np.zeros((2, 3))
    inner = np.zeros(2)
    colliders = {}
    normal_down = tangent_scale = native_tangent_bound = 0.
    for contact in sample["contacts"]:
        pad = contact["pad"]
        force = np.asarray(contact["force_on_finger_N"])
        normal = contact["force_contact_N"][0]
        world[pad] += force
        if contact["standard_pad"] and contact["inner"]:
            inner[pad] += normal
        geometry = colliders.setdefault(contact["geom"], np.zeros(3))
        # Upward forces are positive on the membrane.
        z = contact["normal_on_pad"][2]
        normal_z = -normal * z
        geometry += (-force[2], normal_z, -force[2] - normal_z)
        tangent = normal * np.sqrt(max(0., 1. - z * z))
        normal_down -= normal_z
        tangent_scale += tangent
        native_tangent_bound += max(contact["friction"]) * tangent
    residual = float(np.max(np.abs(world - sample["force_on_fingers_N"])))
    if residual > 1e-10:
        raise ValueError(f"Native finger reactions disagree at {sample['t_s']}: {residual} N")
    tension = -float(world[:, 2].sum())
    if tension > native_tangent_bound - normal_down + 1e-8:
        raise ValueError(f"Observed tension exceeded its friction-cone bound at {sample['t_s']}")
    bound = dict(normal_down_N=normal_down, native_mu_upper_N=native_tangent_bound - normal_down,
                 upper_at_mu_0p9_N=.9 * tangent_scale - normal_down,
                 uniform_mu_for_zero_tension=max(0., normal_down / tangent_scale) if tangent_scale > 0. else None)
    return dict(t_s=sample["t_s"], phase=sample["phase"], gap_m=sample["gap_m"],
                command_gap_m=sample["command_gap_m"], tension_N=tension,
                support_N=sample["support_N"], standard_inner_normal_N=inner.tolist(),
                residual_N=residual, colliders={name: force.tolist() for name, force in colliders.items()},
                fixed_contact_friction_bound=bound)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    samples = [resolve(row) for row in json.loads(args.source.read_bytes())]
    time = np.array([row["t_s"] for row in samples])
    np.testing.assert_allclose(np.diff(time), .002, atol=1e-9, rtol=0.)
    lifting = [row for row in samples if row["phase"] in ("lift", "lift_hold")]
    bilateral = [row for row in samples if min(row["standard_inner_normal_N"]) > .001]
    accepted = [row for row in lifting if min(row["standard_inner_normal_N"]) > .001 and
                row["tension_N"] > .001 and row["support_N"] <= .05]
    times = [row["t_s"] for row in accepted]
    spans = np.split(times, np.flatnonzero(np.diff(times) > .00200001) + 1)
    result = dict(samples=len(samples), sample_period_s=.002, bilateral_samples=len(bilateral),
                  peak_force_residual_N=max(row["residual_N"] for row in samples),
                  first_bilateral=next(iter(bilateral), None), first_lift=next(iter(lifting), None),
                  peak_bilateral_tension=max(bilateral, key=lambda row: row["tension_N"], default=None),
                  lift_peak_tension=max(lifting, key=lambda row: row["tension_N"], default=None),
                  longest_positive_unsupported_bilateral_s=max((s[-1] - s[0] for s in spans if len(s)), default=0.),
                  collider_columns=("total_tension_N", "normal_tension_N", "tangential_tension_N"))
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
