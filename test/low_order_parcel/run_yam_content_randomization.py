"""Sample archived package geometries and replay the same YAM grasp protocol."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experience.yam_platform.protocol import run  # noqa: E402
from src.low_order_parcel import yam_adapter  # noqa: E402
from src.low_order_parcel.lumped_contact_proxy import ContactState  # noqa: E402

GEOMETRY = ROOT / "test/runs/2026-11-20/bag/rss2027-stage1-geometric-slack-budget-r1/slack-limits.csv"


def vector(row, names):
    return [float(row[name]) for name in names]


def draw_samples(seed):
    with GEOMETRY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(seed)
    for group in ("0.30", "0.60", "0.90"):
        candidates = []
        for row in rows:
            if row["group"] != group:
                continue
            size = vector(row, ("payload_x_m", "payload_y_m", "payload_z_m"))
            inner = vector(row, ("cavity_x_m", "cavity_y_m"))
            if (.065 < size[0] < .22 and .065 < size[1] < .22 and .025 < size[2] < .085
                    and .22 < inner[0] < .38 and .18 < inner[1] < .30):
                candidates.append(row)
        sample = candidates[rng.integers(len(candidates))]
        size = vector(sample, ("payload_x_m", "payload_y_m", "payload_z_m"))
        density = float(rng.uniform(90., 180.))
        options = dict(
            payload_size_m=size, payload_mass_kg=float(np.prod(size) * density),
            bag_inner_size_m=vector(sample, ("cavity_x_m", "cavity_y_m")),
            payload_offset_xy_m=vector(sample, ("offset_x_m", "offset_y_m")),
            grasp_face="side", complete_motion=True, lift_height=.25, lift_speed_m_s=.05,
            simulation_seconds=22.,
        )
        yield dict(name=f"fill-{int(float(group) * 100):02d}", sample_id=int(sample["sample"]),
                   fill=float(group), source_side_slack_m=float(sample["slack_y_minus_m"]),
                   density_kg_m3=density, options=options)


def run_trial(cfg, output):
    robot = None

    def retain_robot(current, sample):
        nonlocal robot
        robot = current

    result = run(yam_adapter, cfg, output, on_sample=retain_robot)
    if result["outcome"] != "pickup_release_ok" or robot is None:
        return
    # Continue the same simulated state after the protocol confirms release.
    # The recorded verdict and timing still describe the standard trial.
    sim = robot.sim
    stride = max(1, round(.02 / cfg.dt))
    poses, times, clearance, support = [], [], [], []
    for step in range(round(.70 / cfg.dt)):
        robot.command(robot.target, cfg.grasp_opening_m)
        sim.step()
        if step % stride == 0:
            if (sim.state.contacts[0, :, ContactState.CAPTURE] > .5).any():
                raise RuntimeError("A released parcel was unexpectedly recaptured during the landing replay")
            poses.append(sim.d.qpos.copy())
            times.append(float(sim.d.time))
            clearance.append(float(sim.last_step.table_clearance[0, 0]))
            support.append(float(sim.last_step.support_N[0, 0]))
    np.savez_compressed(output / "post-release.npz", qpos=poses, time=times,
                        clearance_m=clearance, support_N=support)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = list(draw_samples(args.seed))
    manifest = dict(seed=args.seed, source=str(GEOMETRY.relative_to(ROOT)), samples=samples)
    (args.output / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for sample in samples:
        print(json.dumps(sample), flush=True)
        run_trial(yam_adapter.Config(**sample["options"]), args.output / sample["name"])


if __name__ == "__main__":
    main()
