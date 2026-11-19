"""Verify the full patched run and summarize measured single-environment cost."""

import argparse
import json
from pathlib import Path


def main():
    default_run = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=default_run)
    parser.add_argument("--cpu", type=Path, default=default_run.parent / "rss2027-stage1-pickup-baseline-cpu-r4"
                        / "recording" / "trial" / "result.json")
    args = parser.parse_args()
    run = args.run
    recording = run / "recording"
    result = json.loads((recording / "summary.json").read_text())
    trace = json.loads((recording / "trial/trace.json").read_text())
    cpu = json.loads(args.cpu.read_text())
    window = [row for row in trace if row["phase"] == "lift_hold"]
    hold_phase = next((phase for phase in result["phases"] if phase["phase"] == "lift_hold"), None)
    hold_s = hold_phase["end_s"] - hold_phase["start_s"] if hold_phase else 0.
    memories = [sample["process_memory_mib"] for sample in result["gpu_samples"]
                if sample["process_memory_mib"] is not None]
    overflow = [sample["overflow_bits"] for sample in result["gpu_samples"]]
    geometry = [row["payload_geometry"] for row in trace if "payload_geometry" in row]
    max_outside = max((row["outside_corners"] for row in geometry), default=None)
    max_intersection = max((row["geometric_depth_upper_m"] for row in geometry), default=None)
    min_grip = min((min(row["grip_each_N"]) for row in window), default=0.)
    max_support = max((row["support_N"] for row in window), default=None)
    min_lift = min((row["com"][2] - result["initial_com"][2] for row in window), default=None)
    dt, duration, wall = result["config"]["dt"], result["sim_time_s"], result["wall_s"]
    checks = dict(
        motion=result["outcome"] == "pickup_ok" and result["motion_completed"],
        duration=3. <= duration <= 5.,
        hold=bool(window) and hold_s >= .20 - dt / 2 and min_grip > .001 and max_support <= .05,
        numerical=not result["warnings"] and result["peaks"]["penetration_mm"] <= 1.,
        enclosure=len(geometry) == len(trace) and max_outside == 0 and max_intersection <= .001,
        capacity=bool(overflow) and not any(overflow),
    )
    verified = all(checks.values())
    report = dict(
        verified=verified, checks=checks, outcome=result["outcome"], reason=result["reason"],
        model_film_vertices=result["dimensions"]["film_vertices"], dt_s=dt,
        sim_time_s=duration, wall_s=wall, sampled_peak_gpu_memory_mib=max(memories, default=None),
        peak_native_penetration_mm=result["peaks"]["penetration_mm"],
        sampled_max_exposed_corners=max_outside, sampled_max_mesh_payload_depth_m=max_intersection,
        hold_samples=len(window), hold_duration_s=hold_s, hold_min_grip_per_finger_N=min_grip,
        hold_max_support_N=max_support, hold_min_content_rise_m=min_lift,
        cpu_sim_time_s=cpu["sim_time_s"], cpu_wall_s=cpu["wall_s"],
    )
    if verified:
        step_cost_s = wall * dt / duration
        report["serial_scaling"] = dict(
            scope="same closed-loop workload, one environment, excluding learning",
            equivalent_physics_steps_per_wall_second=1. / step_cost_s,
            gpu_hours_per_million_physics_steps=step_cost_s * 1e6 / 3600.,
        )
    with (run / "benchmark.json").open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
