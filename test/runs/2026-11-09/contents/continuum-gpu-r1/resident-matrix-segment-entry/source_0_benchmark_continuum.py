"""Resident CUDA-only step loops across continuum kappa and world counts.

This measures initial contact settling, not a completed grasp or its controller.
Compilation, warmup, state restore, readback and validity checks are outside timing.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "contents"), str(ROOT / "test/flex3d"), str(ROOT)]
import warp as wp
import numpy as np
from continuum.grasp_adapter import Config, scene
from continuum.gpu import Simulation
from continuum.model import preset, sample, shape_parameters


def configurations(matrix):
    if not matrix: return [dict(id="hard-box", material="hard", shape="box", **asdict(preset("hard")))]
    cases = []
    for material in ("hard", "bendable", "compressible"):
        for shape in ("box", "rounded", "pillow", "tapered"):
            p = replace(preset(material), **shape_parameters(shape))
            cases.append(dict(id=f"{material}-{shape}", material=material, shape=shape, **asdict(p)))
    rng = np.random.default_rng(20261109)
    for i in range(12):
        material = ("hard", "bendable", "compressible")[i % 3]
        cases.append(dict(id=f"random-{i:02d}-{material}", material=material, shape="box", **asdict(sample(rng, material))))
    return cases


def check_group(cpu_sim, nworld, steps, repeats, output):
    gpu = Simulation(cpu_sim.m, cpu_sim.d, nworld=nworld, nconmax=1024, njmax=4096)
    initial = gpu.state()
    rows = []
    with wp.ScopedDevice(gpu.device):
        for _ in range(20): gpu.step()
        gpu.assert_valid()
        start_event, end_event = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for repeat in range(repeats):
            gpu.restore(initial)
            actual_steps, elapsed, wall = 0, 0., 0.
            repeat_start = time.perf_counter()
            while actual_steps < steps:
                count = min(100, steps - actual_steps)
                wp.synchronize_device(gpu.device)
                wall_start = time.perf_counter()
                wp.record_event(start_event)
                for _ in range(count): wp.capture_launch(gpu.graph)
                wp.record_event(end_event)
                wp.synchronize_device(gpu.device)
                wall += time.perf_counter() - wall_start
                elapsed += wp.get_event_elapsed_time(start_event, end_event) / 1000
                actual_steps += count
                # No host reads inside the measured segment. Only diagnostics
                # leave the device here; physics state remains resident.
                health = gpu.health()
                if any(h["failure_bits"] for h in health): break
            valid = actual_steps == steps and all(h["failure_bits"] == 0 for h in health)
            rows.append(dict(repeat=repeat, steps=actual_steps, gpu_seconds=elapsed,
                             synchronized_wall_seconds=wall, end_to_end_seconds=time.perf_counter()-repeat_start,
                             valid=valid, health=health))
            if not valid:
                np.savez_compressed(output / f"failure-{nworld}w.npz", integration_state=gpu.state(),
                                    integration_state_spec=gpu.state_spec,
                                    overflow=gpu.data.overflow.numpy())
                break
    result = dict(valid=all(row["valid"] for row in rows) and len(rows) == repeats,
                  nworld=nworld, software=gpu.software(), rows=rows)
    if result["valid"]:
        seconds = statistics.median(r["gpu_seconds"] for r in rows)
        rate = steps / seconds
        result.update(median_gpu_steps_per_second=rate, median_gpu_ms_per_step=1000 / rate,
                      aggregate_world_steps_per_second=nworld * rate,
                      median_wall_steps_per_second=steps / statistics.median(r["synchronized_wall_seconds"] for r in rows),
                      per_world_realtime_ratio=rate * cpu_sim.m.opt.timestep)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--worlds", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Limit cases for an entry check")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or any(w < 1 for w in args.worlds): parser.error("positive counts required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((ROOT / "contents/continuum").glob("*.py")),
               ROOT / "bag_sim/flex_residual.py", ROOT / "test/flex3d/grasp_platform.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for i, source in enumerate(sources): (args.output / f"source_{i}_{source.name}").write_bytes(source.read_bytes())
    cases = configurations(args.matrix)[:args.limit]
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    result = dict(hostname=platform.node(), sources=hashes, full_robot=True, seed=20261109,
                  control="held at initialized values", steps_per_repeat=args.steps, repeats=args.repeats,
                  protocol="resident step graphs; CUDA-event segments of 100 steps, no host reads in timing; initial contact settling only",
                  groups=[])
    for case in cases:
        folder = args.output / case["id"]
        folder.mkdir()
        options = {k: v for k, v in case.items() if k != "id"}
        options["category"] = "compressible" if case["material"] == "compressible" else "bendable"
        robot = scene(Config(**options, pinch_coeff=0.))
        for worlds in args.worlds:
            try: group = check_group(robot.sim, worlds, args.steps, args.repeats, folder)
            except Exception as error:
                group = dict(nworld=worlds, valid=False, error=str(error), traceback=traceback.format_exc())
            group["case"] = case
            result["groups"].append(group)
            (folder / f"{worlds}w.json").write_text(json.dumps(group, indent=2))
            print(json.dumps(dict(case=case["id"], worlds=worlds, valid=group["valid"],
                  steps_per_second=group.get("median_gpu_steps_per_second"),
                  aggregate=group.get("aggregate_world_steps_per_second"),
                  stopped_at=group.get("rows", [{}])[-1].get("steps"), error=group.get("error"))), flush=True)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    result["sources_unchanged"] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    result["valid_groups"] = sum(g["valid"] for g in result["groups"])
    result["invalid_groups"] = len(result["groups"]) - result["valid_groups"]
    result["all_groups_valid"] = result["invalid_groups"] == 0 and result["sources_unchanged"]
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("sources", "groups")}), flush=True)
    return int(not result["sources_unchanged"])


if __name__ == "__main__": raise SystemExit(main())
