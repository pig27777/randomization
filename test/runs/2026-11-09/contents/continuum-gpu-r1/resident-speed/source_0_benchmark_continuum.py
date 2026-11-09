"""Locked one-world resident CUDA graph timing, with the full YAM at fixed control.

This measures initial contact settling, not a completed grasp or its controller.
Compilation, warmup, state restore, readback and validity checks are outside timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "contents"), str(ROOT / "test/flex3d"), str(ROOT)]
import warp as wp
from continuum.grasp_adapter import Config, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((ROOT / "contents/continuum").glob("*.py")),
               ROOT / "bag_sim/flex_residual.py", ROOT / "test/flex3d/grasp_platform.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for i, source in enumerate(sources): (args.output / f"source_{i}_{source.name}").write_bytes(source.read_bytes())
    robot = scene(Config(material="hard", shape="box", pinch_coeff=0.))
    sim = robot.sim
    sim.prepare()
    gpu = sim.gpu
    initial = gpu.state()
    for _ in range(20): gpu.step()
    gpu.assert_valid()
    rows = []
    with wp.ScopedDevice(gpu.device):
        start_event, end_event = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for repeat in range(3):
            gpu.restore(initial)
            wp.synchronize_device(gpu.device)
            wall_start = time.perf_counter()
            wp.record_event(start_event)
            for _ in range(400): gpu.step()
            wp.record_event(end_event)
            wp.synchronize_device(gpu.device)
            wall = time.perf_counter() - wall_start
            elapsed = wp.get_event_elapsed_time(start_event, end_event) / 1000
            health = gpu.health()[0]
            rows.append(dict(repeat=repeat, steps=400, gpu_seconds=elapsed, synchronized_wall_seconds=wall,
                             gpu_steps_per_second=400 / elapsed, wall_steps_per_second=400 / wall,
                             valid=health["failure_bits"] == 0, health=health))
    unchanged = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    valid = all(row["valid"] for row in rows) and unchanged
    result = dict(hostname=platform.node(), software=gpu.software(), sources=hashes, sources_unchanged=unchanged,
                  valid=valid, nworld=1, full_robot=True, control="held at initialized values",
                  protocol="initial contact settling only; no live YAM controller or grasp evaluation",
                  timestep=sim.m.opt.timestep, nv=sim.m.nv, nplastic=sim.m.nplastic, rows=rows)
    if valid:
        rate = statistics.median(r["gpu_steps_per_second"] for r in rows)
        result.update(median_gpu_steps_per_second=rate, median_gpu_ms_per_step=1000 / rate,
                      median_wall_steps_per_second=statistics.median(r["wall_steps_per_second"] for r in rows),
                      gpu_realtime_ratio=rate * sim.m.opt.timestep)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "rows")}), flush=True)
    return int(not valid)


if __name__ == "__main__": raise SystemExit(main())
