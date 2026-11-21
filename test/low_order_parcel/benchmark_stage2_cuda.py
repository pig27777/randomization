"""Time complete, verified Stage 2 probe batches on one CUDA device."""

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import torch
from torch.utils._pytree import tree_map

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.low_order_parcel.lumped_contact_proxy import ContactState, LumpedContactProxy, ProxyParams  # noqa: E402
from src.low_order_parcel.macro_parcel_sim import MacroParcelSim, ParcelAction

CONFIG = ROOT / "configs/proxy_material_envelopes.yaml"
ARCHIVE = ROOT / "test/runs/2026-11-20/bag"
BASELINES = ("rss2027-stage1-warp-film-normal-r1", "rss2027-stage1-warp-film-normal-30pct-r1",
             "rss2027-stage1-pickup-200-warp-r1")
BATCH_REFERENCE = ARCHIVE / "yam-batch-full-grasp-r1/probe-8000-200us/benchmark.json"
DT = .002
DURATION = 4.77


def build_sim(config, worlds, device="cuda"):
    materials = tuple(config["materials"])
    nominal = [config["materials"][materials[i % len(materials)]]["nominal"] for i in range(worlds)]
    arrays = {field.name: torch.as_tensor([row["proxy"][field.name] for row in nominal], dtype=torch.float32,
                                          device=device).reshape(worlds, -1) for field in fields(ProxyParams)}
    proxy = LumpedContactProxy(ProxyParams(**arrays), DT)
    mass = proxy.params.S_bag.new_tensor([[row["mass_kg"]] for row in nominal])
    return MacroParcelSim(proxy, mass, torch.zeros((worlds, 3), device=device)), materials


def state_tensors(state):
    return (*state[:-1], *state.payload) if hasattr(state, "payload") else tuple(state)


def process_memory_mib():
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    return max((int(memory) for pid, memory in (line.split(", ") for line in result.splitlines())
                if pid == str(os.getpid()) and memory.isdecimal()), default=0)


@torch.no_grad()
def timed_episode(config, worlds, baselines, *, make_sim=build_sim, grip_N=6., steer=None):
    initial_start = perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sim, labels = make_sim(config, worlds)
    if callable(grip_N):
        grip_N = grip_N(sim)
    times, _, _, actions = sim.probe_actions("side", duration_s=DURATION, grip_N=grip_N)
    steps = len(times)
    state = tree_map(torch.Tensor.clone, sim.initial_state())
    buffers = state_tensors(state)
    cursor = torch.zeros(1, device="cuda", dtype=torch.long)
    finite = torch.ones(worlds, device="cuda", dtype=torch.bool)
    stable = torch.zeros(worlds, device="cuda", dtype=torch.int32)
    peak_tension = sim.mass.new_zeros(worlds)
    admissible = torch.ones(worlds, device="cuda", dtype=torch.bool)
    offset = torch.zeros_like(state.com)

    def reset():
        for current, initial in zip(buffers, state_tensors(sim.initial_state()), strict=True):
            current.copy_(initial)
        cursor.zero_()
        finite.fill_(True)
        stable.zero_()
        peak_tension.zero_()
        admissible.fill_(True)
        offset.zero_()

    def advance():
        action = ParcelAction(*(torch.index_select(channel, 0, cursor).squeeze(0) for channel in actions))
        if steer is not None:
            correction = steer(sim, state, action, offset)
            action = action._replace(pinch_position=action.pinch_position + correction[:, None],
                                     pinch_velocity=action.pinch_velocity + ((correction - offset) / DT)[:, None])
            offset.copy_(correction)
        result = sim.step(state, action)
        for current, update in zip(buffers, state_tensors(result.state), strict=True):
            current.copy_(update)
            finite.logical_and_(torch.isfinite(update.flatten(1)).all(dim=1))
        for channel in result[1:]:
            finite.logical_and_(torch.isfinite(channel.reshape(worlds, -1)).all(dim=1))
        if sim.payload is not None:
            depth = state.payload.d_surf
            admissible.logical_and_(((0. <= depth) & (depth <= sim.payload.depth_max)).all(dim=1)
                                    & (result.table_clearance[:, 0] >= 0.))
        both = (state.contacts[:, :, ContactState.CAPTURE] > .5).all(dim=1)
        airborne = (result.table_clearance[:, 0] > .001) & (result.support_N[:, 0] == 0.)
        quiet = torch.linalg.vector_norm(state.velocity, dim=1) < .020
        stable.copy_(torch.where(both & airborne & quiet, stable + 1, 0))
        peak_tension.copy_(torch.maximum(peak_tension, result.tension.max(dim=1).values))
        cursor.add_(1)

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            advance()
    torch.cuda.current_stream().wait_stream(warm)
    reset()
    torch.cuda.synchronize()
    start = perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        advance()
    reset()
    torch.cuda.synchronize()
    setup_s, capture_s = perf_counter() - initial_start, perf_counter() - start

    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = perf_counter()
    begin.record()
    for _ in range(steps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    wall = perf_counter() - start
    if cursor.item() != steps:
        raise RuntimeError("The graph did not advance exactly one step per replay")
    hold_required = math.ceil(.20 / DT)
    valid = finite & admissible & (stable >= hold_required)
    counts = {label: int(valid[i::len(labels)].sum().item()) for i, label in enumerate(labels)}
    completed = sum(counts.values())
    rate = worlds * steps / wall
    return dict(worlds=worlds, steps_per_world=steps, simulated_s_per_world=steps * DT, wall_s=wall,
                device_timed_s=begin.elapsed_time(end) / 1000, setup_s=setup_s, capture_s=capture_s,
                env_steps_per_s=rate, sim_seconds_per_wall_second=rate * DT,
                verified_sim_seconds_per_wall_second=completed * steps * DT / wall,
                verified_pickups=completed, verified_by_material=counts, pickups_per_hour=3600. * completed / wall,
                min_final_continuous_hold_s=float(stable.min().item()) * DT,
                final_bilateral_capture_worlds=int((state.contacts[:, :, ContactState.CAPTURE] > .5).all(dim=1).sum().item()),
                invalid_geometry_worlds=worlds - int(admissible.sum().item()),
                nonfinite_worlds=worlds - int(finite.sum().item()), peak_tension_N=float(peak_tension.max().item()),
                sampled_process_gpu_memory_mib=process_memory_mib(),
                peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                aggregate_pickup_speedup_vs_stage1={name: completed / wall * old["wall_s"] for name, old in baselines.items()},
                passed=completed == worlds)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worlds", type=int, nargs="+", default=(512, 2048, 4096, 8192))
    args = parser.parse_args()
    if min(args.worlds) < 1 or not torch.cuda.is_available():
        parser.error("The complete throughput test needs positive batches and CUDA PyTorch")
    raw = CONFIG.read_bytes()
    config = json.loads(raw)
    baseline_records = {key: json.loads((ARCHIVE / key / "benchmark.json").read_bytes()) for key in BASELINES}
    baselines = {key: {field: row[field] for field in ("wall_s", "sim_time_s", "dt_s")}
                 for key, row in baseline_records.items()}
    short_reference = json.loads(BATCH_REFERENCE.read_bytes())
    short_rate = short_reference["env_steps_per_s"] * short_reference["dt_s"]
    report = dict(machine=platform.node(), device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, dt_s=DT, nominal_config_sha256=hashlib.sha256(raw).hexdigest(),
                  mode="complete 4.77 s proxy side-grasp and hover on CUDA Graph; four nominal materials",
                  stage1_reference=dict(mode="archived single-world complete articulated YAM with deformable shell", **baselines),
                  high_fidelity_batch_reference=dict(mode="64 physics steps at an archived hold; no full-grasp inference",
                                                      worlds=short_reference["worlds"], sim_seconds_per_wall_second=short_rate,
                                                      file=BATCH_REFERENCE.relative_to(ROOT).as_posix()),
                  cases=[])
    with args.output.open("x") as output:
        for worlds in args.worlds:
            row = timed_episode(config, worlds, baselines)
            row["physical_throughput_ratio_vs_high_fidelity_hold_probe"] = row["sim_seconds_per_wall_second"] / short_rate
            report["cases"].append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(row["passed"] for row in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
