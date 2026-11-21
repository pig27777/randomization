"""Step A1, instrument the frozen P6 episode without changing the contact dynamics."""
import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test/low_order_parcel"))
from test_unified_payload import STAGE2, DT, heterogeneous_probe, hover_grip, hover_steering
from benchmark_stage2_cuda import DURATION, timed_episode


def csv_rows(path, rows):
    with path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_file(path, value):
    with path.open("x") as output:
        json.dump(value, output, allow_nan=False, indent=2)
        output.write("\n")


@torch.inference_mode()
def run(output, worlds):
    output.mkdir(parents=True, exist_ok=False)
    sim, materials = heterogeneous_probe(STAGE2, worlds)
    # Keep references to the very inputs used by this instance; all reads occur
    # before the next graph replay. The old capture state is updated in-place
    # by the benchmark, so preserve a copy at the call boundary.
    original_step = sim.proxy.step
    def record_contact(state, inputs):
        nonlocal contact, boundary, was_captured
        was_captured = state[:, :, 1].clone()
        boundary = inputs
        contact = original_step(state, inputs)
        return contact

    sim.proxy.step = record_contact
    contact = boundary = was_captured = None
    tape = None
    names = []
    def observe(cursor, action, result, finite, admissible, stable):
        nonlocal tape, names
        delta = boundary.pinch_position - boundary.bag_attachment
        unit = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axial = ((boundary.pinch_velocity - boundary.attachment_velocity) * unit).sum(-1)
        channels = {
            "T_N": result.tension, "elastic_N": contact.elastic_tension,
            "axial_mps": axial, "slip_mps": result.slip_rate,
            "normal_N": boundary.normal_force, "command_N": action.normal_force,
            "cap_before": was_captured, "cap": result.state.contacts[:, :, 1],
            "reserve_m": result.state.contacts[:, :, 0],
            "pad_vx": action.pinch_velocity[:, :, 0], "pad_vy": action.pinch_velocity[:, :, 1],
            "pad_vz": action.pinch_velocity[:, :, 2], "anchor_axial_mps": (boundary.attachment_velocity * unit).sum(-1),
            "indent_m": result.state.payload.d_surf, "clearance_m": result.table_clearance,
            "support_N": result.support_N, "com_velocity_mps": result.state.velocity, "speed_mps": result.state.velocity.norm(dim=-1, keepdim=True),
            "final_run_steps": stable[:, None], "finite": finite[:, None], "geometry_ok": admissible[:, None],
        }
        frame = torch.cat(tuple(channels.values()), dim=1)
        if tape is None:
            names = [f"{key}_{j}" for key, v in channels.items() for j in range(v.shape[1])]
            tape = frame.new_empty((round(DURATION / DT), *frame.shape))
        tape.index_copy_(0, cursor, frame.unsqueeze(0))

    stats = timed_episode(STAGE2, worlds, {}, make_sim=lambda *_: (sim, materials),
                          grip_N=hover_grip, steer=hover_steering, observe=observe)
    peak = tape[:, :, :2].amax(dim=(0, 2)).cpu().numpy()
    tail = tape[-1].cpu().numpy()
    col = names.index
    passed = (tail[:, col("final_run_steps_0")] >= 100) & tail[:, -2:].all(axis=1)
    ranks = np.argsort(-peak, kind="stable")
    failed = np.flatnonzero(~passed)
    selected = np.union1d(ranks[:20], failed)
    wave = tape.index_select(1, torch.as_tensor(selected, device=tape.device)).cpu().numpy()
    assert int(passed.sum()) == stats["verified_pickups"] and float(peak.max()) == stats["peak_tension_N"]
    if worlds == 8192:
        assert stats["verified_by_material"] == {"pe":2037,"kraft":2021,"bubble":2038,"paperboard":1907}
        assert stats["final_bilateral_capture_worlds"] == 8162 and abs(float(peak.max()) - 343.974884) < .01
    assert stats["nonfinite_worlds"] == stats["invalid_geometry_worlds"] == 0
    np.savez_compressed(output / "waveforms.npz", world_id=selected, columns=np.array(names),
                        time_s=np.arange(len(wave)) * DT, frames=wave)
    rank = np.argsort(ranks) + 1
    samples, events, parameters = [], [], {}
    for obj in (sim.payload.params, sim.proxy.params):
        parameters[type(obj).__name__] = {f.name:getattr(obj, f.name).index_select(
            0, torch.as_tensor(selected, device=tape.device)).cpu().tolist() for f in fields(obj)}
    params = {}
    for i, world in enumerate(selected):
        w = wave[:, i]
        t, pad = np.unravel_index(w[:, :2].argmax(), w[:, :2].shape)
        def value(key): return float(w[t, col(f"{key}_{pad}")])
        params[int(world)] = {group:{k:v[i] for k,v in values.items()} for group,values in parameters.items()}
        proxy = params[int(world)]["ProxyParams"]
        dtail = w[-100:]
        gate = {"capture":dtail[:, col("cap_0"):col("cap_1")+1].min(axis=1) < .5,
                "clearance":dtail[:, col("clearance_m_0")] <= .001,
                "support":dtail[:, col("support_N_0")] != 0,
                "speed":dtail[:, col("speed_mps_0")] >= .020}
        bad = [key for key, mask in gate.items() if mask.any()]
        assert bool(bad) != bool(passed[world])
        losses = {}
        for finger in range(2):
            captured = w[:, col(f"cap_{finger}")]
            was_active = w[:, col(f"cap_before_{finger}")] > .5
            drops = np.flatnonzero(was_active & (captured < .5))
            reason = "held_at_end" if captured[-1] else "never_captured"
            if not captured[-1] and len(drops):
                last_drop = drops[-1]
                causes = []
                if w[last_drop, col(f"reserve_m_{finger}")] <= 1e-8:
                    causes.append("empty_reserve")
                if w[last_drop, col(f"normal_N_{finger}")] < proxy["F_release"][0]:
                    causes.append("normal_below_release")
                reason = "+".join(causes)
                assert reason
            losses[f"end_pad_{finger}"] = reason
            losses[f"last_drop_s_{finger}"] = float(drops[-1] * DT) if len(drops) else None
        damp = proxy["c_d"][0] * (value("axial_mps") - value("slip_mps"))
        samples.append(dict(world_id=world, material=materials[world % len(materials)], rank=rank[world],
            pass_hold=int(passed[world]), peak_N=value("T_N"), time_s=t*DT, pad=pad,
            elastic_N=value("elastic_N"), damping_N=damp, normal_N=value("normal_N"),
            command_N=value("command_N"), muFn_N=value("normal_N")*proxy["mu_eff"][pad],
            capture_on_peak=int(not value("cap_before") and value("cap")), slip_mps=value("slip_mps"),
            over50_steps=int((w[:, :2].max(axis=1)>50).sum()),
            longest_hold_s=w[:, col("final_run_steps_0")].max()*DT,
            end_hold_s=w[-1,col("final_run_steps_0")]*DT, failed_gates="+".join(bad),
            max_COM_speed_mps=float(w[:, col("speed_mps_0")].max()),
            tail_max_speed_mps=float(dtail[:, col("speed_mps_0")].max()), **losses,
            **{f"{k}_bad_steps":int(v.sum()) for k,v in gate.items()}))
        for j in range(max(0,t-5), min(t+6,len(w))):
            events.append(dict(world_id=world, frame=j, time_s=j*DT, **dict(zip(names,w[j],strict=True))))
    csv_rows(output / "worlds.csv", [dict(world_id=i,material=materials[i%len(materials)],rank=rank[i],
             peak_N=peak[i],pass_hold=int(passed[i])) for i in range(worlds)])
    csv_rows(output / "selected.csv", sorted(samples,key=lambda row:row["rank"]))
    csv_rows(output / "peak_windows.csv", events)
    json_file(output / "parameters.json", params)
    stats.update(n_over_50=int((peak>50).sum()), n_failed_over_50=int(((peak>50)&~passed).sum()),
                 n_selected=len(selected), seed=1726481, dt=DT, device=torch.cuda.get_device_name(),
                 torch=str(torch.__version__))
    json_file(output / "p6.json",stats)
    print(json.dumps(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--worlds",type=int,default=8192)
    args = parser.parse_args()
    run(args.output,args.worlds)
