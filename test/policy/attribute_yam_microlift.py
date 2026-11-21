"""Decompose the hover gates and optionally recover missing 2 ms loads by exact replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def longest(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return float(max(edges[1::2] - edges[::2], default=0) * dt)


def analyze(root, replay_loads):
    rows = json.loads((root / "results.json").read_text())
    hold, decisions = [], []
    for row in rows:
        path = root / row["trial"]
        d = json.loads((path / "result.json").read_text())
        a = np.load(path / "boundary.npz")["time_clearance_support_speed_left_right"]
        time, z, support, speed, left, right = a.T
        if row["rule"] == "continue":
            when = row["decision"]["time_s"]
            idx = np.searchsorted(time, when)
            assert abs(time[idx]-when) < 1e-8
            decision = dict(trial=row["trial"], time_s=when, actual_rise_mm=row["decision"]["actual_rise_m"]*1000,
                            support_weight_ratio=support[idx]/(9.81*row["mass_kg"]),
                            clearance_mm=z[idx]*1000, bilateral=bool(left[idx] and right[idx]),
                            load_delta_N=row["decision"]["late_load_N"])
            lift_start = next(p["start_s"] for p in d["phases"] if p["phase"] == "lift")
            obs = d["micro_lift"]["observations"]
            close_z = np.median([s[4] for s in obs if s[1] == "close" and lift_start-.10 <= s[0] < lift_start])
            fk = obs[-1]
            decision.update(fk_sample_offset_ms=1000*(fk[0]-when),
                            fk_rise_from_close_mm=1000*(fk[4]-close_z))
            if (path/"loads.npz").exists():
                loads = np.load(path/"loads.npz")
                decision["tension_N"] = loads["tension_N"][idx].tolist()
                decision["gamma_rgb"] = row["probe"].get("gamma_rgb")
            decisions.append(decision)
        if row["platform"] != "pickup_release_ok":
            continue
        phases = d["phases"]
        h = next(p for p in reversed(phases) if p["phase"] == "lift_hold")
        start = next(p["start_s"] for p in reversed(phases) if p["phase"] == "lift")
        dwell = (time > h["start_s"]) & (time <= h["end_s"])
        air = (z > .001) & (support == 0.) & (left > .5) & (right > .5)
        hold.append(dict(trial=row["trial"], duration_s=h["end_s"]-h["start_s"], steps=int(dwell.sum()),
                         speed_ge_20=int((dwell & (speed >= .02)).sum()),
                         clearance_le_1=int((dwell & (z <= .001)).sum()),
                         supported=int((dwell & (support != 0.)).sum()),
                         lost_finger=int((dwell & ((left <= .5) | (right <= .5))).sum()),
                         peak_speed_mm_s=1000*speed[dwell].max(), best_s=row["stable_s"],
                         best_ignoring_speed_s=longest(air & (time>start) & (time<=h["end_s"]), d["config"]["dt"])))
    if replay_loads:
        from run_yam_microlift import trial
        group = [r for r in rows if r["rule"] == "continue"]
        with TemporaryDirectory() as temp, ProcessPoolExecutor(max_workers=9) as pool:
            jobs = [(r["kind"],r["mass_kg"],"continue",Path(temp)/r["trial"],False,False,False) for r in group]
            for old, rerun, item in zip(group, pool.map(trial, jobs), decisions, strict=True):
                assert rerun["prefix"] == old["prefix"] and old["trial"] == item["trial"]
                path = Path(temp)/old["trial"]
                key = "time_clearance_support_speed_left_right"
                orig = np.load(root/old["trial"]/"boundary.npz")[key]
                np.testing.assert_array_equal(np.load(path/"boundary.npz")[key], orig)
                idx = np.searchsorted(orig[:,0], item["time_s"])
                item["tension_N"] = np.load(path/"loads.npz")["tension_N"][idx].tolist()
    return dict(hold=hold, decisions=decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-loads", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run, args.replay_loads)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
