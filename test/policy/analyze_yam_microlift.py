"""Verify matched trajectories and summarize a completed micro-lift matrix."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import numpy as np


def analyze(root):
    rows = json.loads((root / "results.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(rows) == len(set(manifest["jobs"])) == 27
    assert sorted(manifest["jobs"]) == sorted(r["trial"] for r in rows)
    groups = defaultdict(dict)
    for row in rows:
        groups[row["kind"], row["mass_kg"]][row["rule"]] = row
    checks, probe = [], []
    for key, group in groups.items():
        assert set(group) == {"continue", "retry", "feedback"}
        dirs = {rule: root / r["trial"] for rule, r in group.items()}
        raw = {k: json.loads((p / "result.json").read_text()) for k, p in dirs.items()}
        anchor = raw["continue"]
        decision = anchor["micro_lift"]["decision"]["time_s"]
        streams = {}
        for rule, item in raw.items():
            assert item["config"] == anchor["config"] and not item["warnings"]
            assert item["source_sha256"] == anchor["source_sha256"]
            assert (dirs[rule] / "robot_assets.json").read_bytes() == (dirs["continue"] / "robot_assets.json").read_bytes()
            assert item["micro_lift"]["prefix_sha256"] == anchor["micro_lift"]["prefix_sha256"]
            assert item["micro_lift"]["observations"] == anchor["micro_lift"]["observations"]
            with np.load(dirs[rule] / "boundary.npz") as tape:
                streams[rule] = tape["time_clearance_support_speed_left_right"]
        ref = streams["continue"]
        for array in streams.values():
            np.testing.assert_array_equal(array[array[:, 0] <= decision], ref[ref[:, 0] <= decision])
        actual = "retry" if group["feedback"]["retry"] else "continue"
        np.testing.assert_array_equal(streams["feedback"], streams[actual])
        assert (dirs["feedback"] / "trace.json").read_bytes() == (dirs[actual] / "trace.json").read_bytes()
        checks.append(group["feedback"]["trial"])
        _, z, support, _, left, right = ref[ref[:, 0] <= decision][-1]
        d = anchor["micro_lift"]["decision"]
        probe.append(dict(kind=key[0], mass_kg=key[1], decision_s=decision,
                          actual_rise_mm=1000*d["actual_rise_m"], clearance_mm=1000*z,
                          bilateral=bool(left and right), support_weight_fraction=support/(9.81*key[1]),
                          late_load_N=d["late_load_N"], decision=actual))

    counts = {}
    for rule in ("continue", "feedback", "retry"):
        selected = [r for r in rows if r["rule"] == rule]
        counts[rule] = dict(n=len(selected), success=sum(r["success"] for r in selected),
                            pickup_release=sum(r["platform"] == "pickup_release_ok" for r in selected),
                            drop=sum(r["dropped"] for r in selected), over=sum(r["over"] for r in selected),
                            regrasp=sum(r["retry"] for r in selected), mean_sim_s=mean(r["sim_s"] for r in selected),
                            best_stable_s=max(r["stable_s"] for r in selected))
    return dict(counts=counts, probes=probe, verified_pairs=checks,
                objects_with_either_fixed_action_success=sum(
                    g["retry"]["success"] or g["continue"]["success"] for g in groups.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(analyze(parser.parse_args().run), indent=2, allow_nan=False))
