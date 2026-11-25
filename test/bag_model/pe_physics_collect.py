"""Consolidate completed pilot outputs without overwriting or hiding any source trial."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from pe_physics import save_json


def collect(paths, expected_backend):
    manifests, configs, rows, arrays = [], [], [], {}
    identities = set()
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["backend"] != expected_backend or manifest["smoke"]:
            raise ValueError(f"{path} is not a full {expected_backend} pilot run")
        manifests.append(dict(run=path.name, **manifest))
        configs.append(dict(run=path.name, **json.loads((path / f"{expected_backend}_config.json").read_text())))
        with np.load(path / "raw_trajectories.npz") as trace:
            for row in json.loads((path / "summary.json").read_text()):
                case = row["case"]
                if case in identities:
                    raise ValueError(f"Duplicate {expected_backend} conditions in primary evidence: {case}")
                identities.add(case)
                row = dict(backend=expected_backend, run=path.name, **row)
                if case == "B01" and f"{case}/wrench" in trace.files:
                    time = trace[f"{case}/time"]
                    tail = time >= time[-1] - .10
                    weight = 9.81 * trace[f"{case}/masses"].sum()
                    reaction = -trace[f"{case}/wrench"][tail, -2, 2].mean()
                    row.update(late_table_reaction_N=float(reaction),
                               late_static_support_error_fraction=float(abs(reaction - weight) / weight))
                if case.startswith("B06") and f"{case}/fixture_target" in trace.files:
                    start = trace[f"{case}/fixture_target"][0, 0, 2]
                    height = trace[f"{case}/fixtures"][:, 0, 2]
                    force = trace[f"{case}/wrench"][:, 0, 2]
                    active = np.flatnonzero(force > .05)
                    row["contact_onset_mm"] = 1000 * float(start - height[active[0]]) if len(active) else None
                    if float(force.max()) > 5. and "probe_force_limit_exceeded" not in row["violations"]:
                        row["violations"] = [*row["violations"], "probe_force_limit_exceeded"]
                        if row["status"] == "pilot_pass":
                            row["status"] = "physics_gate_failure"
                rows.append(row)
                keys = (key for key in trace.files if key.startswith(f"{case}/"))
                arrays.update({key: trace[key] for key in keys})
    return manifests, configs, rows, arrays


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--flex", nargs="+", type=Path, required=True)
    ap.add_argument("--vbd", nargs="+", type=Path, default=[])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("manifest.json", "raw_trajectories.npz", "summary.json")):
        raise SystemExit("Primary evidence already exists in the destination")
    backends, results, raw, common = {}, [], {}, {}
    for backend, inputs in (("flex", args.flex), ("vbd", args.vbd)):
        if not inputs:
            continue
        manifests, configs, trials, arrays = collect(inputs, backend)
        backends[backend] = manifests
        results.extend(trials)
        raw.update({f"{backend}/{k}": v for k, v in arrays.items()})
        common[backend] = arrays
        save_json(args.output / f"{backend}_config.json", configs)
    topology = {}
    if len(common) == 2:
        for key in (k for k in common["flex"] if k.endswith("/reference") and k in common["vbd"]):
            tag = key.split("/")[0]
            geometry = ("reference", "current", "faces", "masses")
            topology[tag] = {k: bool(np.array_equal(common["flex"][f"{tag}/{k}"], common["vbd"][f"{tag}/{k}"]))
                             for k in geometry}
    save_json(args.output / "manifest.json",
              dict(assembled_at=datetime.now().astimezone().isoformat(), backends=backends,
                   pairwise_input_equality=topology))
    save_json(args.output / "summary.json", results)
    np.savez_compressed(args.output / "raw_trajectories.npz", **raw)
    excluded = {"config", "gauge_response", "backend_config"}
    fields = list(dict.fromkeys(key for row in results for key in row if key not in excluded))
    with (args.output / "trial_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
