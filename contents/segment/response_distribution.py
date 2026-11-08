"""Paper R3/RJ response coverage, evaluated with the immutable v1 source package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "top_grasp_v1_2"


def save(path, value):
    def finite(v):
        if isinstance(v, dict):
            return {k: finite(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [finite(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, (np.integer, np.bool_)):
            return v.item()
        return v
    with Path(path).open("x") as f:
        json.dump(finite(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def initialize(reference):
    sys.path.insert(0, str(reference))
    # Workers use frozen imports before any active model module can be loaded.
    global model, sampling, protocols, adapter
    import model
    import sampling
    import protocols
    import mj_adapter as adapter
    observe_protocols(protocols)


def observe_protocols(protocols):
    """Retain Runner evidence which frozen R2 omits from its early-failure return.

    This only retains a reference to the original Runner, and leaves its stepping,
    stopping rules and diagnostics unchanged.
    """
    original = protocols.Runner
    class ObservedRunner(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            protocols.measurement_runner = self
    protocols.Runner = ObservedRunner


def measure(job):
    sampler, category, index, seed, destination = job
    destination = Path(destination)
    destination.mkdir()
    os.chdir(destination)
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, ("hard", "bendable", "compressible").index(category)]))
    started = time.perf_counter()
    row = dict(sampler=sampler, category=category, index=index, seed=seed, errors={})
    try:
        if sampler == "RJ":
            desc = sampling.sample_rj(rng, category)
            k = desc.kappa
        else:
            k = sampling.sample_kappa(rng, category)
            desc = model.build(k, omega_max=adapter.OMEGA_MAX)
        row.update(kappa=k.to_dict(), internal=desc.internal)
        for test, execute in (("r1y", lambda: protocols.r1(k, "y", desc=desc)),
                              ("r2", lambda: protocols.r2(k, "edge", desc=desc))):
            try:
                row[test] = execute()
                runner = protocols.measurement_runner
                if row[test].get('fail'):
                    row[test]['failure_state'] = runner.failure_state
                    row[test]['diagnostics'] = runner.diagnostics
            except Exception:
                row["errors"][test] = traceback.format_exc()
        a, b = row.get("r1y", {}), row.get("r2", {})
        row["valid_r1y"] = bool(not a.get("fail") and a.get("stiffness", 0) > 0
                                 and np.isfinite(a.get("stiffness", np.nan)) and not a.get("lower_bound", True))
        row["valid_r2"] = bool(not b.get("fail") and b.get("droop", 0) > 0
                                and np.isfinite(b.get("droop", np.nan)) and not b.get("dropped", True)
                                and not b.get("table_touch", True) and abs(b.get("pad_fz_ratio", 0) - 1) <= .05)
        row["valid_joint"] = row["valid_r1y"] and row["valid_r2"]
        row["confirmed_failure"] = bool(a.get("fail") or b.get("fail"))
    except Exception:
        row["errors"]["build"] = traceback.format_exc()
    row["wall_s"] = time.perf_counter() - started
    save(destination / "result.json", row)
    return row


def summarize(rows):
    summary = {}
    for sampler in ("R3", "RJ"):
        for category in ("hard", "bendable", "compressible"):
            part = [r for r in rows if r["sampler"] == sampler and r["category"] == category]
            def distribution(test, field, validity):
                values = [r[test][field] for r in part if r.get(validity)]
                return dict(n=len(values), quantiles=dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                            np.quantile(values, (0, .05, .25, .5, .75, .95, 1)).tolist())) if values else None)
            summary[f"{sampler}:{category}"] = dict(
                total=len(part), valid_r1y=sum(r.get("valid_r1y", False) for r in part),
                valid_r2=sum(r.get("valid_r2", False) for r in part),
                valid_joint=sum(r.get("valid_joint", False) for r in part),
                confirmed_failures=sum(r.get("confirmed_failure", False) for r in part),
                execution_errors=sum(bool(r["errors"]) for r in part),
                stiffness_lower_bounds=sum(r.get("r1y", {}).get("lower_bound", False) for r in part),
                squeeze_y_N_per_mm=distribution("r1y", "stiffness", "valid_r1y"),
                droop=distribution("r2", "droop", "valid_r2"))
    return summary


def paired_summary(rows):
    """Matched comparisons exclude both sides when either simulation fails."""
    result = {}
    for category in ('hard', 'bendable', 'compressible'):
        left = {r['index']: r for r in rows if r['category'] == category and r['sampler'] == 'R3'}
        right = {r['index']: r for r in rows if r['category'] == category and r['sampler'] == 'RJ'}
        pairs = [(a, right[i]) for i, a in left.items()]
        excluded = [a['index'] for a, b in pairs if a.get('confirmed_failure') or b.get('confirmed_failure')
                    or a['errors'] or b['errors']]
        metrics = {}
        for test, field, flag in (('r1y', 'stiffness', 'valid_r1y'), ('r2', 'droop', 'valid_r2')):
            valid = [(a, b) for a, b in pairs if a['index'] not in excluded and a.get(flag) and b.get(flag)]
            metrics[field] = dict(n=len(valid), rows=[dict(index=a['index'], R3=a[test][field], RJ=b[test][field])
                                                       for a, b in valid])
        result[category] = dict(total_pairs=len(pairs), excluded_simulation_or_execution=excluded, metrics=metrics)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--count", type=int, default=200, help="objects per sampler")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = (BASE / args.run_id).resolve()
    out.mkdir()
    reference = (BASE / "reference_v1/contents/segment").resolve()
    tag_path = ROOT / "results/freeze_v1_newton/release_v1/v1.json"
    tag = json.loads(tag_path.read_text())
    assert hashlib.sha256((tag_path.parent / "v1.tar.gz").read_bytes()).hexdigest() == tag["archive_sha256"]
    for name, expected in tag["source_sha256"].items():
        assert hashlib.sha256((reference / name).read_bytes()).hexdigest() == expected, name
    save(out / "metadata.json", dict(v1_tag=tag, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     count_per_sampler=args.count, seed=args.seed, workers=args.workers))
    categories = ("hard", "bendable", "compressible")
    jobs = [(sampler, categories[i % 3], i // 3, args.seed, str(out / f"{sampler}_{i:03d}"))
            for i in range(args.count) for sampler in ("R3", "RJ")]
    rows = []
    with Pool(args.workers, initializer=initialize, initargs=(reference,)) as pool:
        for row in pool.imap_unordered(measure, jobs):
            rows.append(row)
            print(f"{row['sampler']} {row['category']} {row['index']}: valid={row.get('valid_r1y')}/{row.get('valid_r2')} "
                  f"sim_fail={row.get('confirmed_failure')} error={bool(row['errors'])} {row['wall_s']:.1f}s", flush=True)
    rows.sort(key=lambda r: (r["category"], r["index"], r["sampler"]))
    # Paired geometry is checked from actual draws, rather than inferred from equal seeds.
    for category in categories:
        a = [r for r in rows if r["category"] == category and r["sampler"] == "R3"]
        b = [r for r in rows if r["category"] == category and r["sampler"] == "RJ"]
        for left, right in zip(a, b):
            assert left.get("kappa") == right.get("kappa"), (category, left["index"])
    save(out / "summary.json", dict(groups=summarize(rows), paired_groups=paired_summary(rows), rows=rows))


if __name__ == "__main__":
    main()
