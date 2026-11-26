"""Retrain C2 from the r1 packets and evaluate frozen picks on paired CPU Flex states."""
import argparse
from collections import Counter
from hashlib import sha256
import multiprocessing as mp
import os
from pathlib import Path
import tarfile

import mujoco
import numpy as np
import torch

from pe_physics import save_json
from rss_cpu_flex_c1c2 import CANDIDATES, DT, Job, STATES
from rss_cpu_flex_c1c2_policy import (CHECKPOINT_NAMES, MODES, SEED, branch, closure_metrics,
                                     fixed_grasp, load, matched_score_shift, packet,
                                     ranking_metrics, rows)
from rss_cpu_flex_c2_model import C2Model, Normalization, inputs, training_example

ARCHIVE = Path(__file__).resolve().parents[1] / "runs/2026-11-26/cpu-flex-policy-v3-c1c2-r1"
NAMES = list(CANDIDATES)
GATE_SPAN = .05
FIT = dict(seed=SEED, epochs=100, batch_size=32, patience=18, lr=.001, scorer_lr=.0003, decay=.01, support_weight=.3)


def prepare(work, existing_branches):
    work.mkdir(parents=True, exist_ok=True)
    for name, folders in (("c1_raw_packets", ("collect",)),
                          ("physical_eval_packets", ("closed", "matched", "branches"))):
        if all((work / folder).is_dir() for folder in folders):
            continue
        with tarfile.open(ARCHIVE / f"{name}.tar.gz", "r:gz") as archive:
            archive.extractall(work, filter="data")
    if existing_branches.is_file():
        with tarfile.open(existing_branches, "r:gz") as archive:
            archive.extractall(work, filter="data")


def verified_splits(work):
    summary = load(ARCHIVE / "c1_dataset_summary.json")
    indexed = rows(work / "collect")
    by_id = {r["initial_state_id"]: (p, r) for p, r in indexed}
    assignments = summary["initial_state_ids"]
    ids = [id_ for member_ids in assignments.values() for id_ in member_ids]
    if len(indexed) != 450 or len(set(ids)) != len(ids) or set(ids) != by_id.keys():
        raise ValueError("Archived initial-state assignment does not cover C1 exactly once")
    for key in ("episode_id", "probe_state_id", "seed"):
        if len({r[key] for _, r in indexed}) != len(indexed):
            raise ValueError(f"C1 identity repeated: {key}")
    if any((r["physics_validity"] is True) != (type(r["task_success"]) is bool) for _, r in indexed):
        raise ValueError("Archived validity and success labels disagree")
    splits = {}
    for split, member_ids in assignments.items():
        valid = [id_ for id_ in member_ids if by_id[id_][1]["physics_validity"] is True]
        if sorted(valid) != summary["valid_bce_ids"][split]:
            raise ValueError(f"C1 valid IDs do not match the archived {split} split")
        splits[split] = [training_example(*by_id[id_], NAMES) for id_ in valid]
    if [len(splits[k]) for k in ("train", "val", "test")] != [265, 56, 55]:
        raise ValueError("Expected valid split sizes 265/56/55")
    if sum(x[-1] for split in splits.values() for x in split) != 116:
        raise ValueError("C1 positive count differs from r1")
    return splits, [by_id[id_][1] for id_ in assignments["train"]]


def checked_origin(path, row):
    _, state = packet(path)
    if state.size == 0 or sha256(state.tobytes()).hexdigest() != row["post_try_sha256"]:
        raise ValueError(f"Invalid post-Try integration checkpoint: {path}")
    job = Job(row["state"], row["probe"], row["grasp"], row["replicate"], row["phase"])
    if job.seed != row["seed"] or job.id != row["initial_state_id"]:
        raise ValueError(f"Native fixture identity differs from r1: {path}")
    return inputs(path, row)


def matched_examples(work):
    matched = {r["state"]: (p, r) for p, r in rows(work / "matched")}
    with np.load(ARCHIVE / "score_example_initial_rgb.npz", allow_pickle=False) as arrays:
        rgb = arrays["rgb"]
        if set(matched) != set(STATES) or set(arrays["state"]) != set(STATES) or rgb.shape != (3, 240, 320, 3):
            raise ValueError("Three 320 × 240 matched RGB examples required")
        if not np.array_equal(rgb, np.broadcast_to(rgb[0], rgb.shape)):
            raise ValueError("Matched RGB pixels differ")
        rgb_hash = sha256(rgb[0].tobytes()).hexdigest()
    samples = [checked_origin(*matched[s]) for s in STATES]
    if any(not np.array_equal(x[k], samples[0][k]) for x in samples for k in (0, 2)):
        raise ValueError("Matched V0, Try action or candidate coordinates differ")
    entries = [dict(state=s, initial_state_id=matched[s][1]["initial_state_id"],
                    initial_mesh_sha256=matched[s][1]["initial_mesh_sha256"]) for s in STATES]
    if len({x["initial_mesh_sha256"] for x in entries}) != 1:
        raise ValueError("Matched initial mesh fingerprints differ")
    return samples, entries, rgb_hash


def score_examples(models, samples, entries):
    return [dict(entry, scores={mode: model.predict(sample) for mode, model in models.items()})
            for sample, entry in zip(samples, entries, strict=True)]


def enforce_gate(shift):
    if shift["Blind"]["max_across_state_probability_span"] != 0.:
        raise ValueError("Blind predictions differ on matched public inputs")
    informed = shift["Informed"]
    span = informed["max_across_state_probability_span"]
    if (span is None or not np.isfinite(span) or span < GATE_SPAN
            or len(set(informed["argmax_by_state"].values())) < 2):
        raise RuntimeError(f"Offline anti-collapse gate failed: {informed}")


def fit_models(out, work, width, dropout, scorer_lr, support_weight):
    fit = dict(FIT, scorer_lr=scorer_lr, support_weight=support_weight)
    splits, fixed_rows = verified_splits(work)
    samples, entries, rgb_hash = matched_examples(work)
    models, metrics = {}, {}
    fixed, bounds = fixed_grasp(fixed_rows)
    for mode in MODES[1:]:
        torch.manual_seed(SEED)
        norm = Normalization.fit(splits["train"], mode)
        model = C2Model(norm, width, dropout)
        metrics[mode] = model.fit(splits, **fit)
        metrics[mode]["active_history_channels"] = norm.indices.tolist()
        models[mode] = model
        print(mode, "epoch", metrics[mode]["best_epoch"], metrics[mode]["val"], flush=True)

    examples = score_examples(models, samples, entries)
    shift = matched_score_shift(examples)
    diagnostics = models["Informed"].diagnostics(splits["train"][:32], samples, support_weight)
    print("Matched score shift", shift, "gradient and embedding diagnostics", diagnostics, flush=True)
    result = dict(models=metrics, fixed_grasp=fixed, fixed_selection_lower_bound=bounds,
                  selector_selection_split="C1 train only", training=fit, width=width, z_dim=32,
                  dropout=dropout, normalization_epsilon=1e-6, info_mode="v0_only",
                  matched_rgb_sha256=rgb_hash, score_examples=examples, matched_score_shift=shift,
                  diagnostics=diagnostics)
    try:
        enforce_gate(shift)
    except (RuntimeError, ValueError):
        save_json(out / "offline-gate-failure.json", result)
        raise

    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for mode, model in models.items():
        metrics[mode]["test"] = model.evaluate(splits["test"], FIT["batch_size"])
        file = checkpoints / CHECKPOINT_NAMES[mode]
        torch.save(dict(width=width, dropout=dropout, state=model.state_dict()), file)
        metrics[mode]["checkpoint_sha256"] = sha256(file.read_bytes()).hexdigest()
    save_json(out / "c2_metrics.json", result)


def load_models(out, metrics):
    models = {}
    for mode, name in CHECKPOINT_NAMES.items():
        file = out / "checkpoints" / name
        if sha256(file.read_bytes()).hexdigest() != metrics["models"][mode]["checkpoint_sha256"]:
            raise ValueError(f"C2 checkpoint fingerprint differs: {file}")
        weights = torch.load(file, map_location="cpu", weights_only=True)
        model = C2Model(Normalization(weights["state"]["normalization.indices"]),
                        weights["width"], weights["dropout"])
        model.load_state_dict(weights["state"])
        models[mode] = model.eval()
    return models


def check_branch(work, row, grasp):
    target = work / "branches" / f"{row['initial_state_id']}_{grasp}.json"
    if not target.exists():
        return False
    cached = load(target)
    for key, expected in dict(initial_state_id=row["initial_state_id"],
                              probe_state_id=row["probe_state_id"], grasp=grasp,
                              post_try_sha256=row["post_try_sha256"]).items():
        if cached[key] != expected:
            raise ValueError(f"Cached branch {key} mismatch: {target}")
    if not target.with_suffix(".npz").is_file():
        raise ValueError(f"Missing physical trace: {target}")
    if (cached["physics_validity"] is True) != (type(cached["task_success"]) is bool):
        raise ValueError(f"Cached branch validity disagrees with the label: {target}")
    return True


def evaluate(out, work):
    metrics = load(out / "c2_metrics.json")
    models = load_models(out, metrics)
    samples, entries, rgb_hash = matched_examples(work)
    examples = score_examples(models, samples, entries)
    enforce_gate(matched_score_shift(examples))
    if examples != metrics["score_examples"] or rgb_hash != metrics["matched_rgb_sha256"]:
        raise ValueError("Reloaded frozen weights do not reproduce the offline gate")

    prior = load(ARCHIVE / "branch_plan.json")
    old_seeds = {r["seed"] for _, r in rows(work / "collect")}
    heldout = rows(work / "closed")
    if (Counter(r["state"] for _, r in heldout) != dict.fromkeys(STATES, 15)
            or len({r["seed"] for _, r in heldout}) != 45
            or any(r["seed"] in old_seeds for _, r in heldout)):
        raise ValueError("Expected three sets of 15 independent physical holdouts")
    origins, pending = {}, []
    cached_ranking = cached_closed = 0
    splits = load(ARCHIVE / "c1_dataset_summary.json")["initial_state_ids"]
    for split, ids in prior["ranking_states"].items():
        eligible = set(splits[split])
        for id_ in ids:
            old = prior["origins"][id_]
            path = work / old["origin"]
            row = load(path)
            if id_ not in eligible or row["initial_state_id"] != id_:
                raise ValueError(f"Archived ranking origin is outside {split}: {id_}")
            sample = checked_origin(path, row)
            for grasp in NAMES:
                if grasp == row["grasp"]:
                    continue
                if not check_branch(work, row, grasp):
                    raise ValueError(f"Missing archived rank label: {id_} {grasp}")
                cached_ranking += 1
            origins[id_] = dict(origin=old["origin"], purpose="ranking",
                                scores={m: model.predict(sample) for m, model in models.items()})

    for path, row in heldout:
        id_ = row["initial_state_id"]
        origin = str(path.relative_to(work))
        if prior["origins"][id_]["origin"] != origin or prior["origins"][id_]["purpose"] != "closed":
            raise ValueError(f"Closure origin differs from r1: {id_}")
        sample = checked_origin(path, row)
        scores = {mode: model.predict(sample) for mode, model in models.items()}
        choices = {mode: NAMES[int(np.argmax(values))] for mode, values in scores.items()}
        choices["Best Fixed"] = metrics["fixed_grasp"]
        branches = sorted(set(choices.values()))
        origins[id_] = dict(origin=origin, purpose="closed", scores=scores, choices=choices, branches=branches)
        for grasp in branches:
            if not check_branch(work, row, grasp):
                pending.append((work, origin, grasp))
            elif grasp in prior["origins"][id_]["branches"]:
                cached_closed += 1
    plan = dict(origins=origins, ranking_states=prior["ranking_states"], score_examples=examples)
    save_json(out / "branch_plan.json", plan)
    print(f"Offline gate rechecked from disk. Physical branches: {cached_closed} cached, "
          f"{len(pending)} pending; {cached_ranking} cached validation/test ranking branches.", flush=True)
    if pending:
        if any(os.environ.get(name) != "1" for name in
               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
            raise ValueError("Each of the 22 CPU workers must have its thread counts set to one")
        np.random.default_rng(SEED).shuffle(pending)
        with mp.get_context("spawn").Pool(22) as pool:
            for name, status in pool.imap_unordered(branch, pending):
                print(name, status, flush=True)
    ranking_metrics(work, plan, metrics)
    save_json(out / "c2_metrics.json", metrics)
    closure = closure_metrics(work, plan)
    new = []
    for path, row in heldout:
        id_ = row["initial_state_id"]
        for grasp in origins[id_]["branches"]:
            if not check_branch(work, row, grasp):
                raise ValueError(f"Missing planned closed-loop result: {id_} {grasp}")
            if grasp not in prior["origins"][id_]["branches"]:
                new.append(f"{id_}_{grasp}")
    if new:
        pending_archive = out / "new_physical_branches.tar.gz.tmp"
        with tarfile.open(pending_archive, "w:gz") as archive:
            for name in sorted(new):
                for suffix in (".json", ".npz"):
                    rel = f"branches/{name}{suffix}"
                    archive.add(work / rel, arcname=rel)
        pending_archive.replace(out / "new_physical_branches.tar.gz")
    closure["physical_provenance"] = dict(
        source_run=ARCHIVE.name, closed_origins=len(heldout), distinct_cached_closed_branches=cached_closed,
        distinct_new_closed_branches=len(new), candidate_branch_packets=sorted(new),
        cached_ranking_branches=cached_ranking, workers=22, threads_per_worker=1,
        engine=mujoco.__version__, timestep_s=DT)
    save_json(out / "closed_loop_eval.json", closure)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--scorer-lr", type=float, default=FIT["scorer_lr"])
    parser.add_argument("--support-weight", type=float, default=FIT["support_weight"])
    args = parser.parse_args()
    if mujoco.__version__ != "4.0.1" or DT != .00005 or torch.version.cuda is not None:
        raise RuntimeError("C2 requires MuJoCo 4.0.1 at 50 µs and PyTorch CPU")
    torch.set_num_threads(1)
    work = args.output / "work"
    prepare(work, args.output / "new_physical_branches.tar.gz")
    if args.stage == "train":
        fit_models(args.output, work, args.width, args.dropout, args.scorer_lr,
                   args.support_weight)
    else:
        evaluate(args.output, work)


if __name__ == "__main__":
    main()
