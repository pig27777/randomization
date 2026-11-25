"""Policy V3 training, checkpoint-paired physical branches and publication outputs."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

from pe_physics import save_json
from rss_cpu_flex_c1c2 import CANDIDATES, FlexMacro, HOVER, Job, PROBE_END, STATES, TRIES, run_grasp

MODES = ("Best Fixed", "Blind", "Informed w/o Effort", "Informed")
CHECKPOINT_NAMES = {"Blind": "blind.pt", "Informed w/o Effort": "informed-no-effort.pt",
                    "Informed": "informed.pt"}
SEED = 60931
TRIALS = 450


def packet(path):
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        h = arrays["history"]
        assert (arrays["padding_mask"][:len(h)] == False).all()
        assert arrays["padding_mask"][len(h):].all()
        return h, arrays["post_try_state"]


def load(path):
    return json.loads(path.read_text())


def rows(directory):
    return [(p, load(p)) for p in sorted(directory.glob("*-*.json"))]


def action_features(probe, clearance_offset_m):
    x, span = TRIES[probe]
    return np.r_[x / .10, 0., .045 / .10, span / .20, clearance_offset_m / .01, np.zeros(8)]


def v3_frames(h):
    """Pack dimensionless fixture observations into the V3 Cartesian and effort fields."""
    raw = np.zeros((len(h), 54), dtype=np.float32)
    raw[:, 24:28] = h[:, 9:13]
    raw[:, 34:37] = (h[:, :3] - HOVER) / .10
    raw[:, 37:40] = h[:, 6:9] / .01
    raw[:, 40:43] = h[:, 3:6] / .25
    raw[:, 46:49] = (h[:, 13:16] - HOVER) / .10
    raw[:, 53] = h[:, 16] / PROBE_END
    return raw


def v3_imports(root):
    sys.path.insert(0, str((root / "source").resolve()))
    import torch
    from policy_v3 import data, geometry, history, training

    torch.set_num_threads(1)
    return torch, data, geometry, history, training


def record(path, row, data, geometry):
    h, _ = packet(path)
    g = None if row["grasp"] is None else row["candidates"][list(CANDIDATES).index(row["grasp"])]
    return data.EpisodeRecord(
        episode_id=row["episode_id"], initial_state_id=row["initial_state_id"],
        probe_state_id=row["probe_state_id"],
        v0=geometry.VisualObservation(**row["v0"]),
        v1=geometry.VisualObservation(**row["v1"]) if row["v1"] else None,
        a_try=action_features(row["probe"], row["probe_summary"].get("clearance_offset_m", 0.)),
        H_try=v3_frames(h), valid_len=len(h),
        grasp_parameters_world=g, info_mode="v0_only", task_success=row["task_success"],
        physics_validity=row["physics_validity"], invalid_stage=row["invalid_stage"],
        failure_reason=",".join(row["invalid_reasons"]) or None, randomization_params={},
        policy_version=0, timestamps=h[:, -1], trajectory_world=h[:, :3], probe_stages=["try"] * len(h),
        probe_info={"candidate_info_mode": "v0_only"}, task_summary={})


def counts(entries):
    valid = [r for r in entries if r["physics_validity"]]
    positives = sum(r["task_success"] is True for r in valid)
    return dict(trials=len(entries), valid=len(valid), positives=positives,
                physics_invalid_rate=1-len(valid)/len(entries) if entries else None,
                positive_rate=positives/len(valid) if valid else None,
                invalid_reasons=dict(Counter(reason for r in entries for reason in r["invalid_reasons"])))


def fixed_grasp(entries):
    # Compare state-balanced lower bounds, preserving invalid outcomes as unknown.
    lower = {}
    for g in CANDIDATES:
        cells = [[r for r in entries if r["state"] == s and r["grasp"] == g] for s in STATES]
        lower[g] = float(np.mean([sum(r["task_success"] is True for r in c)/len(c) if c else 0.
                                  for c in cells]))
    return max(lower, key=lower.get), lower


def profiles(training, geometry):
    return {mode: training.InformationProfile(mode=geometry.InfoMode.V0_ONLY,
                                               history="none" if mode == "Blind" else "all",
                                               effort=mode != "Informed w/o Effort")
            for mode in MODES[1:]}


def train(root):
    torch, data, geometry, _, training = v3_imports(root)
    indexed = rows(root / "collect")
    if len(indexed) != TRIALS:
        raise ValueError(f"Expected {TRIALS} complete collection attempts, found {len(indexed)}")
    by_id = {r["initial_state_id"]: (p, r) for p, r in indexed}
    records = [record(p, r, data, geometry) for p, r in indexed]
    splits = data.GroupSplit.partition(records, seed=SEED)
    data.assert_disjoint(splits)
    if len(by_id) != TRIALS or len({r.probe_state_id for r in records}) != TRIALS:
        raise ValueError("Each formal collection row needs an independent initial and probe state ID")
    split_rows = {split: [by_id[r.initial_state_id][1] for r in group] for split, group in splits.items()}
    split_ids = {split: [r.initial_state_id for r in group] for split, group in splits.items()}
    ids = {split: sorted(r.initial_state_id for r in data.ReplayBuffer.bce_records(group))
           for split, group in splits.items()}
    if any(not by_id[id_][1]["physics_validity"] for eligible in ids.values() for id_ in eligible):
        raise ValueError("Invalid physics outcome in the BCE replay")
    fixed, bounds = fixed_grasp(split_rows["train"])
    summary = dict(**counts([r for p, r in indexed]), candidates=CANDIDATES,
                   splits={key: counts(value) for key, value in split_rows.items()},
                   initial_state_ids=split_ids, valid_bce_ids=ids, group_leakage=False,
                   states={s: counts([r for _, r in indexed if r["state"] == s]) for s in STATES},
                   records="c1_records.jsonl.gz", raw_packets="collect/")
    save_json(root / "c1_dataset_summary.json", summary)
    archive = root / "c1_records.jsonl.gz"
    if archive.exists():
        previous = data.read_records(archive)
        if len(previous) != len(records) or any(a.to_dict() != b.to_dict() for a, b in zip(previous, records)):
            raise ValueError("Existing C1 archive differs from the episode packets")
    else:
        data.write_records(archive, records)
    result = dict(models={}, fixed_grasp=fixed, fixed_selection_lower_bound=bounds,
                  selector_selection_split="C1 train only", split_seed=SEED, epochs=22, batch_size=32)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for mode, profile in profiles(training, geometry).items():
        torch.manual_seed(SEED)
        learner = training.SupervisedTrainEntry(device="cpu", profile=profile, seed=SEED)
        result["models"][mode] = learner.fit(splits, epochs=22, batch_size=32)
        torch.save(dict(encoder=learner.encoder.state_dict(), scorer=learner.scorer.state_dict()),
                   checkpoints / CHECKPOINT_NAMES[mode])
        print(mode, result["models"][mode]["val"], flush=True)
    save_json(root / "c2_metrics.json", result)


def load_scorers(root):
    torch, _, geometry, history, training = v3_imports(root)
    models = {}
    for mode, profile in profiles(training, geometry).items():
        learner = training.SupervisedTrainEntry(device="cpu", profile=profile)
        ckpt = torch.load(root / "checkpoints" / CHECKPOINT_NAMES[mode], map_location="cpu", weights_only=True)
        learner.encoder.load_state_dict(ckpt["encoder"])
        learner.scorer.load_state_dict(ckpt["scorer"])
        learner.encoder.eval()
        learner.scorer.eval()
        models[mode] = learner.encoder, learner.scorer, profile
    return models, geometry, history, training


def predicted(path, row, models, geometry, history, training):
    h, checkpoint = packet(path)
    if not len(checkpoint):
        return {}
    context = geometry.PublicContext(geometry.InfoMode.V0_ONLY,
                                     geometry.VisualObservation(**row["v0"]), lambda: 1/0,
                                     action_features(row["probe"], row["probe_summary"].get("clearance_offset_m", 0.)))
    h = history.encoder_frames(v3_frames(h))
    gs = np.asarray(row["candidates"])
    return {mode: training.score_candidates(encoder, scorer, context, h, gs, profile).tolist()
            for mode, (encoder, scorer, profile) in models.items()}


def plan(root):
    models, geometry, history, training = load_scorers(root)
    summary = load(root / "c1_dataset_summary.json")
    fixed = load(root / "c2_metrics.json")["fixed_grasp"]
    by_id = {r["initial_state_id"]: (p, r) for p, r in rows(root / "collect")}
    origins = {}
    ranking = {}
    for split in ("val", "test"):
        # Uniform deterministic first ID by split/state, conditional only on a completed Try.
        ranking[split] = []
        for state in STATES:
            candidates = sorted(id_ for id_ in summary["initial_state_ids"][split]
                                if by_id[id_][1]["state"] == state and by_id[id_][1]["post_try_sha256"])
            ranking[split].extend(candidates[:2])
    used = set().union(*(set(ids) for ids in ranking.values()))
    for id_ in sorted(used):
        p, r = by_id[id_]
        origins[id_] = dict(origin=str(p.relative_to(root)), purpose="ranking",
                            scores=predicted(p, r, models, geometry, history, training),
                            branches=[g for g in CANDIDATES if g != r["grasp"]])
    closure, matched = rows(root / "closed"), rows(root / "matched")
    if len(closure) != 45 or len(matched) != 3:
        raise ValueError("Closed loop requires 45 fresh states and 3 matched score traces")
    old_seeds = {r["seed"] for _, r in by_id.values()}
    seeds = {r["seed"] for _, r in closure+matched}
    assert len(seeds) == 46 and not (seeds & old_seeds)
    for p, r in closure:
        scores = predicted(p, r, models, geometry, history, training)
        actions = {mode: list(CANDIDATES)[int(np.argmax(values))] for mode, values in scores.items()}
        if scores:
            actions["Best Fixed"] = fixed
        origins[r["initial_state_id"]] = dict(origin=str(p.relative_to(root)), purpose="closed",
                                                scores=scores, choices=actions,
                                                branches=sorted(set(actions.values())))
    plot = []
    blind = []
    for p, r in matched:
        score = predicted(p, r, models, geometry, history, training)
        if score:
            blind.append(score["Blind"])
        plot.append(dict(state=r["state"], initial_state_id=r["initial_state_id"], scores=score,
                         initial_mesh_sha256=r["initial_mesh_sha256"]))
    if len(blind) == 3:
        assert np.array_equal(blind, np.broadcast_to(blind[0], np.shape(blind)))
    assert len({r["initial_mesh_sha256"] for r in plot}) == 1
    save_json(root / "branch_plan.json", dict(origins=origins, ranking_states=ranking, score_examples=plot))
    print("Frozen branch plan", sum(len(r["branches"]) for r in origins.values()), "distinct executions", flush=True)


def branch(args):
    root, origin, grasp_name = args
    row = load(root / origin)
    target = root / "branches" / f"{row['initial_state_id']}_{grasp_name}.json"
    if target.exists():
        return target.stem, "cached"
    _, state = packet(root / origin)
    assert sha256(state.tobytes()).hexdigest() == row["post_try_sha256"]
    job = Job(row["state"], row["probe"], grasp_name, row["replicate"], row["phase"])
    b = FlexMacro(job)
    b.restore(state)
    assert np.array_equal(state, b.integration_state())
    candidate = np.asarray(row["candidates"])[list(CANDIDATES).index(grasp_name)]
    result, trace, checks = run_grasp(b, candidate)
    result.update(initial_state_id=row["initial_state_id"], probe_state_id=row["probe_state_id"],
                  grasp=grasp_name, post_try_sha256=row["post_try_sha256"])
    np.savez_compressed(target.with_suffix(".npz"), grasp_trajectory=trace, grasp_audit=checks)
    pending = target.with_suffix(".tmp")
    save_json(pending, result)
    pending.replace(target)
    return target.stem, f"{result['task_success']} {result['invalid_reasons']}"


def branch_all(root, workers):
    p = load(root / "branch_plan.json")
    (root / "branches").mkdir(exist_ok=True)
    # Interleave all branches so the common worker pool handles C2 ranking and closure.
    tasks = [(root, r["origin"], g) for r in p["origins"].values() for g in r["branches"]]
    np.random.default_rng(SEED).shuffle(tasks)
    with mp.get_context("spawn").Pool(workers) as pool:
        for name, status in pool.imap_unordered(branch, tasks):
            print(name, status, flush=True)


def result_for(root, r, g):
    if r.get("grasp") == g:
        return {key: r[key] for key in ("task_success", "physics_validity", "invalid_stage", "invalid_reasons",
                                        "post_try_sha256")}
    result = load(root / "branches" / f"{r['initial_state_id']}_{g}.json")
    if result["post_try_sha256"] != r["post_try_sha256"]:
        raise ValueError("Physical branch did not start at the declared post-Try checkpoint")
    return result


def ranking_metrics(root, p, metrics):
    for mode in MODES[1:]:
        for split, ids in p["ranking_states"].items():
            eligible = wins = top_known = top_wins = 0
            for id_ in ids:
                item = p["origins"][id_]
                r = load(root / item["origin"])
                outcomes = np.array([result_for(root, r, g)["task_success"] for g in CANDIDATES], dtype=float)
                finite = np.isfinite(outcomes)
                score = np.array(item["scores"][mode])
                if finite.any() and np.any(outcomes == 1) and np.any(outcomes == 0):
                    eligible += 1
                    wins += int(outcomes[np.argmax(np.where(finite, score, -np.inf))])
                top = outcomes[score.argmax()]
                if np.isfinite(top):
                    top_known += 1
                    top_wins += int(top)
            metrics["models"][mode][split]["top1"] = dict(
                attempted_groups=len(ids), rankable_groups=eligible, correct=wins,
                accuracy=wins/eligible if eligible else None, full_pool_top1_valid=top_known,
                full_pool_top1_success=top_wins,
                full_pool_top1_bounds=[top_wins/len(ids), (top_wins+len(ids)-top_known)/len(ids)] if ids else None)


def estimate(y):
    valid = int(np.isfinite(y).sum())
    success = int(np.nansum(y))
    return dict(success=success, valid=valid, attempts=len(y), rate=success/valid if valid else None,
                physics_invalid_rate=1-valid/len(y), unknown_bounds=[success/len(y),
                                                   (success+len(y)-valid)/len(y)])


def adaptation(closed):
    modes = {}
    for mode in MODES:
        densities = {}
        for state in STATES:
            members = [r for r in closed if r["state"] == state and mode in r["choices"]]
            ids = [r["choices"][mode] for r in members]
            density = {g: ids.count(g)/len(ids) if ids else 0. for g in CANDIDATES}
            mean = np.mean([CANDIDATES[g][0] for g in ids], axis=0).tolist() if ids else None
            different = sum(r["choices"][mode] != r["choices"]["Blind"] for r in members)
            densities[state] = dict(n=len(ids), density=density, mean_bag_xyz_m=mean,
                                    different_from_blind=different/len(ids) if ids else None)
        tv = {}
        for a, b in combinations(STATES, 2):
            da, db = densities[a], densities[b]
            tv[f"{a}:{b}"] = .5 * sum(abs(da["density"][g] - db["density"][g])
                                      for g in CANDIDATES) if da["n"] and db["n"] else None
        modes[mode] = dict(states=densities, cross_state_total_variation=tv)
    return modes


def matched_score_shift(examples):
    result = {}
    for mode in MODES[1:]:
        available = [row for row in examples if mode in row["scores"]]
        scores = np.asarray([row["scores"][mode] for row in available])
        result[mode] = dict(examples=len(available),
                            max_across_state_probability_span=float(np.ptp(scores, axis=0).max())
                            if len(available) == len(STATES) else None,
                            argmax_by_state={row["state"]: list(CANDIDATES)[int(np.argmax(q))]
                                             for row, q in zip(available, scores, strict=True)})
    return result


def closure_metrics(root, plan_):
    closed = []
    for id_, item in plan_["origins"].items():
        if item["purpose"] != "closed":
            continue
        r = load(root / item["origin"])
        choices = item["choices"]
        results = {mode: result_for(root, r, choices[mode]) if mode in choices else dict(
            task_success=None, physics_validity=False, invalid_stage="probe",
            invalid_reasons=r["probe_summary"]["invalid_reasons"], post_try_sha256=None) for mode in MODES}
        closed.append(dict(initial_state_id=id_, state=r["state"], seed=r["seed"], probe=r["probe"],
                           choices=choices, scores=item["scores"], results=results))
    values = np.array([[[r["results"][m]["task_success"] for m in MODES] for r in closed
                        if r["state"] == s] for s in STATES], dtype=float)
    if values.shape != (3, 15, 4):
        raise ValueError("Expected 15 held-out physical states per hidden class")
    draws = np.random.default_rng(SEED).integers(15, size=(20000, 3, 15))
    sample = values[np.arange(3)[None, :, None], draws]
    counts_ = np.isfinite(sample).sum(axis=2)
    boot = np.divide(np.nansum(sample, axis=2), counts_, out=np.full(counts_.shape, np.nan), where=counts_ > 0)
    boot = boot.mean(axis=1)
    pool_counts = counts_.sum(axis=1)
    pooled = np.divide(np.nansum(sample, axis=(1, 2)), pool_counts,
                       out=np.full(pool_counts.shape, np.nan), where=pool_counts > 0)
    paired = values[:, :, -1] - values[:, :, 1]
    paired_boot = sample[:, :, :, -1] - sample[:, :, :, 1]
    paired_counts = np.isfinite(paired_boot).sum(axis=(1, 2))
    paired_means = np.divide(np.nansum(paired_boot, axis=(1, 2)), paired_counts,
                             out=np.full(len(draws), np.nan), where=paired_counts > 0)

    def ci(x):
        x = x[np.isfinite(x)]
        return np.quantile(x, [.025, .975]).tolist() if len(x) else None

    policy = {}
    for j, mode in enumerate(MODES):
        by_state = {s: estimate(values[i, :, j]) for i, s in enumerate(STATES)}
        cases = [r["results"][mode] for r in closed]
        policy[mode] = dict(estimate(values[:, :, j].ravel()), states=by_state,
                            invalid_reasons=dict(Counter(k for case in cases for k in case["invalid_reasons"])),
                            state_balanced_rate=(float(np.mean([c["rate"] for c in by_state.values()]))
                                                 if all(c["rate"] is not None for c in by_state.values()) else None),
                            ci95=ci(boot[:, j]), pooled_ci95=ci(pooled[:, j]))
    a, b = policy["Informed"]["state_balanced_rate"], policy["Blind"]["state_balanced_rate"]
    diff = a-b if a is not None and b is not None else None
    pairs = []
    for r in closed:
        a, b = (r["results"][m]["task_success"] for m in ("Informed", "Blind"))
        same = r["choices"].get("Informed") is not None and r["choices"]["Informed"] == r["choices"]["Blind"]
        if same:
            pairs.append((0., 0.))
        else:
            pairs.append(((0. if a is None else float(a)) - (1. if b is None else float(b)),
                          (1. if a is None else float(a)) - (0. if b is None else float(b))))
    return dict(episodes=closed, policies=policy, score_examples=plan_["score_examples"],
                matched_score_shift=matched_score_shift(plan_["score_examples"]),
                bootstrap_draws=len(draws), adaptation=adaptation(closed),
                informed_minus_blind=dict(gain=diff, ci95=ci(boot[:, -1]-boot[:, 1]),
                                          complete_pairs=int(np.isfinite(paired).sum()),
                                          complete_case_gain=float(np.nanmean(paired))
                                          if np.isfinite(paired).any() else None,
                                          complete_case_ci95=ci(paired_means),
                                          unknown_bounds=np.mean(pairs, axis=0).tolist()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("stage", choices=("train", "plan", "branches", "summarize"))
    parser.add_argument("--workers", type=int, default=22)
    args = parser.parse_args()
    root = args.output
    if args.stage == "train":
        train(root)
    elif args.stage == "plan":
        plan(root)
    elif args.stage == "branches":
        branch_all(root, args.workers)
    else:
        metrics = load(root / "c2_metrics.json")
        p = load(root / "branch_plan.json")
        ranking_metrics(root, p, metrics)
        save_json(root / "c2_metrics.json", metrics)
        save_json(root / "closed_loop_eval.json", closure_metrics(root, p))


if __name__ == "__main__":
    main()
