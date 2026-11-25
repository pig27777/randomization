"""Analyze grouped probe predictions and held-out physical grasp evaluations."""
import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
from math import comb
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from torch import nn

from pe_physics import save_json
from rss_exp_ab import CHANNELS, EFFORT, GRASPS, STATES, TRIES

ABLATIONS = ("Full", "w/o Effort", "w/o History")
DECODERS = ("Logistic", "V3 causal TCN")
TASKS = {"3-way": tuple(STATES), "Left vs Right": ("Left", "Right")}
COLORS = ("#0072B2", "#009E73", "#D55E00")
RNG_SEED = 271126


def v3_modules(output):
    snapshot = output / "source" / "policy_v3"
    if not snapshot.is_dir():
        from src.policy_v3 import history, models

        snapshot.mkdir(parents=True)
        (snapshot / "__init__.py").touch()
        for module in (history, models):
            shutil.copyfile(module.__file__, snapshot / Path(module.__file__).name)
    sys.path.insert(0, str(snapshot.parent.resolve()))
    from policy_v3 import history, models

    return history, models


def load_rows(path, experiment):
    return [(file, json.loads(file.read_text())) for file in sorted(path.glob(f"{experiment}-*.json"))]


def validate_design(entries, n, actions):
    for state, action in itertools.product(STATES, actions):
        reps = {row["replicate"] for row in entries if row["state"] == state and row["action"] == action}
        if reps != set(range(n)):
            raise ValueError(f"Incomplete {state}/{action}: {sorted(reps)}")
    ids = {}
    for row in entries:
        key = row["initial_state_id"]
        fingerprint = (row["initial_mesh_sha256"], row["initial_qpos_sha256"], row["split"])
        if key in ids and ids[key] != fingerprint:
            raise ValueError(f"Initial-state mismatch: {key}")
        ids[key] = fingerprint
    if len(entries) != len(STATES) * len(actions) * n or len(ids) != len(STATES) * n:
        raise ValueError("Unexpected number of trials or unique initial states")
    if len({row["public_shape_sha256"] for row in entries}) != 1:
        raise ValueError("Canonical visible geometry depends on the initial state")


def status_counts(entries):
    invalid = [r for r in entries if not r["physics_validity"]]
    return dict(trials=len(entries), valid=len(entries) - len(invalid), invalid=len(invalid),
                physics_invalid_rate=len(invalid) / len(entries) if entries else None,
                invalid_reasons=dict(Counter(reason for row in invalid for reason in row["invalid_reasons"])))


def group_kfold(groups):
    """Allocate intact groups to the currently smallest of five folds."""
    keys, membership, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(RNG_SEED).permutation(len(keys))
    order = order[np.argsort(-sizes[order], kind="stable")]
    group_fold = np.empty(len(keys), dtype=int)
    fold_sizes = np.zeros(5, dtype=int)
    for group in order:
        fold = int(np.argmin(fold_sizes))
        group_fold[group] = fold
        fold_sizes[fold] += sizes[group]
    return group_fold[membership]


def standardized(train, test):
    center = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True)
    # Suppress division by floating point noise in commanded or parked axes.
    scale = np.maximum(scale, 1e-6)
    return [(values - center) / scale for values in (train, test)]


def fit_logistic(x, y, n_classes):
    model = nn.Linear(x.shape[1], n_classes)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y) + .05 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


def predict(train, labels, test, n_classes, decoder, seed, v3):
    torch.manual_seed(seed)
    train, test = standardized(train, test)
    x, z = [torch.as_tensor(values, dtype=torch.float32) for values in (train, test)]
    y = torch.as_tensor(labels, dtype=torch.long)
    if decoder == "Logistic":
        x, z = x.flatten(1), z.flatten(1)
        model = fit_logistic(x, y, n_classes)
        with torch.no_grad():
            return model(z).argmax(1).numpy()
    history, models = v3
    encoder = models.ProprioHistoryEncoder()
    head = nn.Linear(64, n_classes)
    optimizer = torch.optim.AdamW(itertools.chain(encoder.parameters(), head.parameters()),
                                  lr=.002, weight_decay=.01)

    def logits(frames):
        frames = nn.functional.pad(frames, (0, history.HISTORY_DIM - frames.shape[-1]))
        mask = torch.zeros(frames.shape[:2], dtype=torch.bool)
        action = torch.zeros((len(frames), 13))
        return head(encoder(frames, mask, action))

    for _ in range(80):
        optimizer.zero_grad()
        nn.functional.cross_entropy(logits(x), y).backward()
        optimizer.step()
    with torch.no_grad():
        return logits(z).argmax(1).numpy()


def decode(traces, rows, folds, action, classes, decoder, ablation, v3):
    take = np.array([row["action"] == action and row["state"] in classes and row["physics_validity"]
                     for row in rows])
    x, f = traces[take].copy(), folds[take]
    ids = np.array([row["initial_state_id"] for row, keep in zip(rows, take) if keep])
    y = np.array([classes.index(row["state"]) for row, keep in zip(rows, take) if keep])
    if ablation == "w/o Effort":
        x[:, :, EFFORT] = 0.
    if ablation == "w/o History":
        x = x[:, -1:, :]
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    results, out_of_fold = [], {}
    for fold in range(5):
        train, test = f != fold, f == fold
        if set(ids[train]) & set(ids[test]):
            raise ValueError("Initial states leak across folds")
        if not test.any() or set(y[train]) != set(range(len(classes))):
            results.append(dict(fold=fold, train_n=int(train.sum()), test_n=int(test.sum()), accuracy=None))
            continue
        predictions = predict(x[train], y[train], x[test], len(classes), decoder, RNG_SEED + fold, v3)
        np.add.at(confusion, (y[test], predictions), 1)
        accuracy = float(np.mean(predictions == y[test]))
        results.append(dict(fold=fold, train_n=int(train.sum()), test_n=int(test.sum()), accuracy=accuracy))
        out_of_fold.update({id_: dict(true=classes[a], prediction=classes[b], fold=fold)
                            for id_, a, b in zip(ids[test], y[test], predictions)})
    accuracies = [result["accuracy"] for result in results if result["accuracy"] is not None]
    return dict(accuracy_mean=float(np.mean(accuracies)) if accuracies else None,
                accuracy_std=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None,
                folds=results, classes=classes, confusion_matrix=confusion.tolist(),
                valid_n=int(take.sum()), decoded_n=len(out_of_fold), oof=out_of_fold)


def paired_predictions(first, second):
    ids = sorted(first.keys() & second.keys())
    a = np.array([first[key]["true"] == first[key]["prediction"] for key in ids], dtype=bool)
    b = np.array([second[key]["true"] == second[key]["prediction"] for key in ids], dtype=bool)
    gain, loss = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    n = gain + loss
    p = min(1., 2 * sum(comb(n, i) for i in range(min(gain, loss) + 1)) / 2**n) if ids else None
    return dict(p_raw=p, paired_n=len(ids), gain=gain, loss=loss,
                accuracy_difference=(gain - loss) / len(ids) if ids else None)


def holm(rows):
    ordered = sorted((r for r in rows if r["p_raw"] is not None), key=lambda r: r["p_raw"])
    corrected = 0.
    for index, row in enumerate(ordered):
        corrected = max(corrected, min(1., row["p_raw"] * (len(ordered) - index)))
        row["p_holm"] = corrected


def analyze_a(rows_and_files, output):
    rows = [row for _, row in rows_and_files]
    validate_design(rows, 15, TRIES)
    trace = np.full((len(rows), 47, len(CHANNELS)), np.nan)
    frame_count = []
    for index, (path, row) in enumerate(rows_and_files):
        frames = np.load(path.with_suffix(".npy"), allow_pickle=False)
        frame_count.append(len(frames))
        if len(frames):
            trace[index, :len(frames)] = frames
        if row["physics_validity"] and (len(frames) != 47 or not np.isfinite(frames).all()):
            raise ValueError(f"Incomplete valid probe history: {path}")
    groups = np.array([r["initial_state_id"] for r in rows])
    folds = group_kfold(groups)
    np.savez_compressed(output / "expA_traces.npz", history=trace, channels=CHANNELS, frame_count=frame_count,
                        fold=folds, initial_state_id=groups, state=[r["state"] for r in rows],
                        try_action=[r["action"] for r in rows], seed=[r["seed"] for r in rows],
                        physics_validity=[r["physics_validity"] for r in rows],
                        invalid_reasons=[",".join(r["invalid_reasons"]) for r in rows],
                        xy_m=[r["xy_m"] for r in rows], yaw_deg=[r["yaw_deg"] for r in rows],
                        initial_mesh_sha256=[r["initial_mesh_sha256"] for r in rows],
                        initial_qpos_sha256=[r["initial_qpos_sha256"] for r in rows])
    models, comparisons = {}, []
    v3 = v3_modules(output)
    jobs = itertools.product(TASKS, DECODERS)
    for task, decoder in jobs:
        results = {action: {ablation: decode(trace, rows, folds, action, TASKS[task], decoder, ablation, v3)
                            for ablation in ABLATIONS} for action in TRIES}
        models.setdefault(task, {})[decoder] = results
        for first, second in itertools.combinations(TRIES, 2):
            comparisons.append(dict(task=task, decoder=decoder, action_a=second, action_b=first,
                                    **paired_predictions(results[second]["Full"]["oof"],
                                                         results[first]["Full"]["oof"])))
        print(task, decoder, "complete", flush=True)
    holm(comparisons)
    ablation_comparisons = [
        dict(task=task, decoder=decoder, action=action, ablation=ablation,
             **paired_predictions(results["Full"]["oof"], results[ablation]["oof"]))
        for task, decoders in models.items() for decoder, actions in decoders.items()
        for action, results in actions.items() for ablation in ABLATIONS[1:]
    ]
    holm(ablation_comparisons)
    # Full histories can saturate the task. Compare information available at
    # matched times using the same outer folds and a causal prefix.
    prefixes, prefix_comparisons = {}, []
    for end in (.16, .56, .84, 1.12, 1.40):
        partial = trace[:, :round(end * 25) + 1]
        results = {action: decode(partial, rows, folds, action, TASKS["3-way"], "Logistic", "Full", v3)
                   for action in TRIES}
        prefixes[str(end)] = results
        for first, second in itertools.combinations(TRIES, 2):
            prefix_comparisons.append(dict(end_s=end, action_a=second, action_b=first,
                                           **paired_predictions(results[second]["oof"], results[first]["oof"])))
    holm(prefix_comparisons)
    metrics = dict(**status_counts(rows), models=models, paired_try_comparisons=comparisons,
                   paired_ablation_comparisons=ablation_comparisons,
                   exploratory_prefixes=prefixes, exploratory_prefix_comparisons=prefix_comparisons,
                   frame_hz=25, groups=len(set(groups)), last_frame_s=1.84, rows=rows,
                   chance={"3-way": 1/3, "Left vs Right": .5},
                   v3_source_sha256={Path(mod.__file__).name: sha256(Path(mod.__file__).read_bytes()).hexdigest()
                                     for mod in v3},
                   cell_validity={a: {s: status_counts([r for r in rows if r["action"] == a and r["state"] == s])
                                      for s in STATES} for a in TRIES})
    save_json(output / "expA_metrics.json", metrics)
    return metrics, trace


def estimate(outcomes):
    valid = np.isfinite(outcomes)
    n = int(valid.sum())
    success = int(np.nansum(outcomes))
    return dict(success=success, valid=n, invalid=len(outcomes) - n,
                q=success / n if n else None,
                missing_outcome_bounds=[success / len(outcomes), (success + len(outcomes) - n) / len(outcomes)])


def evaluate_policies(eval_, oracle, fixed):
    selected = dict(oracle=[None if g is None else list(GRASPS)[g] for g in oracle],
                    vision=None if fixed is None else list(GRASPS)[fixed],
                    fixed=None if fixed is None else list(GRASPS)[fixed])
    evaluation = dict(J_oracle=None, J_vision=None, J_fixed=None, delta_hidden=None,
                      delta_hidden_ci95=None, n_complete_pairs=0)
    if fixed is not None and all(g is not None for g in oracle):
        a = eval_[np.arange(len(STATES)), oracle]
        b = eval_[:, fixed]
        valid_pair = np.isfinite(a) & np.isfinite(b)
        evaluation["n_complete_pairs"] = int(valid_pair.sum())
        a_cells, b_cells = [estimate(x) for x in a], [estimate(x) for x in b]
        # If two policies select the same grasp in a state, even an unknown
        # outcome for that state and perturbation cancels in their difference.
        bounds = [(0., 0.) if g == fixed else (
            x["missing_outcome_bounds"][0] - y["missing_outcome_bounds"][1],
            x["missing_outcome_bounds"][1] - y["missing_outcome_bounds"][0])
                  for g, x, y in zip(oracle, a_cells, b_cells)]
        evaluation.update(oracle_cells=a_cells, vision_cells=b_cells,
                          all_outcome_delta_bounds=np.mean(bounds, axis=0).tolist())
        for key, cells in (("J_oracle", a_cells), ("J_vision", b_cells), ("J_fixed", b_cells)):
            if all(c["q"] is not None for c in cells):
                evaluation[key] = float(np.mean([c["q"] for c in cells]))
        if evaluation["J_oracle"] is not None and evaluation["J_vision"] is not None:
            evaluation["delta_hidden"] = evaluation["J_oracle"] - evaluation["J_vision"]
            rng = np.random.default_rng(RNG_SEED)
            draws = rng.integers(4, size=(20000, len(STATES), 4))
            sample_a = a[np.arange(len(STATES))[None, :, None], draws]
            sample_b = b[np.arange(len(STATES))[None, :, None], draws]
            valid_boot = np.isfinite(sample_a).any(axis=2).all(axis=1) & np.isfinite(sample_b).any(axis=2).all(axis=1)
            evaluation["usable_bootstrap_draws"] = int(valid_boot.sum())
            if valid_boot.any():
                delta = np.mean(np.nanmean(sample_a[valid_boot], axis=-1)
                                - np.nanmean(sample_b[valid_boot], axis=-1), axis=1)
                evaluation["delta_hidden_ci95"] = np.quantile(delta, [.025, .975]).tolist()
    return dict(choices=selected, evaluation=evaluation)


def mass_matrix(path):
    rows = [r for _, r in load_rows(path, "B")]
    validate_design(rows, 8, GRASPS)
    values = np.full((len(STATES), len(GRASPS), 8), np.nan)
    for row in rows:
        if row["physics_validity"]:
            values[list(STATES).index(row["state"]), list(GRASPS).index(row["action"]),
                   row["replicate"]] = row["success"]
        elif row["success"] is not None:
            raise ValueError("Invalid run coded as an ordinary outcome")
    splits = {"selection": values[:, :, :4], "evaluation": values[:, :, 4:]}
    selection_seeds = {r["seed"] for r in rows if r["split"] == "selection"}
    evaluation_seeds = {r["seed"] for r in rows if r["split"] == "evaluation"}
    if selection_seeds & evaluation_seeds:
        raise ValueError("Selection and evaluation seeds overlap")
    if len(selection_seeds) != 12 or len(evaluation_seeds) != 12:
        raise ValueError("Unexpected number of independent states per split")
    matrices = {key: {state: {grasp: estimate(split[i, j]) for j, grasp in enumerate(GRASPS)}
                      for i, state in enumerate(STATES)} for key, split in splits.items()}
    matrices["all"] = {state: {grasp: estimate(values[i, j]) for j, grasp in enumerate(GRASPS)}
                       for i, state in enumerate(STATES)}
    select = splits["selection"]
    valid_counts = np.isfinite(select).sum(axis=-1)
    q = np.divide(np.nansum(select, axis=-1), valid_counts, out=np.full(valid_counts.shape, np.nan),
                  where=valid_counts > 0)
    # A policy requires selection observations for every state it is evaluated on.
    choice = lambda a: None if np.isnan(a).all() else int(np.nanargmax(a))
    oracle = [choice(a) for a in q]
    fixed = choice(q.mean(axis=0))
    eval_ = splits["evaluation"]
    primary = evaluate_policies(eval_, oracle, fixed)
    # Rank every pose by the lower end of its selection interval. Unknown
    # physics outcomes keep their bounds and their separate validity status.
    lower = np.nansum(select, axis=-1) / select.shape[-1]
    sensitivity = evaluate_policies(eval_, [int(np.argmax(a)) for a in lower],
                                    int(np.argmax(lower.mean(axis=0))))
    sweep_required = bool(np.any(np.all(values[:, :, 4:] == 1., axis=(0, 2))))
    return dict(**status_counts(rows), **primary, matrices=matrices,
                lower_bound_selection=sensitivity,
                seed_sets=dict(selection=sorted(selection_seeds), evaluation=sorted(evaluation_seeds)),
                heavy_control_triggered=sweep_required, rows=rows)


def export(fig, output, stem):
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures_a(metrics, trace, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75), gridspec_kw={"width_ratios": [1., 1.15]},
                             layout="constrained")
    force = CHANNELS.index("reaction_z_N")
    for axis, action in zip(axes[:, 0], ("Try_Scan", "Try_Left")):
        for state, color in zip(STATES, COLORS):
            take = np.array([r["action"] == action and r["state"] == state and r["physics_validity"]
                             for r in metrics["rows"]])
            x = trace[take, :, force]
            if len(x):
                sec = trace[take][0, :, -1]
                mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
                axis.plot(sec, mean, lw=1.5, color=color, label=state)
                axis.fill_between(sec, mean - sd, mean + sd, color=color, alpha=.15, linewidth=0)
        axis.set(title=action.replace("_", " "), ylabel="Vertical reaction (N)", xlim=(0, 1.84))
        axis.margins(y=.14)
        axis.legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    axes[-1, 0].set_xlabel("Time (s)")
    hues = ("#24496E", "#6CA9B5", "#CFD8E3")
    from matplotlib.ticker import PercentFormatter

    axis = axes[0, 1]
    decoder = "V3 causal TCN"
    x = np.arange(len(TRIES))
    for j, (ablation, hue) in enumerate(zip(ABLATIONS, hues)):
        cells = [metrics["models"]["3-way"][decoder][a][ablation] for a in TRIES]
        for i, cell in enumerate(cells):
            mean, std = cell["accuracy_mean"], cell["accuracy_std"]
            at = x[i] + (j - 1) * .24
            if mean is None:
                axis.text(at, .035, "NA", fontsize=7, ha="center")
                continue
            axis.bar(at, mean, width=.22, color=hue, yerr=std or 0., capsize=2,
                     error_kw={"linewidth": .75, "capthick": .75})
    scan = metrics["models"]["3-way"][decoder]["Try_Scan"]
    test = next(r for r in metrics["paired_ablation_comparisons"] if
                (r["task"], r["decoder"], r["action"], r["ablation"]) ==
                ("3-way", decoder, "Try_Scan", "w/o History"))
    if test["p_holm"] is not None:
        axis.text(2., 1.12, f"{pct(scan['Full']['accuracy_mean'])} → "
                  f"{pct(scan['w/o History']['accuracy_mean'])}\nHolm p = {test['p_holm']:.3g}",
                  fontsize=7, ha="center", va="center")
    axis.set(title=decoder, xticks=x, xticklabels=[a.replace("_", " ") for a in TRIES])
    axis.legend(handles=[Patch(color=h, label=a) for h, a in zip(hues, ABLATIONS)],
                fontsize=7, ncol=3, loc="upper center", frameon=False)

    axis = axes[1, 1]
    prefixes = metrics["exploratory_prefixes"]
    times = sorted(map(float, prefixes))
    for action, color, marker in zip(TRIES, COLORS, ("o", "s", "^")):
        means = [prefixes[str(t)][action]["accuracy_mean"] for t in times]
        axis.plot(times, means, marker=marker, color=color, ms=4, lw=1.4, label=action.replace("_", " "))
    test = next(r for r in metrics["exploratory_prefix_comparisons"]
                if (r["end_s"], r["action_a"], r["action_b"]) == (.56, "Try_Left", "Try_Center"))
    if test["p_holm"] is not None:
        left, center = (prefixes["0.56"][a]["accuracy_mean"] for a in ("Try_Left", "Try_Center"))
        axis.axvline(.56, color=".55", ls=":", lw=.9)
        axis.text(.59, .67, f"{pct(left)} vs. {pct(center)}\nHolm p = {test['p_holm']:.4f}",
                  fontsize=7, va="center")
    axis.legend(fontsize=7, ncol=3, loc="lower right", frameon=False)
    axis.set(title="Early-prefix accuracy (logistic)", xlabel="Probe duration (s)", xticks=times,
             xlim=(times[0] - .03, times[-1] + .04))
    for axis in axes[:, 1]:
        axis.axhline(1/3, linestyle="--", color=".50", lw=.8, zorder=0)
        axis.set(ylim=(0, 1.38 if axis is axes[0, 1] else 1.17),
                 yticks=[0, .25, .5, .75, 1.], ylabel="3-way accuracy")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    export(fig, output, "fig_expA_identifiability")


def pct(v, decimals=1):
    return "NA" if v is None else f"{v:.{decimals}%}"


def figures_b(light, heavy, output, approach_label="Top"):
    datasets = [("200 g", light)] + ([("350 g", heavy)] if heavy else [])
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.2, 2.9 * len(datasets)),
                             gridspec_kw={"width_ratios": [3., 1.27]}, squeeze=False, layout="constrained")
    for (name, data), (ax, info) in zip(datasets, axes):
        matrix = data["matrices"]["evaluation"]
        cells = [[matrix[s][g] for g in GRASPS] for s in STATES]
        array = np.array([[c["q"] if c["q"] is not None else np.nan for c in row] for row in cells])
        palette = matplotlib.colormaps["Blues"].copy()
        palette.set_bad("#EDEDED")
        xx, yy = np.meshgrid(np.arange(len(GRASPS) + 1), np.arange(len(STATES) + 1))
        ax.pcolormesh(xx, yy, array, cmap=palette, vmin=0, vmax=1, edgecolors="white", linewidth=.8)
        for i, row in enumerate(cells):
            for j, c in enumerate(row):
                fg = "white" if c["q"] is not None and c["q"] > .55 else "#162235"
                ax.text(j+.5, i+.5, f"{pct(c['q'], 0)}\n{c['success']}/{c['valid']}", ha="center", va="center",
                        color=fg, fontsize=8)
        ax.set(xticks=np.arange(len(GRASPS))+.5, xticklabels=list(GRASPS),
               yticks=np.arange(len(STATES))+.5, yticklabels=list(STATES), ylabel="Hidden state",
               title=f"{name} core · Independent evaluation")
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        geometry = ax.secondary_xaxis(-.11)
        geometry.set_xticks([1.5, 4.5, 7., 8.5], [f"{approach_label}\n−X / 0 / +X", "Side\n−X / 0 / +X",
                                                   "Outer\n+X / −X", "Tilted"])
        geometry.tick_params(length=0, labelsize=7)
        geometry.spines["bottom"].set_visible(False)
        info.axis("off")
        evaluation = data["evaluation"]
        ci = evaluation["delta_hidden_ci95"]
        sensitivity = data["lower_bound_selection"]["evaluation"]
        alt_ci = sensitivity["delta_hidden_ci95"]
        interval = lambda bounds: f"[{pct(bounds[0])}, {pct(bounds[1])}]" if bounds else "NA"
        delta_label = "$\\Delta_{\\mathrm{hidden}}$"
        lines = [
            (.95, "Complete-cell selection", ""),
            (.83, "Oracle", pct(evaluation["J_oracle"])),
            (.73, "Vision", pct(evaluation["J_vision"])),
            (.63, "Best fixed", pct(evaluation["J_fixed"])),
            (.53, delta_label, pct(evaluation["delta_hidden"])),
            (.43, "95% CI", interval(ci)),
            (.29, "Lower-bound selection", ""),
            (.19, delta_label, pct(sensitivity["delta_hidden"])),
            (.09, "95% CI", interval(alt_ci)),
            (-.07, "Physics invalid", f"{data['invalid']}/{data['trials']}"),
        ]
        for y, label, value in lines:
            info.text(.02, y, label, fontsize=8, fontweight="bold" if not value else "normal", va="center")
            info.text(.97, y, value, fontsize=8, ha="right", va="center")
    export(fig, output, "fig_expB_adaptation_gap")


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows)]


def acc(cell):
    mean, sd = cell["accuracy_mean"], cell["accuracy_std"]
    return f"{pct(mean)} ± {pct(sd)}" if mean is not None else "NA"


def report_a(data):
    models = data["models"]
    rows = []
    for action, decoder in itertools.product(TRIES, DECODERS):
        rows.append([action, decoder, *(acc(models[task][decoder][action][ablation])
                                         for task, ablation in itertools.product(TASKS, ABLATIONS))])
    lines = ["## Exp A · 试探可辨识性", "",
             f"物理有效 {data['valid']}/{data['trials']}，"
             f"`physics_invalid_rate` = {pct(data['physics_invalid_rate'])}。"
             f"记录 {data['groups']} 个初始状态分组，最大试探反力 "
             f"{max(row['peak_reaction_N'] for row in data['rows']):.3f} N。", "",
             "线性解码为 L2 正则化多项 Logistic 回归，输入为展开后的历史。"
             "TCN 复用 Policy V3 的 3 层因果卷积编码器和池化，附加线性分类层。"
             "各折仅用训练集拟合归一化和模型。末帧消融取撤回结束时的 1.84 s 帧。"
             "下表是五折准确率的均值 ± 折间样本标准差。", "",
             *md_table(["试探", "解码器", "三类 Full", "三类去力", "三类末帧",
                        "左右 Full", "左右去力", "左右末帧"], rows), "",
             "全输入下各试探差异按共同有效并有折外预测的初始状态配对。"
             "采用双侧精确 McNemar，12 项比较作 Holm 校正。交叉验证各折共享部分训练状态，"
             "这些 p 值按折外预测的配对结果解读。", ""]
    comparisons = []
    for row in data["paired_try_comparisons"]:
        p = row.get("p_holm")
        comparisons.append([row["task"], row["decoder"], row["action_a"] + " − " + row["action_b"],
                            row["paired_n"], pct(row["accuracy_difference"]), f"{p:.3g}" if p is not None else "NA"])
    lines += md_table(["任务", "解码器", "试探差值", "配对数", "准确率差", "Holm p"], comparisons)
    if all(models[task][decoder][action]["Full"]["accuracy_mean"] == 1.
           for task, decoder, action in itertools.product(TASKS, DECODERS, TRIES)):
        lines += ["", "两个任务的完整历史准确率均达到上限。动作间差异由下方前缀分析衡量。"]
    scan_history = [row for row in data["paired_ablation_comparisons"]
                    if row["task"] == "3-way" and row["action"] == "Try_Scan" and row["ablation"] == "w/o History"]
    if scan_history:
        lines += ["", "长扫的三分类完整历史相对末帧准确率增量为 "
                  + "，".join(f"{r['decoder']} {pct(r['accuracy_difference'])}（Holm p = {r['p_holm']:.3g}）"
                             for r in scan_history if "p_holm" in r)
                  + "。完整历史中的位置、速度和误差仍会反映接触。去力分析只移除显式力通道。"]
    lines += ["", "完整输入混淆矩阵采用真实类为行、预测类为列。三类顺序为 Left、Center、Right，"
              "二类顺序为 Left、Right。各消融的矩阵、配对结果和逐折预测见原始指标。", ""]
    matrix = []
    for action, decoder in itertools.product(TRIES, DECODERS):
        matrix.append([action, decoder, *(str(models[task][decoder][action]["Full"]["confusion_matrix"])
                                          for task in TASKS)])
    lines += md_table(["试探", "解码器", "三类", "左右"], matrix)
    prefixes = data.get("exploratory_prefixes")
    if prefixes:
        lines += ["", "完整历史准确率达到上限后，补做三分类 Logistic 的等时长前缀分析。"
                  "0.16 s 位于下降前。其余窗口只读取截至该时刻的观测，使用相同分组和训练流程。"
                  "这是探索性分析，共 15 项动作配对比较并作 Holm 校正。", "",
                  *md_table(["前缀终点 (s)", *TRIES],
                            [[end, *(acc(results[action]) for action in TRIES)]
                             for end, results in prefixes.items()])]
        significant = [row for row in data["exploratory_prefix_comparisons"] if row.get("p_holm", 1.) < .05]
        if significant:
            lines += ["", *md_table(["终点 (s)", "准确率差", "差值", "Holm p"],
                                    [[row["end_s"], row["action_a"] + " − " + row["action_b"],
                                      pct(row["accuracy_difference"]), f"{row['p_holm']:.3g}"]
                                     for row in significant])]
    lines += ["", "[图 A PDF](fig_expA_identifiability.pdf) · [PNG](fig_expA_identifiability.png) · "
              "[25 Hz 原始历史](expA_traces.npz) · [逐折指标和无效原因](expA_metrics.json)", ""]
    return lines


def report_b(data, title):
    lines = [f"### {title}", "",
             f"参考夹具与袋膜的接触由求解器计算。当前矩阵共 {data['trials']} 次，"
             f"物理无效 {data['invalid']} 次，比例 {pct(data['physics_invalid_rate'])}。"
              "每格给出成功／物理有效数。两个集合均为每格固定 4 次尝试。", ""]
    for key, label in (("selection", "选择集"), ("evaluation", "评测集")):
        lines += [f"**{label}**", ""]
        matrix = data["matrices"][key]
        lines += md_table(["隐藏状态", *GRASPS],
                          [[s, *(f"{matrix[s][g]['success']}/{matrix[s][g]['valid']}" for g in GRASPS)]
                           for s in STATES])
        lines.append("")
    selected, values = data["choices"], data["evaluation"]
    ci = values["delta_hidden_ci95"]
    interval = f"[{pct(ci[0])}, {pct(ci[1])}]" if ci else "NA"
    lines += md_table(["策略", "选择集冻结的抓姿", "独立评测值"],
                      [["Oracle", ", ".join(f"{s} → {g}" for s, g in zip(STATES, selected["oracle"])),
                        pct(values["J_oracle"])],
                       ["Vision", selected["vision"], pct(values["J_vision"])],
                       ["Best Fixed Grasp", selected["fixed"], pct(values["J_fixed"])],
                       ["Δhidden", f"配对 Bootstrap 95% CI {interval}", pct(values["delta_hidden"])]])
    lines += ["", f"Oracle 与 Vision 同时物理有效的评测对为 {values['n_complete_pairs']}/12。"
              "Bootstrap 在每个状态内对同一扰动成对重采样，固定选择集得出的抓姿。"]
    if "usable_bootstrap_draws" in values:
        lines.append(f"20,000 次重采样中，{values['usable_bootstrap_draws']:,} 次覆盖两策略的所有状态。")
    if ci and ci[0] <= 0. <= ci[1]:
        lines.append("区间覆盖 0，当前独立评测对正向自适应缺口的证据不足。")
    bounds = values.get("all_outcome_delta_bounds")
    if bounds:
        lines.append(f"将物理无效视作结果未知时，当前样本的差距界限为 [{pct(bounds[0])}, {pct(bounds[1])}]。")
    sensitivity = data["lower_bound_selection"]
    alt = sensitivity["evaluation"]
    candidate = sensitivity["choices"]["fixed"]
    primary_candidates = [g for g in GRASPS if all(data["matrices"]["selection"][s][g]["valid"] for s in STATES)]
    lines += ["", f"主分析只在选择集三个状态都有有效观测的候选中选择固定策略"
              f"（{len(primary_candidates)}/{len(GRASPS)} 个候选）。"
              f"事后敏感性分析按各选择格成功率下界在全池排名，物理无效继续视作未知结果。"
              f"该规则选出固定／视觉抓姿 {candidate}，独立评测值 {pct(alt['J_fixed'])}，"
              f"Oracle 为 {pct(alt['J_oracle'])}，差距 {pct(alt['delta_hidden'])}。"]
    alt_ci = alt["delta_hidden_ci95"]
    if alt_ci:
        lines.append(f"其配对 Bootstrap 95% CI 为 [{pct(alt_ci[0])}, {pct(alt_ci[1])}]。")
    alt_bounds = alt.get("all_outcome_delta_bounds")
    if alt_bounds:
        lines.append(f"结果未知界限为 [{pct(alt_bounds[0])}, {pct(alt_bounds[1])}]。")
    if alt_ci and alt_ci[0] <= 0. <= alt_ci[1]:
        lines.append("在允许全池参与固定抓姿选择时，正向差距仍需更多物理有效样本确认。")
    lines.append("")
    return lines


def write_report(output, a, b):
    report = output / "exp_ab_report.md"
    marker = "## 正式结果"
    plan = report.read_text().split(marker, 1)[0].rstrip()
    light, heavy = b["light"], b["heavy"]
    n = a["trials"] + light["trials"] + (heavy["trials"] if heavy else 0)
    invalid = a["invalid"] + light["invalid"] + (heavy["invalid"] if heavy else 0)
    lines = [plan, "", marker, "",
             f"正式回合 {n} 次，物理无效 {invalid} 次，合计 {pct(invalid / n)}。"
             "固定 50 µs CPU Flex，使用 PE MVP 参考夹具。"
             "参考夹具按位姿指令驱动两指，内芯始终使用自由刚体动力学。"
             "相同扰动下三种状态的初始可见 RGB 逐像素一致。",
             "[初始图像核对](public-vis-check.log) · [冻结路径](frozen-check.log)", "",
             *report_a(a), "## Exp B · 条件隐藏状态自适应缺口", "",
             "三个状态等权。单一公开视觉等价类使 Vision 和 Best Fixed Grasp 的优化问题相同。"
             "表中 J 按状态平均物理有效回合的成功率。物理无效可能依赖状态或抓姿，"
             "因此同时列出将这部分结果视为未知的界限。", "",
             *report_b(light, "200 g 内芯")]
    if heavy:
        lines += report_b(heavy, "350 g 对照")
    lines += ["[图 B PDF](fig_expB_adaptation_gap.pdf) · [PNG](fig_expB_adaptation_gap.png) · "
              "[抓姿、拆分种子与逐回合指标](expB_matrix.json)", ""]
    report.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--part", choices=("A", "B", "both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    if args.part in ("A", "both"):
        metrics, trace = analyze_a(load_rows(args.output / "episodes", "A"), args.output)
        figures_a(metrics, trace, args.output)
    if args.part in ("B", "both"):
        light = mass_matrix(args.output / "episodes")
        heavy_dir = args.output / "heavy_episodes"
        heavy = mass_matrix(heavy_dir) if heavy_dir.is_dir() else None
        matrix = dict(grasp_poses=GRASPS, light=light, heavy=heavy,
                      split_rule="replicates 0–3 selection, 4–7 evaluation",
                      tie_break="first candidate by public grasp ID",
                      bootstrap="20000 paired, stratified by hidden state, frozen policy choices")
        save_json(args.output / "expB_matrix.json", matrix)
        figures_b(light, heavy, args.output)
        metrics_path = args.output / "expA_metrics.json"
        if metrics_path.is_file() and (args.output / "exp_ab_report.md").is_file():
            write_report(args.output, json.loads(metrics_path.read_text()), matrix)
        print("B complete; heavy control required:", light["heavy_control_triggered"], flush=True)


if __name__ == "__main__":
    main()
