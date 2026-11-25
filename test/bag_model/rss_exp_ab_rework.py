"""Analyze original Exp A observations and the revised held-out Exp B scan."""
import argparse
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import torch

from bag_mvp import make_trial
from pe_physics import MAX_STRETCH, parcel_config, save_json
from rss_exp_ab import (
    AUDIT_DT, B_DURATION, B_REWORK_SEED_ROOT, CHANNELS, CLOSE_START, DT, GRASPS, HOLD_END, JAW_FEEDBACK_DT, JAW_FORCE_LIMIT,
    JAW_MASS, JAW_TARGET, LIFT_DISTANCE, LIFT_END, LIFT_START, MAX_SUPPORT_PAUSES, RELEASE_CLEARANCE,
    SETDOWN_COMPRESSION_LIMIT, SETDOWN_TIME, SETDOWN_TRAVEL, STATES, SUPPORT_PAUSE, TRIES,
    Job, public_envelope,
)
from rss_exp_ab_analysis import (
    ABLATIONS, DECODERS, RNG_SEED, TASKS, decode, figures_a, figures_b, fit_logistic,
    holm, mass_matrix, md_table, paired_predictions, pct, standardized, v3_modules, plt,
)

NOISE = {
    "Clean": (0., 0., 0., 0),
    "Mild": (.0005, .05, .05, 0),
    "Moderate": (.001, .15, .15, 1),
    "Strong": (.002, .30, .30, 2),
}
NOISE_SEED = 2026112602


def perturb(trace, groups, levels):
    position_sd, force_sd, deadzone, delay = levels
    if not position_sd:
        return trace
    keys, membership = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(NOISE_SEED)
    shape = (len(keys), trace.shape[1])
    position_noise = rng.normal(size=(*shape, 3))[membership] * position_sd
    force_noise = rng.normal(size=(*shape, 4))[membership] * force_sd
    delayed = np.maximum(0, np.arange(trace.shape[1]) - delay)
    result = trace.copy()
    result[:, :, :13] = trace[:, delayed, :13]
    result[:, :, :3] += position_noise[:, delayed]
    position_increment = np.diff(position_noise, axis=1, prepend=position_noise[:, :1])
    result[:, :, 3:6] += position_increment[:, delayed] / .04
    result[:, :, 6:9] += trace[:, :, 13:16] - trace[:, delayed, 13:16] - position_noise[:, delayed]
    effort = result[:, :, 9:13] + force_noise[:, delayed]
    result[:, :, 9:13] = np.where(np.abs(effort) >= deadzone, effort, 0.)
    return result


def reflection_audit():
    cfg = parcel_config(make_trial("try-center", DT))
    envelope = public_envelope(cfg)
    points = envelope["current"]
    reflected = points * [-1., 1., 1.]
    distances = np.linalg.norm(reflected[:, None] - points, axis=-1)
    nearest = np.argmin(distances, axis=1)
    faces = {tuple(sorted(face)) for face in envelope["faces"]}
    return dict(vertices=len(points), faces=len(faces),
                matched_vertices_at_1um=int((distances.min(axis=1) <= 1e-6).sum()),
                max_nearest_reflection_distance_m=float(distances.min(axis=1).max()),
                rms_nearest_reflection_distance_m=float(np.sqrt(np.mean(distances.min(axis=1)**2))),
                reflected_faces_found=sum(tuple(sorted(nearest[list(face)])) in faces for face in faces),
                mesh_source=envelope["metadata"]["source"])


def audit_center(trace, rows, folds):
    state = np.array([row["state"] for row in rows])
    center = np.array([row["action"] == "Try_Center" for row in rows])
    times = trace[0, :, -1]
    stats = {}
    for label in STATES:
        values = trace[center & (state == label)]
        stats[label] = dict(n=len(values), mean_by_frame=values.mean(axis=0).tolist(),
                            sd_by_frame=values.std(axis=0, ddof=1).tolist())
    contrast = np.asarray(stats["Left"]["mean_by_frame"]) - stats["Right"]["mean_by_frame"]
    at = np.argmax(np.abs(contrast), axis=0)
    peak_contrast = [{"channel": channel, "time_s": float(times[i]), "left_minus_right": float(contrast[i, j])}
                     for j, (channel, i) in enumerate(zip(CHANNELS, at))]
    pair = center & (state != "Center")
    x, labels, f = trace[pair], (state[pair] == "Right").astype(int), folds[pair]
    weights = {}
    for label, values, time_axis in (("Full", x, times), ("w/o History", x[:, -1:], times[-1:])):
        coefficients = []
        for fold in range(5):
            train, _ = standardized(values[f != fold], values[f == fold])
            tx = torch.as_tensor(train.reshape(len(train), -1), dtype=torch.float32)
            ty = torch.as_tensor(labels[f != fold], dtype=torch.long)
            model = fit_logistic(tx, ty, 2)
            coefficients.append((model.weight[1] - model.weight[0]).detach().numpy())
        mean = np.mean(coefficients, axis=0)
        sd = np.std(coefficients, axis=0, ddof=1)
        weights[label] = [
            dict(channel=CHANNELS[i % len(CHANNELS)], time_s=float(time_axis[i // len(CHANNELS)]),
                 mean_right_minus_left=float(mean[i]), sd=float(sd[i]))
            for i in np.argsort(np.abs(mean))[::-1][:5]
        ]
    return dict(channels=CHANNELS, time_s=times.tolist(), states=stats, peak_mean_contrasts=peak_contrast,
                logistic_top5=weights, x_reflection=reflection_audit(),
                probe=dict(start_x_m=TRIES["Try_Center"][0], span_x_m=TRIES["Try_Center"][1],
                           start_t_s=.62, end_t_s=1.54, core_inner_edge_abs_x_m=.055 - .088 / 2))


def analyze_a(source, output):
    metrics = json.loads((source / "expA_metrics.json").read_text())
    with np.load(source / "expA_traces.npz", allow_pickle=False) as data:
        trace, folds, groups = (data[key] for key in ("history", "fold", "initial_state_id"))
        if list(data["channels"]) != list(CHANNELS) or not data["physics_validity"].all():
            raise ValueError("Unexpected Exp A source channels or invalid episodes")
    rows = metrics["rows"]
    if (trace.shape != (135, 47, len(CHANNELS)) or not np.isfinite(trace).all()
            or not np.array_equal(groups, [row["initial_state_id"] for row in rows])
            or set(folds) != set(range(5)) or len(set(zip(groups, folds))) != 45):
        raise ValueError("Unexpected Exp A trace pairing or group folds")
    metrics["try_center_symmetry"] = audit_center(trace, rows, folds)
    snapshot = source / "source" / "policy_v3"
    if not all((snapshot / name).is_file() for name in metrics["v3_source_sha256"]):
        raise ValueError("Missing frozen r1 V3 decoder snapshot")
    for name, expected in metrics["v3_source_sha256"].items():
        if sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen r1 decoder changed: {name}")
    v3 = v3_modules(source)
    noise_results, tests = {}, []
    for level, config in NOISE.items():
        observations = perturb(trace, groups, config)
        models = metrics["models"] if level == "Clean" else {
            task: {decoder: {action: {
                ablation: decode(observations, rows, folds, action, classes, decoder, ablation, v3)
                for ablation in ABLATIONS
            } for action in TRIES} for decoder in DECODERS} for task, classes in TASKS.items()
        }
        noise_results[level] = dict(
            position_sd_m=config[0], force_sd_N=config[1], force_deadzone_N=config[2],
            delay_frames=config[3], models=models,
        )
        for task, decoder in product(TASKS, DECODERS):
            acts = models[task][decoder]
            for ablation in ABLATIONS:
                for first, second in combinations(TRIES, 2):
                    tests.append(dict(kind="action", level=level, task=task, decoder=decoder, ablation=ablation,
                                      action_a=second, action_b=first,
                                      **paired_predictions(acts[second][ablation]["oof"], acts[first][ablation]["oof"])))
            for action, ablation in product(TRIES, ABLATIONS[1:]):
                tests.append(dict(kind="ablation", level=level, task=task, decoder=decoder,
                                  action=action, ablation=ablation,
                                  **paired_predictions(acts[action]["Full"]["oof"], acts[action][ablation]["oof"])))
            if level != "Clean":
                for action, ablation in product(TRIES, ABLATIONS):
                    baseline = metrics["models"][task][decoder][action][ablation]
                    tests.append(dict(kind="clean_vs_noise", level=level, task=task, decoder=decoder,
                                      action=action, ablation=ablation,
                                      **paired_predictions(baseline["oof"], acts[action][ablation]["oof"])))
        print(f"Exp A {level} complete", flush=True)
    for kind, task, decoder in product(("action", "ablation", "clean_vs_noise"), TASKS, DECODERS):
        holm([test for test in tests if (test["kind"], test["task"], test["decoder"]) == (kind, task, decoder)])
    metrics["noise_sensitivity"] = dict(
        seed=NOISE_SEED, levels=noise_results, paired_tests=tests,
        correction="Holm across all levels within test family × task × decoder",
        noise_model="One seeded Gaussian sensor draw per initial_state_id, shared across actions and scaled across levels. "
                    "Position error shares position noise; velocity uses its causal backward difference. "
                    "Four effort channels receive Gaussian noise before hard dead zone. Delayed observations "
                    "hold the first frame during warm-up. Task commands and the clock are current.",
    )
    metrics["source_files_sha256"] = {name: sha256((source / name).read_bytes()).hexdigest()
                                      for name in ("expA_traces.npz", "expA_metrics.json")}
    save_json(output / "expA_metrics_r2.json", metrics)
    figures_a(metrics, trace, output)
    lines = ["## Exp A · 对称性与观测扰动", "",
             "135/135 条原物理轨迹有效。沿用五折初始状态分组，标准化仅使用各折训练数据。", "",
             "### Try Center 末帧（1.84 s）逐通道均值 ± 1 SD", ""]
    stat = metrics["try_center_symmetry"]["states"]
    lines += md_table(["通道", *STATES],
                      [[channel, *[f"{stat[s]['mean_by_frame'][-1][j]:.6g} ± "
                                    f"{stat[s]['sd_by_frame'][-1][j]:.3g}" for s in STATES]]
                       for j, channel in enumerate(CHANNELS)])
    lines += ["", "### 左右线性判别权重", "",
              "权重为训练折标准化特征上 Right−Left 的 logit 系数，按绝对均值排序。", ""]
    lines += md_table(["输入", "通道", "时间 (s)", "权重均值 ± 折间 SD"],
                      [[ablation, w["channel"], f"{w['time_s']:.2f}",
                        f"{w['mean_right_minus_left']:+.3f} ± {w['sd']:.3f}"]
                       for ablation, weights in metrics["try_center_symmetry"]["logistic_top5"].items()
                       for w in weights])
    lines += ["", "### 四档观测扰动", "",
              "同一初始状态的各动作共享同一随机扰动，命令和时钟保持当前值。"
              "位置误差与位置共享噪声，速度扰动来自位置噪声的因果差分，"
              "四个力通道先加噪再施加死区。各档均在相同扰动分布下重新训练与评测。", ""]
    lines += md_table(["观测档", "位置 σ (mm)", "力 σ (N)", "力死区 (N)", "延迟 (ms)"],
                      [[level, f"{1000*p:.1f}", f"{f:.2f}", f"{d:.2f}", str(delay * 40)]
                       for level, (p, f, d, delay) in NOISE.items()])
    lines.append("")
    for task, decoder in product(TASKS, DECODERS):
        lines += [f"#### {task} · {decoder}", ""]
        lines += md_table(["试探", "输入", *NOISE],
                          [[a.replace("_", " "), b, *[pct(n["models"][task][decoder][a][b]["accuracy_mean"])
                                                    for n in noise_results.values()]]
                           for a, b in product(TRIES, ABLATIONS)])
        lines.append("")
    notable = [x for x in tests if x["kind"] == "clean_vs_noise" and x["action"] == "Try_Center"
               and x["ablation"] == "w/o History"]
    lines += ["配对显著性采用相同初始状态的折外预测做精确 McNemar 检验，"
              "按比较类型、任务与解码器在全部噪声档内做 Holm 校正。"
              "所有原始与调整后的 p 值、配对数和胜负数见 JSON。", ""]
    lines += ["Try Center 末帧消融的 Clean 与扰动档对照如下。", ""]
    lines += md_table(["噪声", "任务 / 解码器", "Holm p", "Clean − 扰动"],
                      [[x["level"],
                        f"{x['task']} / {x['decoder']}", f"{x['p_holm']:.3g}", pct(x["accuracy_difference"])]
                       for x in notable])
    (output / "expA-tables.md").write_text("\n".join(lines) + "\n")


def check_b(data):
    missing = {split: [f"{state}×{action}" for state, cells in data["matrices"][split].items()
                       for action, cell in cells.items() if not cell["valid"]]
               for split in ("selection", "evaluation")}
    primary, lower = data["evaluation"], data["lower_bound_selection"]["evaluation"]
    deltas = primary["delta_hidden"], lower["delta_hidden"]
    return dict(invalid_rate_pass=data["physics_invalid_rate"] <= .05,
                no_empty_cells_pass=not any(missing.values()), missing_cells=missing,
                identical_choices=data["choices"] == data["lower_bound_selection"]["choices"],
                delta_difference=None if None in deltas else deltas[0] - deltas[1])


def load_b(path, phase, mass):
    matrix = mass_matrix(path)
    for row in matrix["rows"]:
        job = Job("B", row["state"], row["action"], row["replicate"], phase, mass)
        if (row["seed"], row["split"], row["core_mass_kg"]) != (job.seed, job.split, mass):
            raise ValueError(f"Unexpected seed, split or load for {job.id}/{job.action}")
        if row["peak_jaw_motor_N"] > JAW_FORCE_LIMIT + 1e-9:
            raise ValueError(f"Jaw actuator limit exceeded for {job.id}/{job.action}")
        if not row["physics_validity"]:
            continue
        if row["invalid_reasons"] or row["peak_edge_ratio"] > MAX_STRETCH:
            raise ValueError(f"Validity fields conflict for {job.id}/{job.action}")
        passed = (row["hold_min_lift_m"] is not None and row["hold_min_lift_m"] >= .010
                  and row["bilateral_fraction"] >= .95 and row["release_drop_m"] is not None
                  and row["release_drop_m"] >= .010)
        if row["success"] != passed:
            raise ValueError(f"Success fields conflict for {job.id}/{job.action}")
    return matrix


def b_tables(output, baseline, light, heavy):
    lines = ["## Exp B · 正式矩阵", ""]
    datasets = [("r1 · 200 g", baseline), ("r2 · 200 g", light)]
    if heavy:
        datasets.append(("r2 · 350 g", heavy))
    lines += md_table(["数据", "物理无效", "无效率", "选择 0/0", "评测 0/0", "无效原因"],
                      [[name, f"{d['invalid']}/{d['trials']}", pct(d["physics_invalid_rate"]),
                        str(len(check_b(d)["missing_cells"]["selection"])),
                        str(len(check_b(d)["missing_cells"]["evaluation"])),
                        ", ".join(f"{k} {v}" for k, v in d["invalid_reasons"].items()) or "无"]
                       for name, d in datasets])
    lines += ["", "评测种子独立于选择集。Oracle 对每个状态分别在选择集选位姿，"
              "Vision 和最佳固定策略在唯一公开外观等价类中选同一全局位姿。"
              "当物理无效时，条件成功率去掉无效分母，全结果界保留这些未知结果。"
              "95% CI 使用评测集 20,000 次按状态分层、按扰动索引配对重采样，选择结果固定。", ""]
    policy_rows = []
    for name, data in datasets:
        for method, d in (("Complete-cell", data), ("Lower-bound", data["lower_bound_selection"])):
            e = d["evaluation"]
            choices = d["choices"]
            interval = lambda key: ("[" + ", ".join(map(pct, e[key])) + "]") if e.get(key) else "NA"
            policy_rows.append([name, method, "/".join(g or "NA" for g in choices["oracle"]),
                                choices["vision"] or "NA", *[pct(e[k]) for k in
                                ("J_oracle", "J_vision", "J_fixed", "delta_hidden")],
                                interval("delta_hidden_ci95"), interval("all_outcome_delta_bounds")])
    lines += md_table(["数据", "选择法", "Oracle L/C/R", "Vision / Fixed", "J Oracle", "J Vision",
                       "J Fixed", "Δ hidden", "95% CI", "全结果差值界"], policy_rows)
    lines += ["", "### 候选修复前后", "",
              "以下数量均为选择与评测合计，双侧指膜接触按稳持窗口计。"
              "闭合期双侧接触可核对新顶部位姿即便在失去负载前是否建立过接触。", ""]
    rows = []
    for state, action in (*zip(STATES, ("g4", "g5", "g6")), *zip(STATES, ("g1", "g2", "g3"))):
        for name, data in datasets[:2]:
            samples = [r for r in data["rows"] if r["state"] == state and r["action"] == action]
            valid = [r for r in samples if r["physics_validity"]]
            rows.append([f"{state}×{action}", name, f"{sum(r['success'] for r in valid)}/{len(valid)}",
                         str(len(samples) - len(valid)),
                         str(sum((r.get("bilateral_close_fraction") or 0) > 0 for r in samples))
                         if name.startswith("r2") else "未存",
                         str(sum((r.get("bilateral_fraction") or 0) > 0 for r in valid)),
                         f"{max(r['peak_edge_ratio'] for r in samples):.4f}",
                         f"{max(r['peak_jaw_N'] for r in samples):.2f}"])
    lines += md_table(["单元格", "数据", "成功/有效", "无效", "闭合双侧", "稳持双侧",
                       "最大边长比", "峰值反力 (N)"], rows)
    lines += [""]
    rows = []
    for name, data in datasets[:2]:
        for g in ("g7", "g8", "g9"):
            samples = [r for r in data["rows"] if r["action"] == g]
            valid = [r for r in samples if r["physics_validity"]]
            lifts = [r["hold_min_lift_m"] for r in valid if r["hold_min_lift_m"] is not None]
            rows.append([g, name, f"{sum(r['success'] for r in valid)}/{len(valid)}",
                         str(sum("jaw_inset_core" in r["invalid_reasons"] for r in samples)),
                         f"{1000 * max(lifts):.2f}" if lifts else "NA"])
    lines += md_table(["位姿", "数据", "成功/有效", "指垫侵入回合", "最佳最低净空 (mm)"], rows)
    for name, data in datasets[1:]:
        checks = check_b(data)
        lines += ["", f"{name} 无效率门禁{'通过' if checks['invalid_rate_pass'] else '未通过'}，"
                  f"完整矩阵门禁{'通过' if checks['no_empty_cells_pass'] else '未通过'}。"
                  f"两种估计选出的策略{'一致' if checks['identical_choices'] else '有差异'}。"]
        for split, missing in checks["missing_cells"].items():
            if missing:
                lines.append(f"{split} 的 0/0 单元格为 {'、'.join(missing)}。")
    if heavy:
        lines += ["", "200 g 评测触发了 350 g 对照，两档采用相同的扰动索引。"]
    elif light["heavy_control_triggered"]:
        lines += ["", "200 g 已触发 350 g 对照，当前段落记录已完成的轻载结果。"]
    else:
        lines += ["", "200 g 的封边抓与 g5 均未达到预定重载触发条件。"]
    report = output / "rework_report.md"
    source = report.read_text() if report.is_file() else "# RSS 2027 Exp A/B r2\n"
    start, end = "<!-- exp-b-results -->", "<!-- /exp-b-results -->"
    block = "\n".join([start, "", *lines, "", end])
    if start in source and end in source:
        before, _, rest = source.partition(start)
        _, _, after = rest.partition(end)
        source = before + block + after
    elif "\n## Exp A · 审计结论" in source:
        source = source.replace("\n## Exp A · 审计结论", f"\n{block}\n\n## Exp A · 审计结论", 1)
    else:
        source = source.rstrip() + "\n\n" + block + "\n"
    report.write_text(source)


def analyze_b(source, output):
    baseline = json.loads((source / "expB_matrix.json").read_text())["light"]
    light = load_b(output / "episodes", "formal", .20)
    heavy_path = output / "heavy_episodes"
    heavy = load_b(heavy_path, "heavy", .35) if heavy_path.is_dir() else None
    evaluation = light["matrices"]["evaluation"]
    outer_trigger = any(evaluation[s][g]["success"] >= 3 for s, g in product(STATES, ("g7", "g8")))
    g5_trigger = sum(evaluation[s]["g5"]["success"] >= 2 for s in STATES) >= 2
    light["heavy_control_triggered"] = outer_trigger or g5_trigger
    matrix = dict(
        grasp_poses=GRASPS, light=light, heavy=heavy,
        controller=dict(timestep_s=DT, audit_dt_s=AUDIT_DT, max_duration_s=B_DURATION,
                        close_start_s=CLOSE_START, lift_start_s=LIFT_START, lift_end_s=LIFT_END,
                        hold_end_s=HOLD_END, lift_m=LIFT_DISTANCE, force_feedback_dt_s=JAW_FEEDBACK_DT,
                        lift_force_reference="measured resting surface load plus 1 N",
                        max_setdown_s=SETDOWN_TIME, max_setdown_travel_m=SETDOWN_TRAVEL,
                        support_pause_s=SUPPORT_PAUSE, max_support_pauses=MAX_SUPPORT_PAUSES,
                        support_load_source="total vertical reaction of external table and catch floor",
                        setdown_compression_limit_N=SETDOWN_COMPRESSION_LIMIT,
                        release_backoff_m=RELEASE_CLEARANCE,
                        force_target_N=JAW_TARGET, motor_limit_N=JAW_FORCE_LIMIT, jaw_mass_kg=JAW_MASS),
        split_rule="replicates 0–3 selection, 4–7 evaluation",
        seed_root=B_REWORK_SEED_ROOT,
        tie_break="first candidate by public grasp ID",
        bootstrap="20000 paired perturbation draws, stratified by state, policy choices frozen",
        gates=dict(light=check_b(light), heavy=check_b(heavy) if heavy else None),
        heavy_trigger=dict(outer_success_3_of_4=outer_trigger, g5_success_2_of_4_in_two_states=g5_trigger),
    )
    save_json(output / "expB_matrix_r2.json", matrix)
    figures_b(light, heavy, output, approach_label="8° cross tilt")
    b_tables(output, baseline, light, heavy)
    print("Exp B complete. Heavy control required:", light["heavy_control_triggered"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--part", choices=("A", "B"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    if args.part == "A":
        analyze_a(args.source, args.output)
    else:
        analyze_b(args.source, args.output)


if __name__ == "__main__":
    main()
