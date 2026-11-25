"""Build the C1/C2 results report and paper figure from the physical archives."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rss_cpu_flex_c1c2 import CANDIDATES, STATES
from rss_cpu_flex_c1c2_policy import MODES, load


def pct(x):
    return "NA" if x is None else f"{x:.1%}"


def interval(values):
    return "NA" if values is None else "[" + ", ".join(pct(x) for x in values) + "]"


def row_table(names, rows):
    return ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows)]


def figure(output, closure):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(11, 5.15), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(1.28, 1.))
    order = sorted(CANDIDATES, key=lambda g: (CANDIDATES[g][0][0], CANDIDATES[g][0][1]))
    idx = [list(CANDIDATES).index(g) for g in order]
    panels = {x["state"]: x for x in closure["score_examples"]}
    for i, state in enumerate(STATES):
        ax = fig.add_subplot(grid[i, 0])
        panel = panels[state]
        for mode, color, marker in (("Blind", "#64676B", "o"), ("Informed", "#0072B2", "D")):
            scores = panel["scores"].get(mode)
            if scores:
                ax.plot(np.arange(len(idx)), np.asarray(scores)[idx], color=color, marker=marker,
                        markersize=3.7, lw=1.3, label=mode)
        ax.set(ylabel=f"{state}\nPred. success", ylim=(0, 1.06), yticks=[0, .5, 1.], xticks=range(len(order)))
        ax.set_xticklabels(order if i == 2 else [""] * len(order))
        if i == 0:
            ax.set_title("Identical initial RGB · identical Try command")
            if panel["scores"]:
                ax.legend(loc="upper left", frameon=False, ncol=2)
        if i == 2:
            ax.set_xlabel("Candidate pose ordered by bag-frame x")
    ax = fig.add_subplot(grid[:, 1])
    palette = ("#D1D4D7", "#64676B", "#67A9B8", "#0072B2")
    for j, (mode, color) in enumerate(zip(MODES, palette, strict=True)):
        policy = closure["policies"][mode]
        value, ci = policy["state_balanced_rate"], policy["ci95"]
        if value is None or ci is None:
            ax.text(j, .05, "NA", ha="center")
            continue
        bottom, top = np.clip(np.array([value-ci[0], ci[1]-value]), 0, None)
        ax.bar(j, value, color=color, width=.69, yerr=[[bottom], [top]], capsize=3)
        ax.text(j, min(1.075, max(value, ci[1]) + .047),
                f"{pct(value)}\n{policy['success']}/{policy['valid']}", ha="center", fontsize=8)
    ax.set(title="Held-out physical grasp success", ylim=(0, 1.17), yticks=np.linspace(0, 1, 5),
           ylabel="State-balanced success (95% paired bootstrap CI)", xticks=range(4),
           xticklabels=("Best Fixed", "Blind", "Informed\nw/o Effort", "Informed"))
    gain = closure["informed_minus_blind"]
    fig.suptitle("Policy V3 · 50 µs CPU Flex · PE reference fixture", fontsize=12)
    if gain["gain"] is not None:
        ax.text(.03, .96, f"Informed − Blind  {100*gain['gain']:+.1f} pp\n95% CI {interval(gain['ci95'])}",
                transform=ax.transAxes, va="top", fontsize=8)
    stem = output / "fig_c2_score_shift_and_success"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=450)
    plt.close(fig)


def number(value):
    return "NA" if value is None else f"{value:.3f}"


def write_report(output, dataset, c2, closure):
    path = output / "c1c2_report.md"
    mark = "## 正式结果"
    plan = path.read_text().split(mark, 1)[0].rstrip()
    lines = [plan, "", mark, "",
             "全部结果来自 50 µs CPU Flex 和 PE MVP 参考夹具。"
             "候选以公共初始外袋坐标定义，V3 的 V0_ONLY 接口屏蔽 v1。"
             "V3 历史编码保留显式试探力、位置、速度、误差与控制指令，"
             "参考夹具未观测的整臂关节字段填零。", "",
             "### C1 · 采集和分组", "",
             *row_table(("划分", "总回合", "有效", "正样本", "正样本率", "物理无效率"),
                        [[split, v["trials"], v["valid"], v["positives"], pct(v["positive_rate"]),
                          pct(v["physics_invalid_rate"])] for split, v in
                         (("All", dataset), *dataset["splits"].items())]), "",
             "episode_id、initial_state_id 和 probe_state_id 全部完成跨集不相交验证。"
             "三组模型使用完全相同的有效训练记录与监督 BCE 目标。", "",
             "### C2 · 留出评分", ""]
    rank = []
    for mode in MODES[1:]:
        entries = []
        for split in ("val", "test"):
            cell = c2["models"][mode][split]
            top = cell["top1"]
            entries += [number(cell["bce"]), number(cell["heldout_auc"]),
                        f"{top['correct']}/{top['rankable_groups']} · {pct(top['accuracy'])}"]
        rank.append([mode, *entries])
    lines += row_table(("模型", "验证 BCE", "验证 AUROC", "验证 Top-1",
                       "测试 BCE", "测试 AUROC", "测试 Top-1"), rank)
    lines += ["", "每个验证及测试状态类别预先选两个完成试探的状态。"
              "对每个选中的状态物理执行整池候选。Top-1 在至少一正一负且有物理标签的组内，"
              "比较实际有效候选中的最高评分与成功标签。原始指标另保留全池原始 argmax 的有效分母及未知界限。", "",
              "### 闭环 · 独立物理评测", ""]
    result_rows = []
    for mode in MODES:
        policy = closure["policies"][mode]
        rates = [f"{r['success']}/{r['valid']}" for r in policy["states"].values()]
        result_rows.append([mode, *rates, f"{policy['success']}/{policy['valid']}",
                            pct(policy["state_balanced_rate"]), interval(policy["ci95"]),
                            interval(policy["unknown_bounds"]), pct(policy["physics_invalid_rate"])])
    lines += row_table(("策略", *STATES, "总成功/有效", "状态等权成功率", "95% CI", "未知结果界限", "物理无效率"),
                       result_rows)
    gain = closure["informed_minus_blind"]
    lines += ["", f"Informed − Blind 为 {pct(gain['gain'])}，配对 Bootstrap 95% CI {interval(gain['ci95'])}。"
              f"两策略同时有效 {gain['complete_pairs']}/45，未知结果界限 {interval(gain['unknown_bounds'])}。"
              "Bootstrap 以初始状态为配对单位，在各隐藏状态中独立重采样，"
              "每个策略只用有效回合估计该状态的成功率。", ""]
    lines += ["### 行为适应度", ""]
    report_rows = []
    for mode in MODES:
        for state, status in closure["adaptation"][mode]["states"].items():
            mean = status["mean_bag_xyz_m"]
            report_rows.append([mode, state, status["n"], pct(status["different_from_blind"]),
                                "NA" if mean is None else ", ".join(f"{x*1000:.1f}" for x in mean)])
    lines += row_table(("策略", "隐藏状态", "选姿数", "不同于 Blind", "袋坐标选姿均值 (mm)"), report_rows)
    lines += ["", "每个候选的选择频率、跨状态总变差距离和按策略的物理无效原因见 JSON。"
              "共用候选时，一个完全相同的积分检查点与动作只执行一次，供对应策略共享。"
              "45 个闭环初始态的种子与全量 C1 分离。", "",
              "[论文图 PDF](fig_c2_score_shift_and_success.pdf) · [PNG](fig_c2_score_shift_and_success.png) · "
              "[数据集](c1_dataset_summary.json) · [离线指标](c2_metrics.json) · "
              "[闭环原始指标](closed_loop_eval.json) · [冻结检查](frozen-check.log)", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    closure = load(out / "closed_loop_eval.json")
    figure(out, closure)
    write_report(out, load(out / "c1_dataset_summary.json"), load(out / "c2_metrics.json"), closure)


if __name__ == "__main__":
    main()
