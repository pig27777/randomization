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
        value, ci = policy["rate"], policy["pooled_ci95"]
        if value is None or ci is None:
            ax.text(j, .05, "NA", ha="center")
            continue
        bottom, top = np.clip(np.array([value-ci[0], ci[1]-value]), 0, None)
        ax.bar(j, value, color=color, width=.69, yerr=[[bottom], [top]], capsize=3)
        ax.text(j, min(1.075, max(value, ci[1]) + .047),
                f"{pct(value)}\n{policy['success']}/{policy['valid']}", ha="center", fontsize=8)
    ax.set(title="Held-out physical grasp success", ylim=(0, 1.17), yticks=np.linspace(0, 1, 5),
           ylabel="Success on physics-valid runs (95% CI)", xticks=range(4),
           xticklabels=("Best Fixed", "Blind", "Informed\nw/o Effort", "Informed"))
    gain = closure["informed_minus_blind"]
    fig.suptitle("Policy V3 · 50 µs CPU Flex · PE reference fixture", fontsize=12)
    if gain["complete_case_gain"] is not None:
        lo, hi = gain["complete_case_ci95"]
        ax.set_xlabel(f"Valid pairs · Informed − Blind {100*gain['complete_case_gain']:+.1f} pp\n"
                      f"95% CI [{100*lo:.1f}, {100*hi:.1f}] pp", labelpad=8, fontsize=8)
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
             *row_table(("隐藏状态", "总回合", "有效", "正样本", "有效内正样本率"),
                        [[state, c["trials"], c["valid"], c["positives"], pct(c["positive_rate"])]
                         for state, c in dataset["states"].items()]), "",
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
                            pct(policy["rate"]), interval(policy["pooled_ci95"]), pct(policy["state_balanced_rate"]),
                            interval(policy["unknown_bounds"]), pct(policy["physics_invalid_rate"])])
    lines += row_table(("策略", *STATES, "总成功/有效", "合并有效成功率", "95% CI", "状态等权成功率",
                        "未知结果界限", "物理无效率"),
                       result_rows)
    gain = closure["informed_minus_blind"]
    lines += ["", f"两策略同时有效 {gain['complete_pairs']}/45。"
              f"该子集的 Informed − Blind 为 {pct(gain['complete_case_gain'])}，"
              f"配对 Bootstrap 95% CI {interval(gain['complete_case_ci95'])}。"
              f"45 个状态的未知结果界限为 {interval(gain['unknown_bounds'])}。"
              "Bootstrap 以初始状态成对、在各隐藏类内分层重采样。", ""]
    if gain["gain"] is None:
        lines += ["至少一个状态的策略在物理有效回合中没有观测，状态等权差值保留为 NA。"
                  "图中的合并有效成功率可能来自不同状态比例。配对差值限于双方均有有效因果对照的子集。", ""]
    else:
        lines += [f"状态等权 Informed − Blind 为 {pct(gain['gain'])}，95% CI {interval(gain['ci95'])}。", ""]
    causes = sorted({cause for mode in MODES for cause in closure["policies"][mode]["invalid_reasons"]})
    if causes:
        lines += [*row_table(("无效归因", *MODES),
                             [[cause, *(closure["policies"][mode]["invalid_reasons"].get(cause, 0)
                                        for mode in MODES)] for cause in causes]), ""]
    lines += ["### 行为适应度", ""]
    report_rows = []
    for mode in MODES:
        for state, status in closure["adaptation"][mode]["states"].items():
            mean = status["mean_bag_xyz_m"]
            report_rows.append([mode, state, status["n"], pct(status["different_from_blind"]),
                                "NA" if mean is None else ", ".join(f"{x*1000:.1f}" for x in mean)])
    lines += row_table(("策略", "隐藏状态", "选姿数", "不同于 Blind", "袋坐标选姿均值 (mm)"), report_rows)
    lines += ["", *row_table(("匹配 RGB 示例", "Blind 峰值抓姿", "Informed 峰值抓姿"),
                             [[case["state"], *(list(CANDIDATES)[int(np.argmax(case["scores"][mode]))]
                                                if mode in case["scores"] else "NA"
                                                for mode in ("Blind", "Informed"))]
                              for case in closure["score_examples"]])]
    for mode, summary in closure["matched_score_shift"].items():
        span = summary["max_across_state_probability_span"]
        if span is not None:
            n_argmax = len(set(summary["argmax_by_state"].values()))
            lines += ["", f"{mode} 在三个逐像素一致的初态中有 {n_argmax} 个不同峰值抓姿，"
                      f"同一候选的最大跨状态分数差为 {span:.3g}。"]
    lines += ["", *row_table(("策略", "Left–Center TV", "Left–Right TV", "Center–Right TV"),
                             [[mode, *(number(v) for v in row["cross_state_total_variation"].values())]
                              for mode, row in closure["adaptation"].items()])]
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
