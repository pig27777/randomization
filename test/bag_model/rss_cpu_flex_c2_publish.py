"""Publish the C2 r2 score migration and checkpoint-paired physics results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from rss_cpu_flex_c1c2 import CANDIDATES, STATES
from rss_cpu_flex_c1c2_policy import MODES, load
from rss_cpu_flex_c1c2_report import interval, number, pct, row_table

DISPLAY = dict(zip(MODES, ("Best Fixed", "Blind", "Informed w/o Effort", "Informed (Ours)"), strict=True))


def ratio(entry):
    return f"{entry['success']}/{entry['valid']}"


def points(value):
    return "NA" if value is None else f"{100 * value:+.1f} pp"


def points_ci(values):
    return "NA" if values is None else "[" + ", ".join(f"{100*x:+.1f}" for x in values) + "] pp"


def figure(out, closure):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.15, 4.15), layout="constrained")
    sections = fig.add_gridspec(1, 2, width_ratios=(1.12, 1.))
    traces = sections[0].subgridspec(3, 1)
    order = sorted(CANDIDATES, key=lambda g: (*CANDIDATES[g][0][:2], int(g[1:])))
    idx = [list(CANDIDATES).index(g) for g in order]
    panels = {case["state"]: case for case in closure["score_examples"]}
    color = "#006C9D"
    for j, state in enumerate(STATES):
        ax = fig.add_subplot(traces[j])
        scores = panels[state]["scores"]
        for mode, label, clr, style in (("Blind", "Blind", "#62686D", "--"),
                                        ("Informed w/o Effort", "w/o effort", "#A05034", ":"),
                                        ("Informed", "Informed", color, "-")):
            ax.plot(np.asarray(scores[mode])[idx], color=clr, linestyle=style, lw=1.3, label=label)
        peak = order[int(np.argmax(np.asarray(scores["Informed"])[idx]))]
        ax.scatter(order.index(peak), max(scores["Informed"]), color=color, edgecolor="white",
                   linewidth=.45, s=25, zorder=4)
        ax.set(ylabel=f"{state}", ylim=(-.045, 1.04), yticks=(0., .5, 1.), xticks=range(len(order)))
        if j == 0:
            ax.set_title("a  Scores on identical initial RGB", loc="left")
            ax.legend(loc="upper left", ncol=3, frameon=False,
                      fontsize=6.5, columnspacing=1.0, handlelength=1.4)
        if j == len(STATES)-1:
            ax.set_xticklabels(order, fontsize=6.5)
            ax.set_xlabel("Candidate pose ordered by bag-frame x")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.grid(axis="y", color="#E8EAED", linewidth=.55)

    summary = sections[1].subgridspec(2, 1, height_ratios=(1.65, 1.))
    ax = fig.add_subplot(summary[0])
    ax.set_title("b  Grasp success · 45 states", loc="left")
    for j, mode in enumerate(MODES):
        item = closure["policies"][mode]
        rate = item["rate"]
        if rate is None:
            ax.text(.02, j, "NA", va="center")
            continue
        lo, hi = item["pooled_ci95"]
        clr = color if mode == "Informed" else "#7C858A"
        ax.plot([lo, hi], [j, j], color=clr, lw=1.25)
        ax.scatter(rate, j, color=clr, marker="D" if mode == "Informed" else "o", s=24, zorder=3)
        ax.annotate(f"{pct(rate)}  ({ratio(item)})", (rate, j), xytext=(0, 7),
                    textcoords="offset points", va="bottom", ha="center", fontsize=7)
    ax.set(xlim=(-.01, 1.01), ylim=(3.35, -.65), yticks=range(4),
           yticklabels=("Fixed", "Blind", "w/o effort", "Informed"),
           xticks=np.linspace(0, 1, 5), xlabel="Success among valid runs (95% CI)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", color="#E8EAED", linewidth=.55)
    table = fig.add_subplot(summary[1])
    table.axis("off")
    table.text(0, .99, "Successes / valid runs by hidden state", va="top", fontweight="bold")
    x = (.44, .66, .88)
    for col, state in zip(x, STATES, strict=True):
        table.text(col, .78, state, ha="center")
    for j, label in enumerate(("Fixed", "Blind", "w/o effort", "Informed")):
        y = .60 - j * .17
        mode = MODES[j]
        clr = color if mode == "Informed" else "#23292E"
        table.text(0, y, label, color=clr)
        for col, state in zip(x, STATES, strict=True):
            table.text(col, y, ratio(closure["policies"][mode]["states"][state]), ha="center", color=clr)
    stem = out / "fig_c2_score_shift_and_success"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=450)
    plt.close(fig)


def rss_paragraph(c2, closure):
    policies = closure["policies"]
    inform, blind = policies["Informed"], policies["Blind"]
    gain = closure["informed_minus_blind"]
    shifts = closure["matched_score_shift"]["Informed"]
    picks = ", ".join(shifts["argmax_by_state"][s] for s in STATES)
    tv = "–".join(f"{v:.2f}" for v in (min(closure["adaptation"]["Informed"]["cross_state_total_variation"].values()),
                                      max(closure["adaptation"]["Informed"]["cross_state_total_variation"].values())))
    return (
        "We trained a FiLM-conditioned causal temporal convolutional scorer on 265 physics-valid trials "
        "from the archived 50 µs MuJoCo CPU Flex collection. The scorer used initial geometry, the public "
        "probe command, and proprioception recorded during the probe. The score combined success "
        "conditional on a valid run with relative candidate support estimated from valid samples and "
        "the known randomized proposal. On three parcel states with "
        "pixel-identical initial RGB and the same command, the largest score change for a fixed grasp "
        f"was {shifts['max_across_state_probability_span']:.3f}. The selected grasps were {picks} "
        f"for Left, Center, and Right, respectively. The held-out test AUROC was "
        f"{c2['models']['Informed']['test']['heldout_auc']:.3f} for Informed and "
        f"{c2['models']['Blind']['test']['heldout_auc']:.3f} for Blind. We evaluated 45 separate "
        "initializations, 15 per hidden state, with the policies branching from identical post-probe "
        f"physics checkpoints within each initialization. Across the Left, Center, and Right states, "
        f"Informed achieved {', '.join(ratio(inform['states'][s]) for s in STATES)} successes per valid run "
        f"and Blind achieved {', '.join(ratio(blind['states'][s]) for s in STATES)}. "
        f"Blind had {blind['states']['Center']['attempts'] - blind['states']['Center']['valid']} invalid "
        "Center runs. "
        f"Informed's pairwise total-variation distance between hidden-state grasp distributions was {tv}. "
        f"Across {gain['complete_pairs']} jointly valid pairs, the paired success-rate difference "
        f"(Informed minus Blind) was {points(gain['complete_case_gain'])} "
        f"(95% stratified paired bootstrap CI {points_ci(gain['complete_case_ci95'])}). "
        f"Treating invalid outcomes as unknown gave a full-cohort difference bounded by "
        f"{points_ci(gain['unknown_bounds'])}."
        + (" The full-cohort difference remains unresolved because these bounds include zero."
           if gain["unknown_bounds"][0] <= 0 <= gain["unknown_bounds"][1] else ""))


def report(out, c2, closure):
    path = out / "c2_fix_report.md"
    title = "## 结果"
    text = [path.read_text().split(title, 1)[0].rstrip(), "", title, "",
            "### C2 离线评分", ""]
    data = []
    for mode, model in c2["models"].items():
        heldout = []
        for split in ("val", "test"):
            m = model[split]
            t = m["top1"]
            heldout += [number(m["bce"]), number(m["heldout_auc"]),
                        f"{t['correct']}/{t['rankable_groups']}",
                        f"{t['full_pool_top1_success']}/{t['full_pool_top1_valid']} / {t['attempted_groups']}"]
        shift = closure["matched_score_shift"][mode]
        data.append([DISPLAY[mode], model["best_epoch"], *heldout,
                     f"{shift['max_across_state_probability_span']:.6f}",
                     " / ".join(shift["argmax_by_state"][s] for s in STATES)])
    text += row_table(("评分器", "最佳 epoch", "验证 BCE", "验证 AUROC", "验证 Top-1",
                       "验证全池成功/有效/组", "测试 BCE", "测试 AUROC", "测试 Top-1",
                       "测试全池成功/有效/组", "最大跨态分数差", "匹配峰值 L/C/R"), data)
    text += ["", "Top-1 在至少有一正一负的状态组内，从有物理标签的候选中比较最高评分。"
             "全池指标先选分数最高的候选，再记录实际标签和有效分母。每个划分有六个沿用 r1 的状态组。"
             "三个 320 × 240 初始 RGB 逐像素相同，V0、公开 Try 和候选坐标相同。最终权重保存后重载并再次通过门禁。", "",
             "### 45 状态闭环", ""]
    data = []
    for mode, item in closure["policies"].items():
        data.append([DISPLAY[mode], *(ratio(item["states"][s]) for s in STATES), ratio(item), pct(item["rate"]),
                     interval(item["pooled_ci95"]), pct(item["state_balanced_rate"]),
                     interval(item["unknown_bounds"]), f"{item['attempts']-item['valid']}/{item['attempts']}",
                     pct(item["physics_invalid_rate"])])
    text += row_table(("策略", "Left 成功/有效", "Center 成功/有效", "Right 成功/有效", "总成功/有效",
                       "合并有效成功率", "95% CI", "状态等权", "全样本未知界限", "无效/尝试", "无效率"), data)
    gain = closure["informed_minus_blind"]
    text += ["", f"双方同时有效 {gain['complete_pairs']}/45。Informed − Blind 为 "
             f"{points(gain['complete_case_gain'])}，配对 Bootstrap 95% CI "
             f"{points_ci(gain['complete_case_ci95'])}。状态等权差为 {points(gain['gain'])}，"
             f"95% CI {points_ci(gain['ci95'])}。全 45 状态的配对未知结果界限为 "
             f"{points_ci(gain['unknown_bounds'])}。20,000 次 Bootstrap 按隐藏状态分层，以共同初始态为配对单位。"
             "某一策略在某类状态没有任何有效回合时，其状态等权成功率留空。", ""]
    r1 = load(out.parent / "cpu-flex-policy-v3-c1c2-r1" / "closed_loop_eval.json")
    text += ["r2 的 Blind 重训后仍在 45 个状态选 g5。下表列出两轮在相同种子上的分状态结果。", ""]
    data = []
    for mode, item in (("r1 Blind", r1["policies"]["Blind"]),
                       ("r1 Informed", r1["policies"]["Informed"]),
                       ("r2 Blind", closure["policies"]["Blind"]),
                       ("r2 Informed", closure["policies"]["Informed"])):
        data.append([mode, ratio(item["states"]["Left"]), ratio(item["states"]["Right"]),
                     ratio(item["states"]["Center"]), 15 - item["states"]["Center"]["valid"]])
    text += row_table(("策略", "Left 成功/有效", "Right 成功/有效", "Center 成功/有效", "Center 无效数"),
                      data)
    center_by_try = []
    for probe in ("Try_Left", "Try_Scan"):
        cases = [r for r in closure["episodes"] if r["state"] == "Center" and r["probe"] == probe]
        picked = [r["choices"]["Informed"] for r in cases]
        valid = sum(r["results"]["Informed"]["physics_validity"] for r in cases)
        success = sum(r["results"]["Informed"]["task_success"] is True for r in cases)
        center_by_try.append([probe, f"{success}/{valid}",
                              ", ".join(f"{g}×{picked.count(g)}" for g in sorted(set(picked)))])
    text += ["", *row_table(("Center 公开试探", "Informed 成功/有效", "候选计数"), center_by_try)]
    causes = sorted({reason for item in closure["policies"].values() for reason in item["invalid_reasons"]})
    text += ["", *row_table(("无效原因", *(DISPLAY[m] for m in MODES)),
                           [[reason, *(closure["policies"][m]["invalid_reasons"].get(reason, 0)
                                       for m in MODES)] for reason in causes]),
             "", "### 行为适应度", ""]
    data = []
    for mode, entry in closure["adaptation"].items():
        selected = []
        for s in STATES:
            cell = entry["states"][s]
            selected.append(", ".join(f"{g}×{round(f * cell['n'])}"
                                      for g, f in cell["density"].items() if f))
        bag_x = " / ".join(f"{1000 * entry['states'][s]['mean_bag_xyz_m'][0]:.1f}" for s in STATES)
        data.append([DISPLAY[mode], *selected, bag_x,
                     *(number(f) for f in entry["cross_state_total_variation"].values())])
    text += row_table(("策略", "Left 选择数", "Center 选择数", "Right 选择数", "选姿平均 x L/C/R (mm)",
                       "Left–Center TV", "Left–Right TV", "Center–Right TV"), data)
    p = closure["physical_provenance"]
    text += ["", f"闭环共 {p['closed_origins']} 个独立初始态，复用 {p['distinct_cached_closed_branches']} 个物理分支，"
             f"新增 {p['distinct_new_closed_branches']} 个物理分支。另复用 {p['cached_ranking_branches']} 个留出排序分支，"
             "排序状态的原始候选标签从 C1 包读取。Best Fixed 沿用 r1 的训练划分状态等权下界规则，"
             "其中无效记为未知，三组 BCE 训练均严格排除无效样本。", "",
             "### RSS 2027 Section 4", "", rss_paragraph(c2, closure), "",
             "[矢量 PDF](fig_c2_score_shift_and_success.pdf) · [450 dpi PNG](fig_c2_score_shift_and_success.png) · "
             "[离线指标](c2_metrics.json) · [闭环指标](closed_loop_eval.json) · "
             "[新增物理分支](new_physical_branches.tar.gz) · [冻结核验](frozen-check.log) · "
             "[原单测](policy-v3-regression.log)", ""]
    path.write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    c2, closure = (load(out / name) for name in ("c2_metrics.json", "closed_loop_eval.json"))
    figure(out, closure)
    report(out, c2, closure)


if __name__ == "__main__":
    main()
