"""TSE 多种子稳健性检验（补充材料专用）。

对 A/B/B1-B5 + B_permVref 共 8 个配置，在 SEEDS 个随机种子上重复，
报告 均值 ± 标准差，并对论文 F1a-F1f 的关键对比做配对显著性检验。

用法：
    python3 run_multiseed.py                 # 默认 30 个种子
    python3 run_multiseed.py --seeds 50      # 自定义种子数
    python3 run_multiseed.py --seeds 10 --ep 1

输出：
    results/multiseed.json        全部原始值 + 汇总 + 检验结果
    results/multiseed_table.md    可直接贴进补充材料的 markdown 表
    results/fig5_multiseed.png    分布箱线图

注意（诚实边界）：
本脚本测量的是**功能架构层**预测在随机种子上的稳健性。
它不构成对"热力学不可逆"这一承压假设的实验检验——该假设需硬件臂
（见 README「已知简化」第 5 条与论文 §4.6）。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import run_experiments as rex

ORDER = ["A", "B", "B1", "B2", "B3", "B4", "B5", "B_permVref"]
METRICS = [
    ("mean_steps", "survival steps", "存活步数"),
    ("cost_per_step", "cost/step", "cost/step"),
    ("pred_err", "self-pred MSE", "自我预测 MSE"),
    ("valence_reg_corr", "valence-reg corr", "效价-调节耦合"),
]

# (预测编号, 配置X, 配置Y, 指标, 说明, 预期方向)
PAIRS = [
    ("F1f", "A", "B", "mean_steps",
     "完整 TSE 与反射式控制器在存活步数上分离", "B > A"),
    ("F1b", "B", "B3", "mean_steps",
     "关闭不可逆性后存活显著延长（作弊态，非优点）", "B3 > B"),
    ("F1a", "B", "B2", "valence_reg_corr",
     "移除 V 后效价-调节耦合消失", "B < B2"),
    ("F1d", "B", "B_permVref", "valence_reg_corr",
     "打乱 V_ref 后调节失去方向性", "B < B_permVref"),
    ("F1e", "B", "B4", "pred_err",
     "移除 kappa 耦合后自我预测误差上升", "B4 > B"),
    ("F1e", "B", "B5", "pred_err",
     "移除 i 的全局整合后自我预测崩溃", "B5 > B"),
]


def summarise(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired_test(x: list[float], y: list[float]) -> dict:
    """配对检验：x、y 必须用同一批 seed 跑出，才能配对。"""
    a, b = np.asarray(x, float), np.asarray(y, float)
    diff = a - b
    out = {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "n": int(a.size),
    }
    if a.size < 3 or np.allclose(diff, 0):
        out["t_p"] = None
        out["wilcoxon_p"] = None
        out["cohens_d"] = None
        return out
    # Cohen's d（配对）
    sd = diff.std(ddof=1)
    out["cohens_d"] = float(diff.mean() / sd) if sd > 1e-12 else None
    try:
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        out["t_p"] = None
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        out["wilcoxon_p"] = None
    return out


def fmt_p(p) -> str:
    if p is None:
        return "—"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def plot_multiseed(data: dict, theta: list[float], path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    groups = ["A", "B", "B3"]
    ax.boxplot([data[g]["mean_steps"] for g in groups], tick_labels=groups,
               showmeans=True)
    ax.set_ylabel("mean survival steps")
    ax.set_title(f"Survival over {len(data['B']['mean_steps'])} seeds\n(F1f / F1b)")

    ax = axes[1]
    groups = ["B", "B2", "B_permVref"]
    ax.boxplot([data[g]["valence_reg_corr"] for g in groups],
               tick_labels=groups, showmeans=True)
    ax.axhline(0, color="#1f2430", lw=0.8)
    ax.set_ylabel("corr(p_reg, V)")
    ax.set_title("Valence-action coupling\n(F1a / F1d)")

    ax = axes[2]
    ax.hist(theta, bins=12, color="#0f766e", alpha=0.85)
    ax.axvline(float(np.mean(theta)), ls="--", color="#dc2626", lw=1.4,
               label=f"mean={np.mean(theta):.2f}")
    ax.set_xlabel("final temp-domain width W_T")
    ax.set_ylabel("seed count")
    ax.set_title("F1c: earned domain width\n(converges to finite value)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30,
                    help="随机种子个数（默认 30）")
    ap.add_argument("--ep", type=int, default=rex.N_EP,
                    help="每个种子跑的回合数（默认同 run_experiments.py）")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    n_ep = args.ep
    os.makedirs("results", exist_ok=True)

    print(f"多种子稳健性检验：{len(seeds)} seeds × {n_ep} episodes × "
          f"{len(ORDER)} 配置")

    data: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m, _, _ in METRICS} for name in ORDER
    }
    for name in ORDER:
        cfg = rex.CONFIGS["B"] if name == "B_permVref" else rex.CONFIGS[name]
        perm = name == "B_permVref"
        for s in seeds:
            r = rex.run_config(name, cfg, n_ep=n_ep, seed=s, permute_vref=perm)
            for m, _, _ in METRICS:
                data[name][m].append(r[m])
        summary = data[name]["mean_steps"]
        print(f"  {name:<12} 存活步数 {np.mean(summary):7.1f} "
              f"± {np.std(summary, ddof=1):6.1f}")

    print("F1c 生存域二阶环（多种子）…")
    theta = [rex.run_theta_experiment(seed=s)[-1] for s in seeds]
    theta_stat = summarise(theta)
    theta_verdict = (
        "converged-finite"
        if 2.5 < theta_stat["mean"] < 14 and theta_stat["std"] < 2.0
        else "check"
    )
    print(f"  W_T 终值 {theta_stat['mean']:.2f} ± {theta_stat['std']:.2f}"
          f"（判定 {theta_verdict}）")

    # ---- 汇总与配对检验 ----
    summary = {name: {m: summarise(data[name][m]) for m, _, _ in METRICS}
               for name in ORDER}

    tests = []
    for fid, xa, xb, metric, desc, direction in PAIRS:
        t = paired_test(data[xa][metric], data[xb][metric])
        tests.append({
            "prediction": fid, "config_x": xa, "config_y": xb,
            "metric": metric, "description": desc, "expected": direction,
            **t,
        })

    out = {
        "meta": {
            "seeds": len(seeds),
            "seed_range": [min(seeds), max(seeds)],
            "episodes_per_seed": n_ep,
            "max_steps": rex.MAX_STEPS,
            "paired_by_seed": True,
            "note": ("功能架构层预测的稳健性检验；不构成对热力学不可逆"
                     "承压假设的实验检验（需硬件臂）。"),
        },
        "summary": summary,
        "paired_tests": tests,
        "theta_multiseed": {**theta_stat, "verdict": theta_verdict,
                            "raw": [float(v) for v in theta]},
        "raw": data,
    }
    with open("results/multiseed.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    plot_multiseed(data, theta,
                   "results/fig5_multiseed.png")
    write_markdown_table(out)

    print("\n" + "=" * 78)
    print("多种子汇总（mean ± sd）")
    print("=" * 78)
    hdr = f"{'config':<12}" + "".join(f"{zh:>18}" for _, _, zh in METRICS)
    print(hdr)
    for name in ORDER:
        row = f"{name:<12}"
        for m, _, _ in METRICS:
            s = summary[name][m]
            row += f"{s['mean']:>10.3f}±{s['std']:<7.3f}"
        print(row)
    print("-" * 78)
    print("配对检验（同一批 seed，故为配对样本）")
    for t in tests:
        print(f"  [{t['prediction']}] {t['config_x']} vs {t['config_y']} · "
              f"{t['metric']}: Δ={t['mean_diff']:+.3f} "
              f"t-p={fmt_p(t['t_p'])} wilcoxon-p={fmt_p(t['wilcoxon_p'])}")
    print(f"\nF1c 域宽终值 {theta_stat['mean']:.2f} ± {theta_stat['std']:.2f}"
          f"（{theta_verdict}）")
    print("已写出：results/multiseed.json / multiseed_table.md / "
          "fig5_multiseed.png")


def write_markdown_table(out: dict) -> None:
    s = out["summary"]
    meta = out["meta"]
    L = []
    L.append(f"多种子稳健性（{meta['seeds']} seeds × "
             f"{meta['episodes_per_seed']} episodes，seed "
             f"{meta['seed_range'][0]}–{meta['seed_range'][1]}，"
             f"同一 seed 下各配置配对比较）\n")
    L.append("| 配置 | 存活步数 | cost/step | 自我预测 MSE | 效价-调节耦合 |")
    L.append("| --- | ---: | ---: | ---: | ---: |")

    def cell(stat: dict, dec: int = 3) -> str:
        return f"{stat['mean']:.{dec}f} ± {stat['std']:.{dec}f}"

    for name in ORDER:
        m = s[name]
        pred = cell(m["pred_err"]) if m["pred_err"]["mean"] > 0 else "—"
        L.append(f"| {name} | {cell(m['mean_steps'], 1)} | "
                 f"{cell(m['cost_per_step'])} | {pred} | "
                 f"{cell(m['valence_reg_corr'])} |")

    L.append("")
    L.append("配对显著性检验（同一批 seed → 配对样本）\n")
    L.append("| 预测 | 对比 | 指标 | 差值 (mean ± sd) | Cohen's d | "
             "配对 t 检验 p | Wilcoxon p |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for t in out["paired_tests"]:
        d = (f"{t['cohens_d']:.2f}" if t["cohens_d"] is not None else "—")
        L.append(f"| {t['prediction']} | {t['config_x']} vs {t['config_y']} | "
                 f"{t['metric']} | {t['mean_diff']:+.3f} ± "
                 f"{t['std_diff']:.3f} | {d} | {fmt_p(t['t_p'])} | "
                 f"{fmt_p(t['wilcoxon_p'])} |")

    th = out["theta_multiseed"]
    L.append("")
    L.append(f"**F1c（生存域二阶环，多种子）**：域宽终值 "
             f"{th['mean']:.2f} ± {th['std']:.2f}（n={th['n']}，"
             f"范围 {th['min']:.2f}–{th['max']:.2f}），判定 **{th['verdict']}**。")
    L.append("")
    L.append(f"> 口径说明：{meta['note']}")
    with open("results/multiseed_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
