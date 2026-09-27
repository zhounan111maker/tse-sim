"""TSE §4 仿真臂：消融实验 A/B/B1-B5 + F1c（生存域二阶环）+ F1d（V_ref 打乱）。

运行：python3 run_experiments.py
输出：results/ 下 4 张图与 summary.json，控制台打印指标表。

对应论文预测：
- F1a  V 对行动因果：B 的调节概率与效价显著相关，B2（去 V）相关消失。
- F1b  不可逆性：B3（去 S_wear 与熵耦合）生存退化，行为接近恒温器。
- F1c  Θ 收敛：去掉外部生存域后，域宽 W_T 收敛到有限值（不发散、不塌缩）。
- F1d  V_ref 因果：打乱 V_ref 后调节行为失去方向性（相关趋零）。
- F1e  kappa 耦合：B4（去 kappa）预测误差与行为与 B 分离。
- F1f  整体对照：B 与 A 在所有指标上应显著差异。
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tse_model as tm

CONFIGS = {
    "A":  dict(use_M=False, use_V=False, use_kappa=False, use_wear=True,
               use_entropy_coupling=True, integrate_M=False),
    "B":  dict(use_M=True, use_V=True, use_kappa=True, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    "B1": dict(use_M=False, use_V=True, use_kappa=True, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    "B2": dict(use_M=True, use_V=False, use_kappa=True, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    "B3": dict(use_M=True, use_V=True, use_kappa=True, use_wear=False,
               use_entropy_coupling=False, integrate_M=True),
    "B4": dict(use_M=True, use_V=True, use_kappa=False, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    "B5": dict(use_M=True, use_V=True, use_kappa=True, use_wear=True,
               use_entropy_coupling=True, integrate_M=False),
}

N_EP = 3
MAX_STEPS = 1500


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_config(name: str, cfg: dict, n_ep: int = N_EP, seed: int = 0,
               v_ref: float | None = None, permute_vref: bool = False) -> dict:
    env = tm.TSEEnv(seed=seed)
    if v_ref is None:
        v_ref = tm.estimate_v_ref(env)
    agent = tm.TSEAgent(env, cfg, v_ref=v_ref, seed=seed)
    agent.permute_vref = permute_vref
    ep_steps, ep_costs = [], []
    pred_errs, V_all, p_reg_all, rel_all, starv_all = [], [], [], [], []
    death_reasons: list[str] = []
    BURN_IN = 120   # 自我模型预热期（M 未训练，V≡0），不计入耦合指标
    for _ in range(n_ep):
        res = tm.run_episode(agent, MAX_STEPS)
        ep_steps.append(res["steps"])
        ep_costs.append(res["cum_cost"] / max(1, res["steps"]))
        if res["pred_err"] > 0:
            pred_errs.append(res["pred_err"])
        V_all.append(res["V"][BURN_IN:])
        p_reg_all.append(res["p_reg"][BURN_IN:])
        rel_all.append(res["reg_relevant"][BURN_IN:])
        starv_all.append(res["starving"][BURN_IN:])
        death_reasons.append(death_reason(agent))
    V = np.concatenate(V_all) if V_all else np.array([])
    P = np.concatenate(p_reg_all) if p_reg_all else np.array([])
    R = np.concatenate(rel_all) if rel_all else np.array([])
    St = np.concatenate(starv_all) if starv_all else np.array([])
    corr = pearson(P, V)
    # 效价-调节耦合只在"调节相关且非饥饿"步上计算：
    # 饥饿期正确行为是觅食，调节通道不参与，不应混入指标。
    mask = R.astype(bool) & (~St.astype(bool))
    reg_corr = pearson(P[mask], V[mask]) if mask.sum() > 10 else 0.0
    from collections import Counter
    reason_dist = dict(Counter(death_reasons))
    return {
        "config": name,
        "mean_steps": float(np.mean(ep_steps)),
        "alive_at_end": float(np.mean([a >= MAX_STEPS for a in ep_steps])),
        "cost_per_step": float(np.mean(ep_costs)),
        "pred_err": float(np.mean(pred_errs)) if pred_errs else 0.0,
        "valence_corr": corr,
        "valence_reg_corr": reg_corr,
        "abs_valence_corr": abs(reg_corr),
        "reg_relevant_frac": float(mask.mean()),
        "death_reasons": reason_dist,
    }


def death_reason(agent: tm.TSEAgent) -> str:
    if agent.E <= 0:
        return "energy"
    if agent.S_load >= tm.S_LOAD_MAX:
        return "load"
    if agent.S_wear >= tm.S_WEAR_MAX:
        return "wear"
    if agent.t_out > agent.out_dead:
        return "temp"
    return "alive"


def run_theta_experiment(seed: int = 0, n_seg: int = 60, seg_len: int = 500) -> list:
    """F1c：去掉外部生存域，域宽 W_T 由行为与成本共同决定（挣来的域）。"""
    env = tm.TSEEnv(seed=seed)
    v_ref = tm.estimate_v_ref(env)
    cfg = CONFIGS["B"]
    W = 8.0
    hist = []
    for seg in range(n_seg):
        agent = tm.TSEAgent(env, cfg, v_ref=v_ref, seed=seed + seg)
        agent.T_min = tm.T_STAR - W / 2.0
        agent.T_max = tm.T_STAR + W / 2.0
        agent.out_dead = 30.0   # F1c：收紧出界死亡阈值，窄域威胁真实
        theta_cost = 0.015 * W * W          # 域宽维持成本 c_Theta(W)
        agent.cfg = dict(agent.cfg, theta_cost=theta_cost)
        tm.run_episode(agent, seg_len, keep_wear=False)
        reason = death_reason(agent)
        if reason == "temp":
            W += 0.5        # 温度死亡 -> 扩宽温度域
        elif reason in ("energy", "load"):
            W -= 0.3        # 能量/负荷死亡 -> 收窄省能量（防作弊）
        else:
            W -= 0.05       # 存活 -> 收缩到最小可行宽度
        W = max(2.5, min(14.0, W))
        hist.append(W)
    return hist


def plot_survival(results: list[dict], path: str) -> None:
    names = [r["config"] for r in results]
    steps = [r["mean_steps"] for r in results]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, steps, color="#4f6ef7")
    for b, v in zip(bars, steps):
        plt.text(b.get_x() + b.get_width() / 2, v + 10, f"{v:.0f}",
                 ha="center", fontsize=9)
    plt.axhline(MAX_STEPS, ls="--", color="#9aa1ad", lw=1)
    plt.text(6.4, MAX_STEPS + 10, "max", fontsize=8, color="#9aa1ad")
    plt.ylabel("mean survival steps")
    plt.title("Ablation: mean survival steps (3 episodes)")
    plt.ylim(0, MAX_STEPS * 1.15)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_valence_corr(path: str, items: list[tuple[str, float]]) -> None:
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    colors = ["#16a34a" if v < -0.15 else "#dc2626" for v in vals]
    plt.figure(figsize=(6.5, 3.8))
    plt.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        plt.text(i, v + (0.02 if v >= 0 else -0.08), f"{v:.2f}",
                 ha="center", fontsize=9)
    plt.axhline(0, color="#1f2430", lw=0.8)
    plt.ylabel("corr(p_reg, V) | reg-relevant steps")
    plt.title("Valence-action coupling (F1a: B vs B2; F1d: B vs shuffled V_ref)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_pred_err(results: list[dict], path: str) -> None:
    names = [r["config"] for r in results if r["config"] in ("B", "B4", "B5")]
    vals = [r["pred_err"] for r in results if r["config"] in ("B", "B4", "B5")]
    plt.figure(figsize=(5.5, 3.6))
    plt.bar(names, vals, color="#4f6ef7")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)
    plt.ylabel("self-prediction MSE")
    plt.title("Self-model prediction error (B vs B4 kappa-off vs B5 no-integration)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_theta(hist: list[float], path: str) -> None:
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(hist, color="#0f766e", lw=2)
    plt.axhline(hist[-1], ls="--", color="#9aa1ad", lw=1)
    plt.text(len(hist) - 1, hist[-1] + 0.15, f"final W={hist[-1]:.1f}",
             ha="right", fontsize=9, color="#0f766e")
    plt.xlabel("segment")
    plt.ylabel("temp domain width W_T")
    plt.title("F1c: domain width converges to finite value (earned, not given)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    os.makedirs("results", exist_ok=True)
    results = [run_config(name, cfg) for name, cfg in CONFIGS.items()]
    # F1d：完整 B 但打乱 V_ref
    results.append(run_config("B_permVref", CONFIGS["B"], permute_vref=True))

    hist = run_theta_experiment()
    theta_final = hist[-1]
    theta_span = max(hist) - min(hist)
    theta_note = (
        "converged-finite" if 0 < theta_span < 4 and 2.5 < theta_final < 14
        else "check"
    )

    plot_survival(results, "results/fig1_survival.png")
    plot_valence_corr(
        "results/fig2_valence.png",
        [(r["config"], r["valence_reg_corr"]) for r in results
         if r["config"] in ("B", "B1", "B2", "B_permVref")],
    )
    plot_pred_err(results, "results/fig3_prederr.png")
    plot_theta(hist, "results/fig4_theta.png")

    summary = {
        "ablation": results,
        "theta_experiment": {
            "final_width": theta_final,
            "width_span": theta_span,
            "verdict": theta_note,
        },
    }
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 66)
    print("TSE §4 仿真臂结果（A/B/B1-B5 + F1c + F1d）")
    print("=" * 66)
    print(f"{'config':<12}{'survival':>10}{'cost/step':>11}{'pred_err':>11}"
          f"{'reg_corr':>11}{'death':>14}")
    for r in results:
        pred = f"{r['pred_err']:.4f}" if r["pred_err"] > 0 else "   --   "
        reasons = ",".join(f"{k}:{v}" for k, v in r["death_reasons"].items())
        print(f"{r['config']:<12}{r['mean_steps']:>10.0f}"
              f"{r['cost_per_step']:>11.3f}{pred:>11}"
              f"{r['valence_reg_corr']:>11.3f}{reasons:>14}")
    print("-" * 66)
    print(f"F1c 生存域二阶环：W_T 终值 {theta_final:.2f}，波动范围 {theta_span:.2f}"
          f"，判定 {theta_note}")
    print("结果图已写入 results/：fig1_survival.png / fig2_valence.png /"
          " fig3_prederr.png / fig4_theta.png")


if __name__ == "__main__":
    main()
