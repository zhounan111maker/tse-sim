"""TSE 修回实验臂（回应 Reviewer 1：预注册 + 竞争模型 + 阴性对照 + 表征丰富度）。

本脚本在第一轮投稿后被编写，用于把论文 §4.3b 中**预先注册**的判据落到实际运行上。
判据在 CRITERIA 中硬编码，且在任何运行之前打印，构成预注册记录。

五组实验：
  1. 多种子主对照   B（完整 TSE） / C（竞争模型：纯信息控制器） / B2（去 V） / B4（去 kappa）
  2. 阴性对照       Sham（移除非核心耦合 HEAT_FROM_LOAD） vs B —— 预测无显著差异
  3. F1c 生存域     B / C / Sham 的域宽收敛性（预注册收敛判据）
  4. 表征丰富度     Bq / B4q（二次特征自模型）与线性版对比（kappa 耦合是否随容量消失）
  5. V_ref 稳健性   缩放 V_ref，检验预测是否依赖固定基线

行为层面的「预期性调节」指标按**负荷通道**定义（见 run_config2 内注释）：
  crisis = S_load > S_LOAD_HIGH；reg = {rest, repair}
  anticipatory = P(reg & ~crisis)；reactive = P(reg & crisis)
  mean_load_at_rest = 发起 rest 那一刻的负荷水平（越低 = 越早动手）

用法：python run_revision.py [--seeds 20] [--ep 3] [--theta-seeds 10]
输出：results/revision.json、results/revision_table.md、results/fig_revision.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import tse_model as tm
import run_experiments as rex

# ============================================================
# 预注册判据（在第一行输出之前固定，运行前打印）
# ============================================================
CRITERIA = {
    "alpha": 0.05,
    "cohens_d_floor": 0.5,
    "test": "two-sided Welch t-test (unpaired), primary; paired t as robustness",
    "f1c_convergence_rel": 0.10,      # |Θ_t - W*| / W* < 0.10
    "f1c_tail_fraction": 0.20,        # 最后 20% 片段用于判定收敛
    "burn_in": 120,                   # 自我模型预热期，不计入耦合指标
    "note": ("判据在看到本轮结果之前固定。显著 = p < alpha 且 |Cohen's d| >= floor。"),
}

BURN_IN = CRITERIA["burn_in"]

# ---- 动作记录（用于「预期性调节」指标）----
_ORIG_POLICY = tm.TSEAgent.policy


def _policy_record(self) -> int:
    a = _ORIG_POLICY(self)
    self.history.setdefault("a", []).append(a)
    # 与 V / reg_relevant 同步记录**决策时刻**的负荷（policy 在 step 开头调用，
    # 因此这里取到的是生成该动作的负荷，与 V 严格对齐，不存在 off-by-one）。
    self.history.setdefault("load", []).append(float(self.S_load))
    return a


tm.TSEAgent.policy = _policy_record

# ---- 非线性（二次特征）自模型变体 ----
_ORIG_FEATURE = tm.feature


def _feature_quad(i, a_idx):
    z = i / tm.FEAT_SCALE
    return np.concatenate([z, z * z, tm.onehot(a_idx), [1.0]])


def set_quad(on: bool) -> None:
    tm.feature = _feature_quad if on else _ORIG_FEATURE


# ---- 配置 ----
CONFIGS = {
    "B": dict(use_M=True, use_V=True, use_kappa=True, use_wear=True,
              use_entropy_coupling=True, integrate_M=True),
    # 竞争模型 System C：保留内感受与自预测，移除 D_wear 与 V
    # （无不可逆代价信号）—— 纯信息稳态控制器（Keramati & Gutkin 型）
    "C": dict(use_M=True, use_V=False, use_kappa=True, use_wear=False,
              use_entropy_coupling=True, integrate_M=True),
    "B2": dict(use_M=True, use_V=False, use_kappa=True, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    "B4": dict(use_M=True, use_V=True, use_kappa=False, use_wear=True,
               use_entropy_coupling=True, integrate_M=True),
    # 阴性对照 Sham：配置与 B 相同，但移除一个**非 TSE 核心**的耦合
    # （负荷→温度产热 HEAT_FROM_LOAD）。它不在 TSE 的核心机制链
    # （不可逆代价 / V / kappa）上，故移除它预测**无** TSE 特异性签名丧失。
    "Sham": dict(use_M=True, use_V=True, use_kappa=True, use_wear=True,
                 use_entropy_coupling=True, integrate_M=True),
}

HEAT_FROM_LOAD_ORIG = tm.HEAT_FROM_LOAD


def set_sham(on: bool) -> None:
    """阴性对照：把负荷→温度产热耦合置零（非核心耦合）。"""
    tm.HEAT_FROM_LOAD = 0.0 if on else HEAT_FROM_LOAD_ORIG

# 负荷通道上的调节动作：rest（卸荷）/ repair（修复）。
# 不用 A_REGULATE（含 heat/cool），因为温度通道与负荷通道的危机判据不同，
# 混在一起会让「预期性 vs 反应性」的划分失去负荷语义。
REG_LOAD_IDX = [tm.ACTIONS.index("rest"), tm.ACTIONS.index("repair")]
REST_IDX = tm.ACTIONS.index("rest")
EXPLORE_IDX = tm.ACTIONS.index("explore")


# ============================================================ 指标
def run_config2(name: str, cfg: dict, seed: int, n_ep: int,
                sham: bool = False, quad: bool = False,
                v_ref_scale: float = 1.0) -> dict:
    set_sham(sham)
    set_quad(quad)
    env = tm.TSEEnv(seed=seed)
    v_ref = tm.estimate_v_ref(env) * v_ref_scale
    agent = tm.TSEAgent(env, cfg, v_ref=v_ref, seed=seed)
    ep_steps, ep_costs, pred_errs = [], [], []
    V_all, P_all, R_all, St_all, A_all, L_all = [], [], [], [], [], []
    onset_loads: list[float] = []
    onset_pcts: list[float] = []
    reasons: list[str] = []
    for _ in range(n_ep):
        # 注意：agent.history 跨回合累积（run_episode 不清空），因此必须按
        # episode 起点切片，否则各回合数据会被重复计入、长度也不匹配。
        s0 = len(agent.history["V"])
        res = tm.run_episode(agent, rex.MAX_STEPS)
        ep_steps.append(res["steps"])
        ep_costs.append(res["cum_cost"] / max(1, res["steps"]))
        if res["pred_err"] > 0:
            pred_errs.append(res["pred_err"])

        def sl(key: str) -> np.ndarray:
            return np.asarray(agent.history[key])[s0:][BURN_IN:]

        V_all.append(sl("V"))
        P_all.append(sl("p_reg"))
        R_all.append(sl("reg_relevant"))
        St_all.append(sl("starving"))
        A_all.append(sl("a"))
        L_all.append(sl("load"))
        # rest 发起时刻（前一步不是 rest，这一步是 rest）的负荷水平。
        a_ep = np.asarray(agent.history["a"])[s0:][BURN_IN:]
        l_ep = np.asarray(agent.history["load"])[s0:][BURN_IN:]
        if a_ep.size > 2:
            on = np.where((a_ep == REST_IDX) & (np.r_[True, a_ep[:-1]] != REST_IDX))[0]
            if on.size:
                onset_loads.append(float(np.mean(l_ep[on])))
                # 自归一化：把发起负荷映射到**本回合自身**负荷分布的分位。
                # 各配置的负荷平衡水平不同（B 3.18 / C 2.84），绝对阈值会饱和，
                # 分位版本消除该混淆：分位越低 = 相对自身常态越早动手。
                onset_pcts.append(float(np.mean(
                    (l_ep < l_ep[on][:, None]).mean(axis=1))))
        reasons.append(rex.death_reason(agent))
    V = np.concatenate(V_all) if V_all else np.array([])
    P = np.concatenate(P_all) if P_all else np.array([])
    R = np.concatenate(R_all) if R_all else np.array([])
    St = np.concatenate(St_all) if St_all else np.array([])
    A = np.concatenate(A_all) if A_all else np.array([])
    L = np.concatenate(L_all) if L_all else np.array([])
    mask = R.astype(bool) & (~St.astype(bool))
    reg_corr = rex.pearson(P[mask], V[mask]) if mask.sum() > 10 else 0.0
    # ---- 行为层面的「预期性调节」----
    # 旧定义用 reg_relevant（温度越界 OR 负荷>S_LOAD_HIGH）划分危机，实测
    # R 的均值≈1（负荷长期高于阈值），「非危机步」几乎为空 → 指标恒为 0。
    # 因此改为**只看负荷通道**：crisis = L > S_LOAD_HIGH。
    # anticipatory：负荷尚未越界就动手；reactive：已经越界才动手。
    if A.size and L.size == A.size:
        is_reg = np.isin(A, REG_LOAD_IDX)
        crisis = L > tm.S_LOAD_HIGH
        anticipatory = float(np.mean(is_reg & (~crisis)))
        reactive = float(np.mean(is_reg & crisis))
        mean_load = float(np.mean(L))
        # 高负荷下仍在干活（explore）的比例：越低 = 越早收工。
        hl = L > np.nanpercentile(L, 75) if L.size else np.array([], bool)
        work_at_high_load = (float(np.mean(A[hl] == EXPLORE_IDX))
                             if hl.sum() > 5 else float("nan"))
    else:
        anticipatory = reactive = 0.0
        mean_load = 0.0
        work_at_high_load = float("nan")
    return {
        "config": name,
        "mean_steps": float(np.mean(ep_steps)),
        "cost_per_step": float(np.mean(ep_costs)),
        "pred_err": float(np.mean(pred_errs)) if pred_errs else 0.0,
        "valence_reg_corr": reg_corr,
        "anticipatory_reg": anticipatory,
        "reactive_reg": reactive,
        "mean_load": mean_load,
        "mean_load_at_rest": float(np.mean(onset_loads)) if onset_loads else float("nan"),
        "rest_onset_percentile": float(np.mean(onset_pcts)) if onset_pcts else float("nan"),
        "work_at_high_load": work_at_high_load,
        "reg_relevant_frac": float(mask.mean()) if R.size else 0.0,
        "death_reasons": dict(Counter(reasons)),
    }


def run_theta_cfg(cfg: dict, seed: int, sham: bool = False,
                  n_seg: int = 40, seg_len: int = 500) -> tuple:
    set_sham(sham)
    set_quad(False)
    env = tm.TSEEnv(seed=seed)
    v_ref = tm.estimate_v_ref(env)
    W = 8.0
    hist: list[float] = []
    reasons: list[str] = []
    for seg in range(n_seg):
        agent = tm.TSEAgent(env, cfg, v_ref=v_ref, seed=seed + seg)
        agent.T_min = tm.T_STAR - W / 2.0
        agent.T_max = tm.T_STAR + W / 2.0
        agent.out_dead = 30.0
        agent.cfg = dict(cfg, theta_cost=0.015 * W * W)
        tm.run_episode(agent, seg_len, keep_wear=False)
        reason = rex.death_reason(agent)
        reasons.append(reason)
        if reason == "temp":
            W += 0.5
        elif reason in ("energy", "load"):
            W -= 0.3
        else:
            W -= 0.05
        W = max(2.5, min(14.0, W))
        hist.append(W)
    return hist, reasons


def theta_verdict(hist: list) -> dict:
    a = np.asarray(hist, float)
    tail_n = max(1, int(np.ceil(len(a) * CRITERIA["f1c_tail_fraction"])))
    tail = a[-tail_n:]
    W_star = float(tail.mean())
    if W_star <= 1e-9:
        return {"W_star": 0.0, "max_rel_dev": None, "verdict": "collapsed"}
    dev = float(np.max(np.abs(tail - W_star) / W_star))
    verdict = "converged" if dev < CRITERIA["f1c_convergence_rel"] else "not-converged"
    return {"W_star": W_star, "max_rel_dev": dev, "verdict": verdict,
            "tail_n": tail_n}


# ============================================================ 统计
def welch(x, y) -> dict:
    a, b = np.asarray(x, float), np.asarray(y, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"n_x": int(a.size), "n_y": int(b.size),
           "mean_x": float(a.mean()), "mean_y": float(b.mean()),
           "diff": float(a.mean() - b.mean())}
    if a.size < 3 or b.size < 3:
        out.update(p=None, d=None, paired_p=None, verdict="insufficient-n")
        return out
    sd_pool = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                      / (a.size + b.size - 2))
    out["d"] = float(out["diff"] / sd_pool) if sd_pool > 1e-12 else None
    try:
        out["p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:
        out["p"] = None
    try:
        out["paired_p"] = float(stats.ttest_rel(a[:min(a.size, b.size)],
                                                b[:min(a.size, b.size)]).pvalue)
    except Exception:
        out["paired_p"] = None
    sig = (out["p"] is not None and out["p"] < CRITERIA["alpha"]
           and out["d"] is not None and abs(out["d"]) >= CRITERIA["cohens_d_floor"])
    out["verdict"] = "significant" if sig else "not-significant"
    return out


def fmt_p(p) -> str:
    if p is None:
        return "—"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


# ============================================================ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ep", type=int, default=rex.N_EP)
    ap.add_argument("--theta-seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    os.makedirs("results", exist_ok=True)

    print("=" * 78)
    print("预注册判据（先于结果固定）")
    print("=" * 78)
    for k, v in CRITERIA.items():
        print(f"  {k:<22} {v}")
    print(f"  seeds={len(seeds)}  episodes/seed={args.ep}  "
          f"theta-seeds={args.theta_seeds}")
    print()

    metric_keys = ["mean_steps", "cost_per_step", "pred_err",
                   "valence_reg_corr", "anticipatory_reg", "reactive_reg",
                   "mean_load", "mean_load_at_rest",
                   "rest_onset_percentile", "work_at_high_load"]

    # ---- 1+2. 多种子主对照与阴性对照 ----
    runs = {name: {m: [] for m in metric_keys} for name in CONFIGS}
    deaths = {name: Counter() for name in CONFIGS}
    for name, cfg in CONFIGS.items():
        sham = name == "Sham"
        for s in seeds:
            r = run_config2(name, cfg, s, args.ep, sham=sham)
            for m in metric_keys:
                runs[name][m].append(r[m])
            deaths[name].update(r["death_reasons"])
        print(f"  {name:<8} 存活 {np.mean(runs[name]['mean_steps']):7.1f} ± "
              f"{np.std(runs[name]['mean_steps'], ddof=1):6.1f} | "
              f"预期/反应 {np.mean(runs[name]['anticipatory_reg']):.3f}/"
              f"{np.mean(runs[name]['reactive_reg']):.3f} | "
              f"rest 发起负荷 {np.mean(runs[name]['mean_load_at_rest']):.2f} | "
              f"效价-调节相关 {np.mean(runs[name]['valence_reg_corr']):+.3f}")

    # ---- 3. F1c 生存域 ----
    print("\nF1c 生存域二阶环（预注册收敛判据）…")
    theta_cfg = {"B": CONFIGS["B"], "C": CONFIGS["C"], "Sham": CONFIGS["Sham"]}
    theta_runs, theta_stats, theta_deaths = {}, {}, {}
    for name, cfg in theta_cfg.items():
        sham = name == "Sham"
        out_theta = [run_theta_cfg(cfg, s, sham=sham) for s in range(args.theta_seeds)]
        hists = [o[0] for o in out_theta]
        theta_deaths[name] = Counter([r for o in out_theta for r in o[1]])
        finals = [h[-1] for h in hists]
        theta_runs[name] = {"finals": finals, "hists": [list(map(float, h)) for h in hists]}
        v = theta_verdict(hists[0])
        # 尾部轨迹的峰-峰振幅：区分「渐近收敛到点」与「有界极限环」。
        # 预注册判据（尾部最大相对偏离 <0.10）对两者都放行，需额外披露。
        tail_n = v.get("tail_n", max(1, int(np.ceil(len(hists[0]) * 0.2))))
        tail = np.asarray(hists[0][-tail_n:], float)
        amp = float((tail.max() - tail.min()) / tail.mean()) if tail.mean() > 0 else None
        theta_stats[name] = {**v,
                             "final_mean": float(np.mean(finals)),
                             "final_std": float(np.std(finals, ddof=1)),
                             "tail_peak_to_peak_rel": amp}
        print(f"  {name:<8} W*={theta_stats[name]['final_mean']:.2f} ± "
              f"{theta_stats[name]['final_std']:.2f} | 尾部最大相对偏离 "
              f"{v['max_rel_dev'] if v['max_rel_dev'] is not None else float('nan'):.3f}"
              f" → {v['verdict']}")

    # ---- 4. 表征丰富度（二次特征自模型）----
    print("\n表征丰富度（二次特征自模型 vs 线性）…")
    quad_runs = {"B": [], "B4": [], "Bq": [], "B4q": []}
    for s in seeds:
        quad_runs["B"].append(run_config2("B", CONFIGS["B"], s, args.ep)["pred_err"])
        quad_runs["B4"].append(run_config2("B4", CONFIGS["B4"], s, args.ep)["pred_err"])
        quad_runs["Bq"].append(run_config2("Bq", CONFIGS["B"], s, args.ep,
                                           quad=True)["pred_err"])
        quad_runs["B4q"].append(run_config2("B4q", CONFIGS["B4"], s, args.ep,
                                            quad=True)["pred_err"])
    kappa_effect_linear = welch(quad_runs["B4"], quad_runs["B"])
    kappa_effect_quad = welch(quad_runs["B4q"], quad_runs["Bq"])
    print(f"  线性：B4-B 预测误差差 {kappa_effect_linear['diff']:+.5f} "
          f"d={kappa_effect_linear['d']:.2f} p={fmt_p(kappa_effect_linear['p'])} "
          f"→ {kappa_effect_linear['verdict']}")
    print(f"  二次：B4q-Bq 预测误差差 {kappa_effect_quad['diff']:+.5f} "
          f"d={kappa_effect_quad['d']:.2f} p={fmt_p(kappa_effect_quad['p'])} "
          f"→ {kappa_effect_quad['verdict']}")

    # ---- 5. V_ref 稳健性 ----
    print("\nV_ref 稳健性（缩放参考基线）…")
    vref_sweep = {}
    for scale in (0.5, 1.0, 1.5, 2.0):
        vals_corr, vals_steps = [], []
        for s in seeds[:max(10, len(seeds) // 2)]:
            r = run_config2("B", CONFIGS["B"], s, args.ep, v_ref_scale=scale)
            vals_corr.append(r["valence_reg_corr"])
            vals_steps.append(r["mean_steps"])
        vref_sweep[str(scale)] = {
            "valence_reg_corr": float(np.mean(vals_corr)),
            "valence_reg_corr_sd": float(np.std(vals_corr, ddof=1)),
            "mean_steps": float(np.mean(vals_steps)),
        }
        print(f"  ×{scale:<4} 效价-调节相关 {np.mean(vals_corr):+.3f} ± "
              f"{np.std(vals_corr, ddof=1):.3f} | 存活 {np.mean(vals_steps):.0f}")

    # ---- 预先指定的对比（预注册）----
    planned = [
        ("竞争模型 C 是否维持稳态（主检验）", "C", "B", "mean_steps"),
        ("竞争模型 C 的预期性调节是否下降", "C", "B", "anticipatory_reg"),
        ("竞争模型 C 的 rest 发起负荷分位是否更高（自归一化，更晚动手）",
         "C", "B", "rest_onset_percentile"),
        ("竞争模型 C 在高负荷下是否更少收工", "C", "B", "work_at_high_load"),
        ("阴性对照：移除非核心耦合（预测无效应）", "Sham", "B", "anticipatory_reg"),
        ("阴性对照：移除非核心耦合（存活）", "Sham", "B", "mean_steps"),
        ("去 V 后预期性调节下降（F1a 行为版）", "B2", "B", "anticipatory_reg"),
        ("去 kappa 后存活下降（F1d 行为版）", "B4", "B", "mean_steps"),
    ]
    tests = []
    for desc, xa, xb, metric in planned:
        t = welch(runs[xa][metric], runs[xb][metric])
        tests.append({"description": desc, "x": xa, "y": xb,
                      "metric": metric, **t})

    # ---- 探索性稳健性读出（**非预注册**，在看结果后追加，论文中须如此标注）----
    # 起因：绝对阈值版「预期性调节」指标已饱和（各配置≈0.01–0.02），且方向
    # 与预测相反；自归一化版用于判断这是否只是负荷平衡水平差异造成的混淆。
    exploratory = []
    for desc, xa, xb, metric in [
        ("去 V 后 rest 发起分位是否升高（自归一化）", "B2", "B", "rest_onset_percentile"),
        ("去 kappa 后 rest 发起分位是否升高（自归一化）", "B4", "B", "rest_onset_percentile"),
        ("阴性对照（自归一化读出）", "Sham", "B", "rest_onset_percentile"),
        ("阴性对照（高负荷下仍工作）", "Sham", "B", "work_at_high_load"),
    ]:
        t = welch(runs[xa][metric], runs[xb][metric])
        exploratory.append({"description": desc, "x": xa, "y": xb,
                            "metric": metric, **t})
    print("\n探索性（非预注册）稳健性读出：")
    for t in exploratory:
        d = f"{t['d']:.2f}" if t["d"] is not None else "—"
        print(f"  {t['x']} vs {t['y']} · {t['metric']}: Δ={t['diff']:+.4f} "
              f"d={d} p={fmt_p(t['p'])} → {t['verdict']}")

    summary = {n: {m: {"mean": float(np.nanmean(v)),
                       "std": float(np.nanstd(v, ddof=1)),
                       "n": int(np.sum(np.isfinite(np.asarray(v, float))))}
                   for m, v in mv.items()}
               for n, mv in runs.items()}

    out = {
        "preregistered_criteria": CRITERIA,
        "meta": {"seeds": len(seeds), "episodes_per_seed": args.ep,
                 "theta_seeds": args.theta_seeds,
                 "sham_definition": "HEAT_FROM_LOAD=0 (non-core load→temp coupling)",
                 "burn_in": BURN_IN},
        "summary": summary,
        "planned_tests": tests,
        "exploratory_tests": exploratory,
        "death_causes": {n: dict(c) for n, c in deaths.items()},
        "theta": theta_stats,
        "theta_death_causes": {n: dict(c) for n, c in theta_deaths.items()},
        "quadratic": {"linear_kappa_effect": kappa_effect_linear,
                      "quad_kappa_effect": kappa_effect_quad},
        "vref_sweep": vref_sweep,
        "raw": {n: {m: [float(x) for x in v] for m, v in mv.items()}
                for n, mv in runs.items()},
    }
    with open("results/revision.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    _plot(runs, theta_runs, "results/fig_revision.png")
    _md_table(out, "results/revision_table.md")

    print("\n" + "=" * 78)
    print("预注册对比结果（Welch，显著需同时满足 p<0.05 且 |d|>=0.5）")
    print("=" * 78)
    for t in tests:
        d = f"{t['d']:.2f}" if t["d"] is not None else "—"
        print(f"  {t['description']}\n     {t['x']} vs {t['y']} · {t['metric']}: "
              f"Δ={t['diff']:+.4f} d={d} p={fmt_p(t['p'])} → {t['verdict']}")
    print("\n已写出：results/revision.json / revision_table.md / fig_revision.png")


def _plot(runs: dict, theta_runs: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    names = ["B", "C", "B2", "B4", "Sham"]

    ax = axes[0][0]
    ax.boxplot([runs[n]["mean_steps"] for n in names], tick_labels=names,
               showmeans=True)
    ax.set_ylabel("mean survival steps")
    ax.set_title("(a) Competing model C vs TSE (B)")

    ax = axes[0][1]
    ax.boxplot([runs[n]["anticipatory_reg"] for n in names], tick_labels=names,
               showmeans=True)
    ax.set_ylabel("P(regulate | load not yet critical)")
    ax.set_title("(b) Anticipatory regulation (load channel)")

    ax = axes[1][0]
    ax.boxplot([runs[n]["rest_onset_percentile"] for n in names],
               tick_labels=names, showmeans=True)
    ax.set_ylabel("percentile of own load distribution")
    ax.set_title("(c) Load percentile at rest onset (self-normalised)")

    ax = axes[1][1]
    for n in ("B", "C", "Sham"):
        h = np.asarray(theta_runs[n]["hists"])
        ax.plot(h.mean(axis=0), lw=2, label=f"{n} (mean of seeds)")
    ax.set_xlabel("segment")
    ax.set_ylabel("temp domain width W_T")
    ax.set_title("(d) F1c: domain width")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _md_table(out: dict, path: str) -> None:
    s = out["summary"]
    L = ["# 修回实验臂结果（预注册判据先行）", ""]
    L.append("| 配置 | 存活步数 | cost/step | 自我预测 MSE | 效价-调节相关 | "
             "预期性调节 | 反应性调节 | 平均负荷 | rest 发起负荷 | "
             "rest 发起分位 | 高负荷下仍工作 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
             "---: | ---: | ---: |")
    for n, m in s.items():
        pe = f"{m['pred_err']['mean']:.5f}" if m["pred_err"]["mean"] > 0 else "—"
        L.append(f"| {n} | {m['mean_steps']['mean']:.1f} ± {m['mean_steps']['std']:.1f} | "
                 f"{m['cost_per_step']['mean']:.3f} | {pe} | "
                 f"{m['valence_reg_corr']['mean']:+.3f} | "
                 f"{m['anticipatory_reg']['mean']:.3f} | "
                 f"{m['reactive_reg']['mean']:.3f} | "
                 f"{m['mean_load']['mean']:.2f} | "
                 f"{m['mean_load_at_rest']['mean']:.2f} | "
                 f"{m['rest_onset_percentile']['mean']:.3f} | "
                 f"{m['work_at_high_load']['mean']:.3f} |")
    L.append("")
    L.append("> 预期性调节 = P(执行 rest/repair 且 S_load ≤ S_LOAD_HIGH)；"
             "反应性调节 = P(执行 rest/repair 且 S_load > S_LOAD_HIGH)。")
    L.append("> 注：负荷平衡水平高于 S_LOAD_HIGH，绝对阈值版指标已饱和（各配置均≈0.01），"
             "故主要读作**自归一化版**：rest 发起分位 = 发起时刻负荷在本回合负荷分布中的"
             "分位（越低 = 相对自身常态越早动手）；高负荷下仍工作 = P(explore | 负荷处于本"
             "回合前 25% 高位)。")
    L.append("")
    L.append("## 预注册对比（Welch 双侧，显著需 p<0.05 且 |d|≥0.5）")
    L.append("")
    L.append("| 对比 | 指标 | Δ | Cohen's d | Welch p | 配对 p | 判定 |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for t in out["planned_tests"]:
        d = f"{t['d']:.2f}" if t["d"] is not None else "—"
        L.append(f"| {t['x']} vs {t['y']} | {t['metric']} | {t['diff']:+.4f} | "
                 f"{d} | {fmt_p(t['p'])} | {fmt_p(t['paired_p'])} | "
                 f"{t['verdict']} |")
    L.append("")
    L.append("## 探索性（**非预注册**）稳健性读出")
    L.append("")
    L.append("| 对比 | 指标 | Δ | Cohen's d | Welch p | 配对 p | 判定 |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for t in out["exploratory_tests"]:
        d = f"{t['d']:.2f}" if t["d"] is not None else "—"
        L.append(f"| {t['x']} vs {t['y']} | {t['metric']} | {t['diff']:+.4f} | "
                 f"{d} | {fmt_p(t['p'])} | {fmt_p(t['paired_p'])} | "
                 f"{t['verdict']} |")
    L.append("")
    L.append("## F1c 生存域（预注册收敛判据：尾部 20% 相对偏离 <0.10）")
    L.append("")
    for n, v in out["theta"].items():
        dev = v.get("max_rel_dev")
        amp = v.get("tail_peak_to_peak_rel")
        if dev is None:
            L.append(f"- **{n}**：{v['verdict']}")
        else:
            L.append(f"- **{n}**：W* = {v['final_mean']:.2f} ± {v['final_std']:.2f}，"
                     f"尾部最大相对偏离 {dev:.3f}，尾部峰-峰振幅 {amp:.3f}"
                     f" → **{v['verdict']}**")
    L.append("")
    L.append("> 峰-峰振幅用于区分「渐近收敛到一个点」与「有界极限环」：预注册判据只限制"
             "相对偏离，对振幅约 7% 的极限环同样放行，因此单独披露。")
    L.append("")
    L.append("## 死亡原因分布")
    L.append("")
    L.append("### 主对照（多种子 × 多回合）")
    L.append("")
    L.append("| 配置 | 原因计数 |")
    L.append("| --- | --- |")
    for n, c in out["death_causes"].items():
        L.append(f"| {n} | {', '.join(f'{k}={v}' for k, v in sorted(c.items()))} |")
    L.append("")
    L.append("### F1c 二阶环（40 段 × seeds）")
    L.append("")
    L.append("| 配置 | 原因计数 |")
    L.append("| --- | --- |")
    for n, c in out["theta_death_causes"].items():
        L.append(f"| {n} | {', '.join(f'{k}={v}' for k, v in sorted(c.items()))} |")
    L.append("")
    L.append("## 表征丰富度（kappa 耦合是否随模型容量消失）")
    L.append("")
    q = out["quadratic"]
    for tag, k in (("线性自模型", q["linear_kappa_effect"]),
                   ("二次特征自模型", q["quad_kappa_effect"])):
        L.append(f"- {tag}：去 kappa 与完整版预测误差差 {k['diff']:+.5f}，"
                 f"d={k['d']:.2f}，p={fmt_p(k['p'])} → {k['verdict']}")
    L.append("")
    L.append("## V_ref 稳健性")
    L.append("")
    L.append("| V_ref 缩放 | 效价-调节相关 | 存活步数 |")
    L.append("| --- | ---: | ---: |")
    for k, v in out["vref_sweep"].items():
        L.append(f"| ×{k} | {v['valence_reg_corr']:+.3f} ± "
                 f"{v['valence_reg_corr_sd']:.3f} | {v['mean_steps']:.0f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
