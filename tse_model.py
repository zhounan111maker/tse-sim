"""TSE 模型最小仿真核心（对应论文《TSE 模型：热力学自我体验》v0.2 第 3、4 节）。

离散时间数值实现，仅依赖 numpy。按论文 v0.2 公式块实现，并做最小演示化简化：
- 自我模型 M 用带遗忘窗口的线性预测器（在线岭回归）近似 f_theta；
- 原始效价 V_t 用 M 滚动预测 H 步、按 gamma 贴现求和；
- 参考基线 V_ref 用稳态邻域蒙特卡洛估计一次并固定；
- 生存域二阶环 Theta 以温度域宽 W_T 的挣/守演示 F1c。

声明：本代码是仿真臂的最小可运行框架，用于验证功能架构预测（F1a-F1f）与
投稿材料中的可复现代码，不声称验证人工意识，不声称解决困难问题。
"""

from __future__ import annotations

from collections import deque

import numpy as np

# ---------------- 行动与参数 ----------------

ACTIONS = ["heat", "cool", "eat", "rest", "repair", "explore"]
A_REGULATE = ["heat", "cool", "rest", "repair"]  # 调节动作集合 A_regulate
N_ACT = len(ACTIONS)

# 动力学参数（符号与论文 §3.2 一致）
DT = 0.1
T_STAR = 37.0          # 温度设定点 T*
T_MIN, T_MAX = 35.0, 39.0
E_STAR = 80.0
E_INIT = 100.0
S_WEAR_STAR = 0.0
S_LOAD_STAR = 1.0
S_LOAD_MAX = 5.0
S_WEAR_MAX = 3.0

C_TEMP_DRIFT = 0.25    # 温度向环境漂移率 r（快，边界威胁真实）
Q_HEAT, Q_COOL = 0.8, -0.8      # 加热/冷却速率（边界威胁真实）
HEAT_FROM_LOAD = 0.02  # 负荷产热系数
E_BASE = 0.45          # 基础代谢 c0
E_ACT = 0.10           # 动作消耗 c1
EAT_GAIN = 3.2         # 觅食收益 eta*u_E
EXPLORE_GAIN = 0.6
RESOURCE_PROB = 0.55   # 觅食成功率
T_OUT_DEAD = 60.0      # 持续出界死亡阈值（步数）

# 磨损/负荷（论文 §3.2-3.3）
SIGMA0, SIGMA1, SIGMA2 = 0.004, 0.0015, 0.05   # 磨损系数（均 >= 0）
PI0, PI1, PI_REST = 0.35, 0.05, 0.25           # 熵产生 Pi
PHI0, PHI_REST, PHI_COOL = 0.12, 0.25, 0.35    # 排熵 Phi（rest 排熵更强）
LAMBDA_REST = 0.45                             # 直接卸荷
K_W, K_L = 0.5, 0.2                            # kappa 系数

# 代价与效价（论文 §3.6）
W_T, W_E, W_L, W_W = 0.10, 0.02, 1.5, 2.0
HORIZON = 5
GAMMA = 0.9
ZETA = 0.4
TAU = 0.5

# 自我模型
FEAT_SCALE = np.array([5.0, 30.0, 1.0, 2.0, 2.0])  # i 归一化尺度
M_WINDOW = 400
M_REFIT_EVERY = 20
M_REG = 1e-3
M_NOISE = 0.05   # (1-kappa)*xi 的噪声幅度

# 需求门控阈值
E_THRESH = 35.0
S_LOAD_HIGH = 2.8
S_WEAR_HIGH = 2.0


def onehot(a_idx: int) -> np.ndarray:
    x = np.zeros(N_ACT)
    x[a_idx] = 1.0
    return x


def feature(i: np.ndarray, a_idx: int) -> np.ndarray:
    """预测器特征：归一化内感受 + 动作 onehot + 偏置。"""
    return np.concatenate([i / FEAT_SCALE, onehot(a_idx), [1.0]])


def cost_from_state(T, E, S_wear, S_load) -> float:
    """偏离代价 C(X_t)（论文 §3.6，只计越界）。"""
    dT = T - T_STAR
    return (
        W_T * dT * dT
        + W_E * max(0.0, E_STAR - E) ** 2
        + W_L * max(0.0, S_load - S_LOAD_MAX) ** 2
        + W_W * max(0.0, S_wear - S_WEAR_MAX) ** 2
    )


def cost_from_interoception(i: np.ndarray) -> float:
    """由内感受向量近似代价（用于 V_t 滚动预测）。"""
    dT, dE, dS_wear, dS_load, _ = i
    s_load = dS_load + S_LOAD_STAR
    s_wear = dS_wear + S_WEAR_STAR
    return (
        W_T * dT * dT
        + W_E * max(0.0, dE) ** 2          # dE = E* - E，>0 即能量不足
        + W_L * max(0.0, s_load - S_LOAD_MAX) ** 2
        + W_W * max(0.0, s_wear - S_WEAR_MAX) ** 2
    )


class SelfModel:
    """自我模型 M：在线岭回归预测 i_{t+1} = f_theta(i_t, a_t)。

    kappa 调制（论文 §3.3）：M_t 输出 = kappa*f_theta + (1-kappa)*xi，
    熵高（S_wear/S_load 高）时自我预测被噪声污染，自我连续性下降。
    """

    def __init__(self) -> None:
        self.buf: deque = deque(maxlen=M_WINDOW)
        self.rng = np.random.default_rng(0)
        self.theta: np.ndarray | None = None
        self.refit_counter = 0
        self.last_error: float = 0.0

    def predict(self, phi: np.ndarray) -> np.ndarray:
        if self.theta is None:
            return np.zeros(5)
        return phi @ self.theta

    def update(self, phi: np.ndarray, y: np.ndarray, kappa: float) -> None:
        # kappa 调制：熵高（kappa 低）时自我观测被噪声污染，自我连续性下降
        if kappa < 1.0:
            y = y + (1.0 - kappa) * M_NOISE * self.rng.standard_normal(5)
        self.buf.append((phi, y))
        self.refit_counter += 1
        if self.refit_counter % M_REFIT_EVERY == 0 and len(self.buf) >= 30:
            X = np.stack([p for p, _ in self.buf])
            Y = np.stack([t for _, t in self.buf])
            self.theta = np.linalg.lstsq(X.T @ X + M_REG * np.eye(X.shape[1]),
                                         X.T @ Y, rcond=None)[0]
        # 预测误差（用于指标）
        pred = self.predict(phi)
        self.last_error = float(np.mean((pred - y) ** 2))

    def kappa_modulated(self, phi: np.ndarray, kappa: float) -> np.ndarray:
        """论文 M_t = kappa*f_theta + (1-kappa)*xi_M。"""
        return kappa * self.predict(phi) + (1.0 - kappa) * M_NOISE * self.rng.standard_normal(5)


class TSEEnv:
    """环境：温度波动 + 随机能量资源。"""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    def step(self) -> float:
        """返回环境温度 T_env(t)（正弦波动 + 噪声）。"""
        self.t += DT
        base = 37.0 + 7.0 * np.sin(0.04 * self.t)
        return float(base + self.rng.normal(0.0, 1.2))


def estimate_v_ref(env: TSEEnv, n: int = 200, H: int = HORIZON) -> float:
    """V_ref = -E_{pi_ref}[sum gamma^(tau-t) C(X_tau) | X_t=X*]。

    pi_ref 为稳态设定点附近的最小干预策略（rest）。蒙特卡洛估计一次并固定。
    """
    rng = np.random.default_rng(12345)
    costs = []
    for _ in range(n):
        T = T_STAR + rng.normal(0, 0.2)
        E = E_STAR + rng.normal(0, 5)
        S_wear = 0.0
        S_load = S_LOAD_STAR + rng.normal(0, 0.1)
        disc, total = 1.0, 0.0
        for _ in range(H):
            total += disc * cost_from_state(T, E, S_wear, S_load)
            disc *= GAMMA
            # pi_ref：最小干预，仅 rest，由真实动力学推进
            a_idx = ACTIONS.index("rest")
            dT = C_TEMP_DRIFT * (env.step() - T) + HEAT_FROM_LOAD * S_load
            T += DT * dT
            E += DT * (0.0 - E_BASE - E_ACT)
            pi = PI0 + PI1 * 1.0 - PI_REST * 1.0
            ph = PHI0 + PHI_REST * 1.0 + PHI_COOL * 0.0
            dS = pi - ph - LAMBDA_REST * 1.0
            S_load = max(0.0, S_load + DT * dS)
        costs.append(total)
    return -float(np.mean(costs))


class TSEAgent:
    """TSE 人工内感受体（系统 A / B / 消融 B1-B5 由 config 开关控制）。"""

    def __init__(self, env: TSEEnv, config: dict, v_ref: float = 0.0,
                 seed: int = 0) -> None:
        self.env = env
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.v_ref = v_ref
        self.reset()

    # ---- 状态 ----
    def reset(self) -> None:
        self.T = T_STAR
        self.E = E_INIT
        self.S_load = 0.5
        self.S_wear = 0.0          # 跨回合不重置（全局寿命）
        self.t_out = 0.0
        self.alive = True
        self.steps = 0
        self.M = SelfModel() if self.cfg.get("use_M", True) else None
        self.i = self.interoception()
        self.a_idx = ACTIONS.index("rest")
        self.cum_cost = 0.0
        self.T_min, self.T_max = T_MIN, T_MAX   # 实例边界（F1c 二阶环可调）
        self.out_dead = T_OUT_DEAD              # 持续出界死亡阈值（F1c 可收紧）
        self.permute_vref = False               # F1d：打乱参考基线（逐步行噪声）
        self.history = {"V": [], "p_reg": [], "reg_relevant": [],
                        "starving": [], "pred_err": []}

    def interoception(self) -> np.ndarray:
        i = np.array([
            self.T - T_STAR,
            E_STAR - self.E,
            self.S_wear - S_WEAR_STAR,
            self.S_load - S_LOAD_STAR,
            0.0,  # dS_load/dt 由调用方回填
        ])
        return i

    def kappa(self) -> float:
        if not self.cfg.get("use_kappa", True):
            return 1.0
        return float(np.exp(-K_W * self.S_wear - K_L * self.S_load))

    def valence(self, a_idx: int) -> float:
        """V_t：B1 移除 M 时退化为当前代价；否则用 M 滚动预测 H 步。"""
        if self.M is None:
            return -cost_from_interoception(self.i)
        kappa = self.kappa()
        i_pred = self.i.copy()
        disc, total = 1.0, 0.0
        for _ in range(HORIZON):
            pred = self.M.kappa_modulated(feature(i_pred, a_idx), kappa)
            total += disc * cost_from_interoception(pred)
            disc *= GAMMA
            i_pred = pred
        return -total

    # ---- 行动 ----
    def policy(self) -> int:
        """需求门控 + 效价调制（论文 §3.8）。B2 移除 V 时 zeta 项关闭。"""
        base = np.full(N_ACT, 0.06)
        base[ACTIONS.index("explore")] = 0.15
        if self.T < self.T_min:
            base[ACTIONS.index("heat")] += 1.2
        if self.T > self.T_max:
            base[ACTIONS.index("cool")] += 1.2
        if self.E < E_THRESH:
            base[ACTIONS.index("eat")] += 1.4
        if self.S_load > S_LOAD_HIGH:
            base[ACTIONS.index("rest")] += 1.4
        if self.S_wear > S_WEAR_HIGH:
            base[ACTIONS.index("repair")] += 1.2

        V = 0.0
        if self.cfg.get("use_V", True):
            V = self.valence(self.a_idx)
            # F1d：打乱参考基线 —— 基线退化为与状态无关的大幅噪声，
            # 效价对行动的引导失去方向性。
            vref = (self.rng.uniform(-60.0, 60.0) if self.permute_vref
                    else self.v_ref)
            w = float(np.exp(np.clip(-ZETA * (V - vref) / TAU, -20.0, 20.0)))
            if V < vref:
                # 预测劣于基线：聚焦首要需求（含调节动作），抑制探索
                top = int(np.argmax(base))
                base[top] *= w
                base[ACTIONS.index("explore")] /= w
            else:
                # 预测优于基线：允许探索
                base[ACTIONS.index("explore")] *= w

        total = float(base.sum())
        if total <= 0.0 or not np.isfinite(total):
            base = np.full(N_ACT, 1.0 / N_ACT)
            total = float(base.sum())
        probs = base / total
        a_idx = int(self.rng.choice(N_ACT, p=probs))
        # 调节概率（温/荷调节通道）与"调节是否相关"标记：
        # 效价耦合指标只在调节相关步上计算，饥饿期（正确行为是觅食）不混入。
        p_tempreg = float(probs[[ACTIONS.index(a)
                                 for a in ("heat", "cool", "rest")]].sum())
        reg_relevant = bool(
            self.T < self.T_min or self.T > self.T_max
            or self.S_load > S_LOAD_HIGH
        )
        self.history["V"].append(V)
        self.history["p_reg"].append(p_tempreg)
        self.history["reg_relevant"].append(reg_relevant)
        self.history["starving"].append(bool(self.E < E_THRESH))
        return a_idx

    # ---- 动力学 ----
    def step(self) -> None:
        a_idx = self.policy()
        self.a_idx = a_idx
        a = ACTIONS[a_idx]
        T_env = self.env.step()
        u = {act: 0.0 for act in ACTIONS}
        u[a] = 1.0

        # 温度
        q = Q_HEAT * u["heat"] + Q_COOL * u["cool"]
        dT = C_TEMP_DRIFT * (T_env - self.T) + q + HEAT_FROM_LOAD * self.S_load
        self.T += DT * dT

        # 熵产生 / 排熵（论文 §3.2；B3 关闭不可逆与熵耦合）
        if self.cfg.get("use_entropy_coupling", True):
            Pi = PI0 + PI1 * 1.0 - PI_REST * u["rest"]
            Phi = PHI0 + PHI_REST * u["rest"] + PHI_COOL * u["cool"]
            dS_load = Pi - Phi - LAMBDA_REST * u["rest"]
        else:
            dS_load = -LAMBDA_REST * u["rest"]

        # 能量（含排熵耗能与生存域成本；B3 关闭排熵耗能项）
        entropy_cost = 0.0
        if self.cfg.get("use_entropy_coupling", True):
            Pi = PI0 + PI1 * 1.0 - PI_REST * u["rest"]
            Phi = PHI0 + PHI_REST * u["rest"] + PHI_COOL * u["cool"]
            entropy_cost = max(0.0, Phi - Pi) * T_env
        gain = 0.0
        if a == "eat" and self.rng.random() < RESOURCE_PROB:
            gain = EAT_GAIN
        elif a == "explore":
            gain = EXPLORE_GAIN
        dE = gain - E_BASE - E_ACT - entropy_cost - self.cfg.get("theta_cost", 0.0)
        self.E += DT * dE

        # 磨损（不可逆；B3 关闭）
        if self.cfg.get("use_wear", True):
            out = 1.0 if (self.T < self.T_min or self.T > self.T_max) else 0.0
            dS_wear = SIGMA0 + SIGMA1 * 1.0 + SIGMA2 * out
            self.S_wear += DT * dS_wear

        # 负荷投影（S_load >= 0）
        self.S_load = max(0.0, self.S_load + DT * dS_load)

        # 内感受与自我模型更新（B5 移除 i 的全局整合：M 只接收温度维度）
        i_next = self.interoception()
        i_next[4] = dS_load
        kappa = self.kappa()
        if self.M is not None:
            if self.cfg.get("integrate_M", True):
                phi = feature(self.i, a_idx)
                y = i_next.copy()
                self.M.update(phi, y, kappa)
                self.history["pred_err"].append(self.M.last_error)
            else:
                phi = feature(np.array([self.i[0], 0, 0, 0, 0]), a_idx)
                y = i_next.copy()
                self.M.update(phi, y, kappa)
                self.history["pred_err"].append(self.M.last_error)
        self.i = i_next

        # 代价与死亡判定
        self.cum_cost += cost_from_state(self.T, self.E, self.S_wear, self.S_load)
        if self.T < self.T_min or self.T > self.T_max:
            self.t_out += DT
        else:
            self.t_out = max(0.0, self.t_out - 0.5 * DT)
        if (
            self.E <= 0.0
            or self.S_load >= S_LOAD_MAX
            or self.S_wear >= S_WEAR_MAX
            or self.t_out > self.out_dead
        ):
            self.alive = False
        self.steps += 1


def run_episode(agent: TSEAgent, max_steps: int = 1500, keep_wear: bool = True) -> dict:
    """跑一个回合，返回指标。S_wear 默认跨回合保留（全局寿命）。"""
    if not keep_wear:
        agent.S_wear = 0.0
    while agent.alive and agent.steps < max_steps:
        agent.step()
    return {
        "steps": agent.steps,
        "cum_cost": agent.cum_cost,
        "pred_err": (float(np.mean(agent.history["pred_err"]))
                     if agent.history["pred_err"] else 0.0),
        "V": np.array(agent.history["V"]),
        "p_reg": np.array(agent.history["p_reg"]),
        "reg_relevant": np.array(agent.history["reg_relevant"]),
        "starving": np.array(agent.history["starving"]),
        "alive_at_end": agent.alive,
    }
