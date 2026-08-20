"""
增广拉格朗日方法（ALM）+ 共轭梯度法（CG）核心逻辑
=================================================

结构：
  1. generate_problem(): 构造带 KKT reference 的 QP 测试问题
  2. cg_solve():          内层 CG 求解对称正定线性系统
  3. alm_solve():         外层 ALM 迭代，返回收敛轨迹与效率统计

说明：
  子问题线性系统由
      min_x  L_c(x, lambda) = 1/2 x^T H x - q^T x
                             + lambda^T(Ax-b) + c/2 ||Ax-b||^2
  的一阶最优性条件得到：
      (H + c A^T A) x = q - A^T lambda + c A^T b
"""

from __future__ import annotations

import numpy as np


def generate_problem(
    n: int = 100,
    m: int = 20,
    kappa: float = 100.0,
    seed: int = 0,
) -> dict:
    """生成一个带 KKT reference 的 QP 测试问题。

    问题：
        min_x  1/2 x^T H x - q^T x
        s.t.   Ax = b

    通过预设 (x*, lambda*) 反算 b, q，使 KKT 条件精确成立：
        H x* - q + A^T lambda* = 0
        A x* = b
    """
    rng = np.random.default_rng(seed)

    # 1. 对称正定 H，条件数为 kappa
    #    特征值从 1 到 kappa 对数均匀分布，最小特征值 1，最大特征值 kappa
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigvals = np.geomspace(1.0, kappa, n)
    H = (Q * eigvals) @ Q.T
    H = (H + H.T) / 2.0  # 数值对称化

    # 2. 随机稠密行满秩 A
    while True:
        A = rng.standard_normal((m, n))
        if np.linalg.matrix_rank(A) == m:
            break

    # 3. 预设 KKT reference
    x_star = rng.standard_normal(n)
    lam_star = rng.standard_normal(m)  # 非零，便于乘子相对误差计算

    # 4. 反算 b 与 q，使 KKT 条件精确成立
    b = A @ x_star
    q = H @ x_star + A.T @ lam_star

    return {
        "H": H,
        "A": A,
        "b": b,
        "q": q,
        "x_star": x_star,
        "lam_star": lam_star,
        "kappa": kappa,
    }


def cg_solve(
    M: np.ndarray,
    rhs: np.ndarray,
    tol: float = 1e-6,
    max_iter: int | None = None,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """用 CG 求解对称正定线性系统 M x = rhs，支持热启动。

    返回：
        x: 近似解
        it: 实际迭代步数
    """
    if max_iter is None:
        max_iter = M.shape[0]

    x = np.zeros_like(rhs) if x0 is None else np.array(x0, dtype=float).copy()
    r = rhs - M @ x
    p = r.copy()
    rs_old = r @ r
    norm_r0 = np.sqrt(rs_old)

    if norm_r0 == 0.0:
        return x, 0

    for it in range(1, max_iter + 1):
        Mp = M @ p
        alpha = rs_old / (p @ Mp)
        x = x + alpha * p
        r = r - alpha * Mp

        rs_new = r @ r
        # 相对初始残差终止准则
        if np.sqrt(rs_new) <= tol * norm_r0:
            return x, it

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, max_iter


def alm_solve(
    H: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    c: float,
    eps_inner: float = 1e-4,
    max_outer: int = 500,
    tol_p: float = 1e-6,
    tol_s: float = 1e-6,
    x0: np.ndarray | None = None,
    lam0: np.ndarray | None = None,
    x_star: np.ndarray | None = None,
    lam_star: np.ndarray | None = None,
    inner_max_iter: int | None = None,
) -> dict:
    """ALM 外层迭代 + CG 内层求解。

    返回字典包含：
        x, lam         : 最终解与乘子
        n_outer        : 实际外层迭代数
        total_inner    : 累计内层 CG 迭代步数
        converged      : 是否在外层终止准则下收敛
        history        : 每轮轨迹列表
    """
    n = H.shape[0]
    m = A.shape[0]

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    lam = np.zeros(m) if lam0 is None else np.array(lam0, dtype=float)

    At = A.T
    AtA = At @ A
    Atb = At @ b

    # 最优目标值，用于相对误差
    if x_star is not None:
        f_star = 0.5 * x_star @ (H @ x_star) - q @ x_star
    else:
        f_star = None

    if lam_star is not None:
        lam_norm = np.linalg.norm(lam_star)
    else:
        lam_norm = None

    history: list[dict] = []
    total_inner = 0
    converged = False

    if inner_max_iter is None:
        # 理论上 CG 至多 n 步收敛；数值上允许更多步以保证子问题精度
        inner_max_iter = max(n, 1000)

    for k in range(1, max_outer + 1):
        # 子问题：min_x L_c(x, lam_k)
        M = H + c * AtA
        rhs = q - At @ lam + c * Atb

        x_new, inner_iters = cg_solve(
            M, rhs, tol=eps_inner, max_iter=inner_max_iter, x0=x
        )
        total_inner += inner_iters

        # 乘子更新
        lam_new = lam + c * (A @ x_new - b)

        # 残差与指标
        rp = np.linalg.norm(A @ x_new - b)
        rs = np.linalg.norm(H @ x_new - q + At @ lam_new)

        record = {
            "outer_iter": k,
            "rp": rp,
            "rs": rs,
            "inner_iters": inner_iters,
        }

        if f_star is not None:
            f_val = 0.5 * x_new @ (H @ x_new) - q @ x_new
            record["f_rel"] = abs(f_val - f_star) / max(abs(f_star), 1e-30)

        if lam_norm is not None:
            record["lam_rel"] = np.linalg.norm(lam_new - lam_star) / max(lam_norm, 1e-30)

        history.append(record)

        x, lam = x_new, lam_new

        if rp <= tol_p and rs <= tol_s:
            converged = True
            break

    return {
        "x": x,
        "lam": lam,
        "n_outer": len(history),
        "total_inner": total_inner,
        "converged": converged,
        "history": history,
    }


if __name__ == "__main__":
    # 快速冒烟测试：固定基准问题，跑一个 c
    prob = generate_problem(n=100, m=20, kappa=100.0, seed=0)

    result = alm_solve(
        prob["H"],
        prob["A"],
        prob["b"],
        prob["q"],
        c=1.0,
        eps_inner=1e-4,
        max_outer=500,
        x_star=prob["x_star"],
        lam_star=prob["lam_star"],
    )

    print(f"converged = {result['converged']}")
    print(f"n_outer   = {result['n_outer']}")
    print(f"total_inner = {result['total_inner']}")
    print(f"final ||rp|| = {result['history'][-1]['rp']:.3e}")
    print(f"final ||rs|| = {result['history'][-1]['rs']:.3e}")

    if "f_rel" in result["history"][-1]:
        print(f"final f_rel = {result['history'][-1]['f_rel']:.3e}")
    if "lam_rel" in result["history"][-1]:
        print(f"final lam_rel = {result['history'][-1]['lam_rel']:.3e}")
