"""
结果绘图脚本
============

读取 experiment_output/ 下的批量实验结果，生成以下图：
  fig1_convergence.png  实验 1：不同 c 下 ALM 原始残差收敛轨迹
  fig2_ustar.png        实验 2：U 型效率曲线与计算最优 c*
  fig3_eps_compare.png  实验 3：不同内层精度下的 U 型曲线对比
  fig4_kappa_compare.png 实验 3：不同问题条件数下的 U 型曲线对比
  fig5_cond_corr.png    内层迭代步数与子问题条件数相关性

运行：
  python plot_results.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# matplotlib 缓存目录设为本地可写目录，避免默认目录权限问题
_BASE = Path(__file__).resolve().parent
_MPL_CACHE = _BASE / ".mplcache"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 路径与数据加载
# ---------------------------------------------------------------------------
OUT_DIR = _BASE / "experiment_output"
FIG_DIR = _BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-12]
KAPPA_LIST = [10.0, 100.0, 1000.0, 10000.0]
C_LIST = [10.0 ** i for i in range(-4, 5)]


def load_data():
    summary = np.load(OUT_DIR / "summary.npz", allow_pickle=True)
    records = summary["records"]
    traj = np.load(OUT_DIR / "exp1_trajectories.npz", allow_pickle=True)
    return records, traj


def avg_total_inner(records, kappa, eps):
    """返回 (c, avg_total_inner, conv_count) 列表，仅统计收敛 trials。"""
    sub = [r for r in records if r["kappa"] == kappa and r["eps"] == eps]
    out = []
    for c in sorted(set(float(r["c"]) for r in sub)):
        conv = [r for r in sub if r["c"] == c and r["converged"]]
        if conv:
            out.append((c, float(np.mean([r["total_inner"] for r in conv])), len(conv)))
        else:
            out.append((c, np.nan, 0))
    return out


def best_c_from_converged(records, kappa, eps):
    """在收敛 trials 中选平均总内层迭代数最小的 c。"""
    data = avg_total_inner(records, kappa, eps)
    valid = [d for d in data if not np.isnan(d[1])]
    if not valid:
        return None
    return min(valid, key=lambda d: d[1])[0]


def make_seed(kappa: float, trial: int) -> int:
    return int(round(np.log10(kappa))) * 10000 + trial


# ---------------------------------------------------------------------------
# 图 1：实验 1 收敛轨迹
# ---------------------------------------------------------------------------
def plot_fig1(traj):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, eps in zip(axes, [1e-4, 1e-12]):
        for c in C_LIST:
            # 取 trial=0 的代表性轨迹
            mask = (
                (traj["c"] == c)
                & (traj["eps"] == eps)
                & (traj["trial"] == 0)
            )
            if not np.any(mask):
                continue
            idx = int(np.flatnonzero(mask)[0])
            y = traj["rp"][idx]
            ax.semilogy(np.arange(1, len(y) + 1), y, lw=1.2,
                        label=f"$c={c:g}$")

        ax.set_xlabel("外层迭代步数 $k$")
        ax.set_ylabel("原始残差 $\\|r_p^k\\|_2$")
        ax.set_title(f"实验 1：$\\kappa(H)=100$，$\\epsilon_{{\\mathrm{{inner}}}}={eps:g}$")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("ALM 原始残差收敛轨迹（不同惩罚参数 $c$）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig1_convergence.png")


# ---------------------------------------------------------------------------
# 图 2：实验 2 U 型效率曲线
# ---------------------------------------------------------------------------
def plot_fig2(records):
    kappa, eps = 100.0, 1e-4
    data = avg_total_inner(records, kappa, eps)
    c_star = best_c_from_converged(records, kappa, eps)

    fig, ax = plt.subplots(figsize=(8, 5))

    cs = [d[0] for d in data]
    ys = [d[1] for d in data]
    ax.semilogx(cs, ys, "o-", color="tab:blue", label="平均总内层迭代数（收敛 trials）")

    if c_star is not None:
        y_star = [d[1] for d in data if d[0] == c_star][0]
        ax.axvline(c_star, color="red", ls="--", alpha=0.7)
        ax.plot(c_star, y_star, "r*", ms=16,
                label=f"计算最优 $c^*={c_star:g}$")

    ax.set_xlabel("惩罚参数 $c$（对数刻度）")
    ax.set_ylabel("平均总内层迭代次数 $N_{\\mathrm{inner,total}}$")
    ax.set_title(f"实验 2：U 型效率曲线（$\\kappa(H)=100$，$\\epsilon_{{\\mathrm{{inner}}}}=10^{{-4}}$）")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_ustar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig2_ustar.png")


# ---------------------------------------------------------------------------
# 图 3：实验 3 不同内层精度下的 U 型曲线对比
# ---------------------------------------------------------------------------
def plot_fig3(records):
    kappa = 100.0
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    for eps, mk in zip(EPS_LIST, markers):
        data = avg_total_inner(records, kappa, eps)
        cs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.semilogx(cs, ys, marker=mk, lw=1.5,
                    label=f"$\\epsilon_{{\\mathrm{{inner}}}}={eps:g}$")

    ax.set_xlabel("惩罚参数 $c$（对数刻度）")
    ax.set_ylabel("平均总内层迭代次数 $N_{\\mathrm{inner,total}}$")
    ax.set_title(f"实验 3：不同内层精度下的 U 型曲线（$\\kappa(H)={kappa:g}$）")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_eps_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig3_eps_compare.png")


# ---------------------------------------------------------------------------
# 图 4：实验 3 不同问题条件数下的 U 型曲线对比
# ---------------------------------------------------------------------------
def plot_fig4(records):
    eps = 1e-4
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    for kappa, mk in zip(KAPPA_LIST, markers):
        data = avg_total_inner(records, kappa, eps)
        cs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.semilogx(cs, ys, marker=mk, lw=1.5,
                    label=f"$\\kappa(H)={kappa:g}$")

    ax.set_xlabel("惩罚参数 $c$（对数刻度）")
    ax.set_ylabel("平均总内层迭代次数 $N_{\\mathrm{inner,total}}$")
    ax.set_title(f"实验 3：不同问题条件数下的 U 型曲线（$\\epsilon_{{\\mathrm{{inner}}}}={eps:g}$）")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_kappa_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig4_kappa_compare.png")


# ---------------------------------------------------------------------------
# 图 5：内层迭代步数与子问题条件数相关性
# ---------------------------------------------------------------------------
def plot_fig5(records):
    import alm_core

    # 重新生成与实验相同 seed 的问题（trial=0），计算每个 c 的子问题条件数
    cond_data = []  # (kappa, c, cond)
    for kappa in KAPPA_LIST:
        prob = alm_core.generate_problem(
            n=100, m=20, kappa=kappa, seed=make_seed(kappa, 0)
        )
        H, A = prob["H"], prob["A"]
        AtA = A.T @ A
        for c in C_LIST:
            Hc = H + c * AtA
            cond_data.append((kappa, c, float(np.linalg.cond(Hc))))

    # 从 records 中取收敛 trials 的平均单轮内层迭代数，按 ε 分组
    data_by_eps = {eps: ([], []) for eps in EPS_LIST}
    color_map = {eps: f"C{i}" for i, eps in enumerate(EPS_LIST)}

    for kappa, c, cond in cond_data:
        for eps in EPS_LIST:
            sub = [
                r for r in records
                if r["kappa"] == kappa and r["c"] == c
                and r["eps"] == eps and r["converged"]
            ]
            if not sub:
                continue
            avg_per_outer = float(np.mean(
                [r["total_inner"] / max(r["n_outer"], 1) for r in sub]
            ))
            data_by_eps[eps][0].append(cond)
            data_by_eps[eps][1].append(avg_per_outer)

    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, color in color_map.items():
        xs, ys = data_by_eps[eps]
        if not xs:
            continue

        ax.loglog(
            xs, ys, "o", color=color, ms=4,
            label=f"$\\epsilon_{{\\mathrm{{inner}}}}={eps:g}$",
        )

        # 按 ε 分组分别拟合，避免不同精度层级混合成一条强行拟合线
        if len(xs) >= 2:
            lx = np.log10(np.array(xs))
            ly = np.log10(np.array(ys))
            coef = np.polyfit(lx, ly, 1)
            x_fit = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
            y_fit = 10 ** (coef[0] * np.log10(x_fit) + coef[1])
            ax.loglog(
                x_fit, y_fit, "--", color=color, lw=1.2,
                label=f"拟合 $\\epsilon={eps:g}$：$y\\propto\\kappa^{{{coef[0]:.2f}}}$",
            )
            print(f"  [fig5] ε={eps:g}: log10 y = {coef[1]:.2f} + {coef[0]:.2f} log10 κ")

    ax.set_xlabel("子问题条件数 $\\kappa(H_c)$")
    ax.set_ylabel("平均单轮内层迭代次数 $N_{\\mathrm{inner}}$")
    ax.set_title("内层迭代步数与子问题条件数的相关性")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_cond_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig5_cond_corr.png")


def plot_fig6_heatmap(records):
    """实验 3 核心图：c* 随 (κ, ε) 漂移的二维热力图。"""
    # 实验 3 的 4×3 网格；行方向从下到上为 ε 从小到大
    kappa_list = [10.0, 100.0, 1000.0, 10000.0]
    eps_list = [1e-6, 1e-4, 1e-2]

    Z = np.full((len(eps_list), len(kappa_list)), np.nan)
    for i, eps in enumerate(eps_list):
        for j, kappa in enumerate(kappa_list):
            c_star = best_c_from_converged(records, kappa, eps)
            if c_star is not None:
                Z[i, j] = np.log10(c_star)

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [
        np.log10(min(kappa_list)), np.log10(max(kappa_list)),
        np.log10(min(eps_list)), np.log10(max(eps_list)),
    ]
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", extent=extent)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10} c^*$")
    cbar.set_ticks([0, 1, 2, 3])

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"])
    ax.set_yticks([-6, -4, -2])
    ax.set_yticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"])
    ax.set_xlabel(r"$\log_{10}\kappa(H)$")
    ax.set_ylabel(r"$\log_{10}\epsilon_{\mathrm{inner}}$")
    ax.set_title(r"$c^*$ 随 $\kappa(H)$ 与 $\epsilon_{\mathrm{inner}}$ 的漂移热力图", pad=12)

    # 拟合 log10 c* = a + b log10 κ + d log10 ε_inner
    valid = [
        (j, i)
        for i in range(len(eps_list))
        for j in range(len(kappa_list))
        if not np.isnan(Z[i, j])
    ]
    if valid:
        X = np.array([
            [1.0, np.log10(kappa_list[j]), np.log10(eps_list[i])]
            for j, i in valid
        ])
        y = np.array([Z[i, j] for j, i in valid])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, d = coef
        y_pred = X @ coef
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        formula = (
            rf"拟合: $\log_{{10}} c^* = {a:.2f} + {b:.2f}\log_{{10}}\kappa "
            rf"+ {d:.2f}\log_{{10}}\epsilon_{{\mathrm{{inner}}}}$"
        )
        print(formula + f", R^2={r2:.3f}")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_cstar_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig6_cstar_heatmap.png")


def plot_fig7_condition_curve(records):
    """补充图：子问题条件数 κ(H_c) 随 c 的变化，并标注各 (κ, ε) 的计算最优 c*。

    用于解释 H5：κ(H) 增大时 c* 为何没有减小。
    """
    import alm_core

    kappa_list = [10.0, 100.0, 1000.0, 10000.0]
    eps_list = [1e-2, 1e-4, 1e-6, 1e-12]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    print("\n[fig7] c* 对应的子问题条件数 κ(H_{c*}):")
    for kappa in kappa_list:
        prob = alm_core.generate_problem(
            n=100, m=20, kappa=kappa, seed=make_seed(kappa, 0)
        )
        H, A = prob["H"], prob["A"]
        AtA = A.T @ A
        conds = []
        for c in C_LIST:
            Hc = H + c * AtA
            conds.append(float(np.linalg.cond(Hc)))

        ax.loglog(
            C_LIST, conds, marker="o", ms=3, lw=1.3,
            label=f"$\\kappa(H)={kappa:g}$",
        )

        for eps in eps_list:
            c_star = best_c_from_converged(records, kappa, eps)
            if c_star is None:
                continue
            idx = C_LIST.index(c_star)
            cond_star = conds[idx]
            ax.plot(c_star, cond_star, "k*", ms=11)
            print(f"  κ(H)={kappa:g}, ε={eps:g}: c*={c_star:g}, "
                  f"κ(H_c*)={cond_star:.3e}")

    ax.set_xlabel("惩罚参数 $c$（对数刻度）")
    ax.set_ylabel(r"子问题条件数 $\kappa(H_c)$（对数刻度）")
    ax.set_title(r"子问题条件数 $\kappa(H_c)$ 随 $c$ 的变化及计算最优 $c^*$ 位置")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_hc_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存 fig7_hc_condition.png")


def main():
    records, traj = load_data()
    print(f"records: {len(records)} runs")
    plot_fig1(traj)
    plot_fig2(records)
    plot_fig3(records)
    plot_fig4(records)
    plot_fig5(records)
    plot_fig6_heatmap(records)
    plot_fig7_condition_curve(records)
    print(f"图片已输出到: {FIG_DIR}")


if __name__ == "__main__":
    main()
