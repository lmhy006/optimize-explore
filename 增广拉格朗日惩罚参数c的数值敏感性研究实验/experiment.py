"""
参数网格批量实验脚本
====================

覆盖实验方案中的：
  - 实验 1：基础 c-敏感性分析（κ=100，ε=1e-4 / 1e-12）
  - 实验 2：计算成本权衡与最优 c（κ=100，ε=1e-4）
  - 实验 3：条件数 × 内层精度的耦合影响（κ × ε 全网格）

输出：
  experiment_output/
    summary.npz          所有 run 的聚合统计
    summary.csv          同上，CSV 格式便于查看
    c_star.csv           每个 (κ, ε) 的计算最优 c*
    exp1_trajectories.npz 实验 1 的收敛轨迹（可选）

用法：
  python experiment.py             # 按实验方案完整运行（较慢）
  python experiment.py --quick     # 快速冒烟测试
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import alm_core

# ---------------------------------------------------------------------------
# 实验参数（与实验方案一致）
# ---------------------------------------------------------------------------
N = 100
M = 20

C_LIST = [10.0 ** i for i in range(-4, 5)]          # 9 个 c，覆盖 9 个数量级
EPS_LIST = [1e-2, 1e-4, 1e-6, 1e-12]                # 1e-12 作为精确 ALM 近似
KAPPA_LIST = [10.0, 100.0, 1000.0, 10000.0]

TOL_P = 1e-6
TOL_S = 1e-6

DEFAULT_TRIALS = 20
DEFAULT_MAX_OUTER = 500
DEFAULT_INNER_MAX_ITER = 1000

# 实验 1 需要保存轨迹的 ε 集合
EXP1_EPS = [1e-4, 1e-12]

# ---------------------------------------------------------------------------
# 结果记录结构
# ---------------------------------------------------------------------------
RECORD_DTYPE = np.dtype([
    ("kappa", "f8"),
    ("trial", "i4"),
    ("c", "f8"),
    ("eps", "f8"),
    ("n_outer", "i4"),
    ("total_inner", "i4"),
    ("converged", "bool"),
    ("rp", "f8"),
    ("rs", "f8"),
    ("f_rel", "f8"),
    ("lam_rel", "f8"),
    ("cpu", "f8"),
])


def make_seed(kappa: float, trial: int) -> int:
    """为 (κ, trial) 生成稳定随机种子。"""
    return int(round(np.log10(kappa))) * 10000 + trial


def run_single(
    prob: dict,
    c: float,
    eps_inner: float,
    max_outer: int,
    inner_max_iter: int,
) -> tuple[dict, dict]:
    """运行一次 ALM，返回 (聚合统计, alm_solve 原始结果)。"""
    t0 = time.time()
    res = alm_core.alm_solve(
        prob["H"],
        prob["A"],
        prob["b"],
        prob["q"],
        c=c,
        eps_inner=eps_inner,
        max_outer=max_outer,
        tol_p=TOL_P,
        tol_s=TOL_S,
        x_star=prob["x_star"],
        lam_star=prob["lam_star"],
        inner_max_iter=inner_max_iter,
    )
    cpu = time.time() - t0

    last = res["history"][-1]
    summary = {
        "n_outer": res["n_outer"],
        "total_inner": res["total_inner"],
        "converged": res["converged"],
        "rp": last["rp"],
        "rs": last["rs"],
        "f_rel": last.get("f_rel", np.nan),
        "lam_rel": last.get("lam_rel", np.nan),
        "cpu": cpu,
    }
    return summary, res


def run_all_combinations(
    c_list: list[float],
    eps_list: list[float],
    kappa_list: list[float],
    trials: int,
    max_outer: int,
    inner_max_iter: int,
    save_exp1_traj: bool = True,
) -> tuple[list[dict], dict | None]:
    """运行 (κ, ε, c, trial) 网格。

    同一 (κ, trial) 复用同一个随机问题，保证不同 ε 和 c 的对比公平。
    返回：
      records: 聚合统计列表
      exp1_traj: 实验 1 轨迹数据（若 save_exp1_traj 为 True）
    """
    records: list[dict] = []

    exp1_traj = None
    if save_exp1_traj:
        exp1_traj = {
            "c": [],
            "eps": [],
            "trial": [],
            "rp": [],
            "rs": [],
            "f_rel": [],
            "lam_rel": [],
        }

    def active_eps(kappa: float) -> list[float]:
        # κ=100 覆盖实验 1/2/3 需要的全部精度；其他 κ 只需实验 3 的前三个精度
        return eps_list if kappa == 100.0 else eps_list[:3]

    total_runs = (
        sum(len(active_eps(kappa)) for kappa in kappa_list)
        * len(c_list)
        * trials
    )
    run_count = 0

    for kappa in kappa_list:
        for trial in range(trials):
            prob = alm_core.generate_problem(
                n=N, m=M, kappa=kappa, seed=make_seed(kappa, trial)
            )

            for eps in active_eps(kappa):
                for c in c_list:
                    summary, res = run_single(
                        prob, c, eps, max_outer, inner_max_iter
                    )

                    record = {
                        "kappa": kappa,
                        "trial": trial,
                        "c": c,
                        "eps": eps,
                        **summary,
                    }
                    records.append(record)

                    # 实验 1 轨迹：κ=100 且 ε 属于实验 1 集合
                    if save_exp1_traj and kappa == 100.0 and eps in EXP1_EPS:
                        hist = res["history"]
                        exp1_traj["c"].append(c)
                        exp1_traj["eps"].append(eps)
                        exp1_traj["trial"].append(trial)
                        exp1_traj["rp"].append(np.array([h["rp"] for h in hist]))
                        exp1_traj["rs"].append(np.array([h["rs"] for h in hist]))
                        exp1_traj["f_rel"].append(
                            np.array([h.get("f_rel", np.nan) for h in hist])
                        )
                        exp1_traj["lam_rel"].append(
                            np.array([h.get("lam_rel", np.nan) for h in hist])
                        )

                    run_count += 1
                    if run_count % 20 == 0 or run_count == total_runs:
                        print(f"  [{run_count}/{total_runs}] κ={kappa:g}, "
                              f"trial={trial}, ε={eps:g}, c={c:g}")

    return records, exp1_traj


def aggregate_c_star(records: list[dict]) -> list[dict]:
    """对每个 (κ, ε) 计算计算最优 c*。

    只使用达到外层终止条件的收敛 trials 计算平均总内层迭代数；
    若某个 (κ, ε) 下没有任何收敛 trial，则 c* 记为 None（无收敛最优）。
    """
    groups: dict[tuple[float, float], dict[float, list[dict]]] = {}

    for rec in records:
        key = (rec["kappa"], rec["eps"])
        groups.setdefault(key, {}).setdefault(rec["c"], []).append(rec)

    rows = []
    for (kappa, eps), c_groups in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        candidates = []
        for c, recs in sorted(c_groups.items()):
            conv_recs = [r for r in recs if r["converged"]]
            conv_rate = len(conv_recs) / len(recs)
            if conv_recs:
                avg_inner = float(np.mean([r["total_inner"] for r in conv_recs]))
                avg_cpu = float(np.mean([r["cpu"] for r in conv_recs]))
                candidates.append({
                    "c": c,
                    "avg_total_inner": avg_inner,
                    "avg_cpu": avg_cpu,
                    "converged_count": len(conv_recs),
                    "conv_rate": conv_rate,
                })

        if candidates:
            # 在收敛 trials 中选平均总内层迭代数最小的 c；
            # 若并列，优先选择收敛率更高的 c。
            best = min(candidates, key=lambda x: (x["avg_total_inner"], -x["conv_rate"]))
            rows.append({
                "kappa": kappa,
                "eps": eps,
                "c_star": best["c"],
                "avg_total_inner": best["avg_total_inner"],
                "avg_cpu": best["avg_cpu"],
                "converged_count": best["converged_count"],
                "conv_rate": best["conv_rate"],
            })
        else:
            rows.append({
                "kappa": kappa,
                "eps": eps,
                "c_star": None,
                "avg_total_inner": np.nan,
                "avg_cpu": np.nan,
                "converged_count": 0,
                "conv_rate": 0.0,
            })
    return rows


def save_records_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "kappa", "trial", "c", "eps", "n_outer", "total_inner",
        "converged", "rp", "rs", "f_rel", "lam_rel", "cpu",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def save_c_star_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "kappa", "eps", "c_star", "avg_total_inner",
        "avg_cpu", "converged_count", "conv_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out["c_star"] is None:
                out["c_star"] = "none"
            writer.writerow(out)


def save_exp1_traj_npz(exp1_traj: dict, path: Path, max_len: int) -> None:
    """将实验 1 轨迹保存为定长 NaN 填充数组。"""
    n_runs = len(exp1_traj["c"])
    shape = (n_runs, max_len)

    rp = np.full(shape, np.nan)
    rs = np.full(shape, np.nan)
    f_rel = np.full(shape, np.nan)
    lam_rel = np.full(shape, np.nan)

    for i, arr in enumerate(exp1_traj["rp"]):
        rp[i, : len(arr)] = arr
    for i, arr in enumerate(exp1_traj["rs"]):
        rs[i, : len(arr)] = arr
    for i, arr in enumerate(exp1_traj["f_rel"]):
        f_rel[i, : len(arr)] = arr
    for i, arr in enumerate(exp1_traj["lam_rel"]):
        lam_rel[i, : len(arr)] = arr

    np.savez(
        path,
        c=np.array(exp1_traj["c"]),
        eps=np.array(exp1_traj["eps"]),
        trial=np.array(exp1_traj["trial"]),
        rp=rp,
        rs=rs,
        f_rel=f_rel,
        lam_rel=lam_rel,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALM 惩罚参数 c 数值敏感性批量实验")
    parser.add_argument("--quick", action="store_true", help="快速冒烟测试（减少重复与迭代）")
    parser.add_argument("--trials", type=int, default=None, help="每组随机实验次数")
    parser.add_argument("--max-outer", type=int, default=None, help="最大外层迭代数")
    parser.add_argument("--inner-max-iter", type=int, default=None, help="CG 最大内层迭代数")
    parser.add_argument("--no-trajectories", action="store_true", help="不保存实验 1 轨迹")
    parser.add_argument("--outdir", type=str, default="experiment_output", help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        trials = args.trials or 1
        max_outer = args.max_outer or 50
        inner_max_iter = args.inner_max_iter or 200
    else:
        trials = args.trials or DEFAULT_TRIALS
        max_outer = args.max_outer or DEFAULT_MAX_OUTER
        inner_max_iter = args.inner_max_iter or DEFAULT_INNER_MAX_ITER

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("参数设置：")
    print(f"  trials         = {trials}")
    print(f"  max_outer      = {max_outer}")
    print(f"  inner_max_iter = {inner_max_iter}")
    print(f"  c_grid         = {C_LIST}")
    print(f"  eps_grid       = {EPS_LIST}")
    print(f"  kappa_grid     = {KAPPA_LIST}")

    # 运行完整网格：
    #  - κ=100 跑全部 4 个 ε（实验 1/2/3 都需要）
    #  - 其他 κ 跑前 3 个 ε（实验 3 需要）
    print("\n开始批量实验...")
    t_start = time.time()

    records, exp1_traj = run_all_combinations(
        c_list=C_LIST,
        eps_list=EPS_LIST,
        kappa_list=KAPPA_LIST,
        trials=trials,
        max_outer=max_outer,
        inner_max_iter=inner_max_iter,
        save_exp1_traj=not args.no_trajectories,
    )
    print(f"批量实验完成，用时 {time.time() - t_start:.1f}s")

    # 保存聚合统计
    records_arr = np.array(
        [
            (
                r["kappa"], r["trial"], r["c"], r["eps"], r["n_outer"],
                r["total_inner"], r["converged"], r["rp"], r["rs"],
                r["f_rel"], r["lam_rel"], r["cpu"],
            )
            for r in records
        ],
        dtype=RECORD_DTYPE,
    )
    np.savez(
        out_dir / "summary.npz",
        records=records_arr,
        c_grid=np.array(C_LIST),
        eps_grid=np.array(EPS_LIST),
        kappa_grid=np.array(KAPPA_LIST),
    )
    save_records_csv(records, out_dir / "summary.csv")
    print(f"已保存: {out_dir / 'summary.npz'}, {out_dir / 'summary.csv'}")

    # 保存 c* 表
    c_star_rows = aggregate_c_star(records)
    save_c_star_csv(c_star_rows, out_dir / "c_star.csv")
    print(f"已保存: {out_dir / 'c_star.csv'}")
    print("\nc* 汇总（仅基于收敛 trials）：")
    for row in c_star_rows:
        c_star = row["c_star"]
        if c_star is None:
            print(f"  κ={row['kappa']:>8g}, ε={row['eps']:>6g} -> c*=无收敛")
        else:
            print(f"  κ={row['kappa']:>8g}, ε={row['eps']:>6g} -> c*={c_star:g} "
                  f"(conv={row['converged_count']}/{row['conv_rate']:.2%})")

    # 保存实验 1 轨迹
    if exp1_traj is not None:
        save_exp1_traj_npz(exp1_traj, out_dir / "exp1_trajectories.npz", max_len=max_outer)
        print(f"已保存: {out_dir / 'exp1_trajectories.npz'}")


if __name__ == "__main__":
    main()
