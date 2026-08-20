# 增广拉格朗日惩罚参数 c 的数值敏感性研究

本项目系统研究增广拉格朗日方法（Augmented Lagrangian Method, ALM）中惩罚参数 \(c\) 对算法行为的影响，围绕以下三个核心研究问题展开：

- **RQ1**：\(c\) 如何影响 ALM 的外层收敛行为？
- **RQ2**：\(c\) 如何影响整体计算效率？
- **RQ3**：最优惩罚参数 \(c^*\) 是否依赖于问题条件数 \(\kappa(H)\) 与内层求解精度 \(\epsilon_{\mathrm{inner}}\)？

项目以对称正定二次规划为基准问题，采用 **KKT reference + ALM + CG** 的实现路线，完成了参数网格批量实验、结果可视化和 LaTeX 实验报告。

---

## 目录结构

```
增广拉格朗日惩罚参数c的数值敏感性研究实验/
├── alm_core.py                 # 核心逻辑：KKT reference、CG 内层求解、ALM 外层迭代
├── experiment.py               # 参数网格批量实验（实验 1/2/3）
├── plot_results.py             # 结果绘图脚本（图 1-7）
├── report.tex                  # 实验报告 LaTeX 源文件
├── report.pdf                  # 编译后的实验报告
├── experiment_output/          # 批量实验输出
│   ├── summary.npz             # 全部 run 的结构化统计
│   ├── summary.csv             # summary.npz 的 CSV 版本
│   ├── c_star.csv              # 各 (κ, ε) 的计算最优 c*
│   └── exp1_trajectories.npz   # 实验 1 的收敛轨迹
├── figures/                    # 生成的图片
│   ├── fig1_convergence.png
│   ├── fig2_ustar.png
│   ├── fig3_eps_compare.png
│   ├── fig4_kappa_compare.png
│   ├── fig5_cond_corr.png
│   ├── fig6_cstar_heatmap.png
│   └── fig7_hc_condition.png
└── README.md
```

---

## 依赖

- Python 3.10+
- NumPy
- Matplotlib
- （可选）TeX Live / XeLaTeX + ctex，用于编译 `report.tex`

安装 Python 依赖：

```bash
pip install numpy matplotlib
```

---

## 快速开始

### 1. 运行批量实验

默认参数与实验方案一致：`trials=20`，`max_outer=500`，`inner_max_iter=1000`。

```bash
python experiment.py
```

快速冒烟测试（减少重复与迭代，用于验证流程）：

```bash
python experiment.py --quick
```

常用参数覆盖：

```bash
python experiment.py --trials 10 --max-outer 300 --inner-max-iter 500
```

输出写入 `experiment_output/`。

### 2. 绘制结果图

```bash
python plot_results.py
```

图片输出到 `figures/`。

### 3. 编译报告

```bash
xelatex report.tex
xelatex report.tex
```

第二次运行是为了解析交叉引用和参考文献超链接。

---

## 核心实现说明

### `alm_core.py`

- `generate_problem()`：生成指定 \(\kappa(H)\) 的对称正定二次规划，并预设 KKT reference \((x^*, \lambda^*)\)，反算 \(b, q\)。
- `cg_solve()`：共轭梯度法求解对称正定线性系统，支持热启动，使用相对初始残差终止。
- `alm_solve()`：ALM 外层迭代，内层调用 CG；记录原始残差、平稳性残差、目标误差、乘子误差、内层迭代步数等。

子问题线性系统：

\[
(H + cA^\top A)x = q - A^\top\lambda + cA^\top b
\]

乘子更新：

\[
\lambda_{k+1} = \lambda_k + c(Ax_{k+1} - b)
\]

### `experiment.py`

- 实验 1：基础 \(c\)-敏感性分析（\(\kappa(H)=100\)，\(\epsilon_{\mathrm{inner}}=10^{-4},10^{-12}\)）
- 实验 2：计算成本权衡与最优 \(c\)（\(\kappa(H)=100\)，\(\epsilon_{\mathrm{inner}}=10^{-4}\)）
- 实验 3：条件数 × 内层精度的耦合影响（\(\kappa(H)\times\epsilon_{\mathrm{inner}}\) 网格）

`c^*` 只从 `converged=True` 的试验中选取，避免未收敛 run 的总内层迭代数误导最优参数判断。

---

## 结果文件说明

- `summary.csv` / `summary.npz`：每个 run 的聚合统计，字段包括：
  `kappa, trial, c, eps, n_outer, total_inner, converged, rp, rs, f_rel, lam_rel, cpu`
- `c_star.csv`：每个 \((\kappa, \epsilon)\) 的计算最优 \(c^*\)，仅基于收敛 trials，并附带收敛次数与收敛率。
- `exp1_trajectories.npz`：实验 1 的逐轮收敛轨迹，按 `c, eps, trial` 组织。

---

## 主要结论

- **H1（支持）**：\(c\) 增大总体提升外层收敛速率，同时增大单轮内层求解成本，二者形成权衡。
- **H2（支持）**：总计算代价随 \(c\) 呈 U 型，存在计算最优 \(c^*\)。
- **H3（部分支持）**：精确 ALM 收敛平稳；不精确 ALM 大 \(c\) 区域易出现平稳性残差平台。
- **H4（支持）**：内层精度越松，最优 \(c^*\) 越小。
- **H5（不支持，需修正）**：\(c^*\) 随 \(\kappa(H)\) 增大并未如预期减小，反而略有增大；真正决定内层难度的是 \(\kappa(H_c)\)，而非 \(\kappa(H)\) 单独决定。

---

## 注意事项

- 完整实验（`trials=20`）需要一定运行时间；建议先用 `--quick` 验证环境。
- Matplotlib 缓存目录在绘图脚本中自动设置为本地 `.mplcache`，避免默认目录权限问题。
- 报告使用 XeLaTeX + ctex 编译，请确保系统安装了中文字体。
