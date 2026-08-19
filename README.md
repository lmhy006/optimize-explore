# 基于 L1 凸松弛的压缩感知信号恢复

本项目实现了一套自研凸优化求解器的稀疏信号恢复方案：自主实现 **FISTA 加速近端梯度算法**，结合 **λ 几何路径 + 残差插值策略** 求解带噪声约束的 L1 恢复问题，并与 **L2 最小范数基线** 进行对照实验。

## 主要特性

- 自研 FISTA 求解 LASSO，不依赖 `sklearn`、`cvxpy` 等外部凸优化库
- 针对约束形式 L1 恢复，设计 λ 几何递减序列 + 残差曲线插值，稳健匹配残差约束
- L2 基线采用解析伪逆闭式解，对比公平且计算高效
- 完整三维参数仿真：观测维度 `M`、稀疏度 `K`、噪声强度 `σ`
- 自动生成学术图表：相变热力图、成功率曲线、误差对比图、样例恢复图
- 完整 LaTeX 实验报告源码

## 项目结构

```text
.
├── FISTA.py               # FISTA 求解 LASSO
├── lasso.py               # λ 路径搜索，求解约束 L1 问题
├── l2_baseline.py         # L2 基线解析伪逆求解
├── experiment.py          # 批量仿真入口（支持小规模冒烟测试）
├── plot_results.py        # 根据仿真结果生成四类图表
├── report.tex             # 实验报告 LaTeX 源码
├── experiment_output/     # 仿真结果数据
│   ├── experiment_results.npy
│   └── experiment_params.npz
├── figures/               # 生成的四类图表
└── README.md
```

## 环境要求

- Python 3.9+
- NumPy
- Matplotlib（仅绘图需要）
- LaTeX（可选，用于本地编译报告：XeLaTeX + ctex）

安装依赖：

```bash
pip install numpy matplotlib
```

## 快速开始

### 1. 小规模冒烟测试

```bash
python experiment.py --smoke
```

验证求解器收敛、指标计算和保存流程是否正常。

### 2. 正式仿真

```bash
python experiment.py
```

默认运行完整参数网格：

- 信号长度 `n = 128`
- 观测维度 `M ∈ {20,30,40,50,60,70,80,90,100}`
- 稀疏度 `K ∈ {3,5,8,12,16,20}`
- 噪声强度 `σ ∈ {0,0.02,0.05,0.10}`
- 每组随机试验 50 次

### 3. 自定义试验次数

```bash
# 先跑 5 次试水，估算全量耗时
python experiment.py --trials 5

# 指定随机种子
python experiment.py --trials 50 --seed 42
```

### 4. 生成图表

```bash
python plot_results.py
```

图表输出到 `figures/`：

- `fig1_heatmap_sigma0.png`：无噪声相变热力图
- `fig2_success_vs_M.png`：成功率随 M 变化曲线
- `fig3_re_L1_vs_L2.png`：L1 与 L2 误差对比
- `fig4_sample_recovery.png`：单次样例恢复对照

## 结果说明

仿真结果保存在 `experiment_output/`：

- `experiment_results.npy`：形状 `(9, 6, 4, 4)`
  - 前三维对应 `M, K, σ`
  - 最后一维为 `[L1 成功率, L2 成功率, L1 平均相对误差, L2 平均相对误差]`
- `experiment_params.npz`：保存 `M, K, σ` 参数网格

## 实验报告

`report.tex` 为完整实验报告源码，包含：

1. 摘要
2. 引言
3. 问题建模与凸优化理论基础
4. 自研求解器：加速近端梯度算法（FISTA）
5. 仿真实验方案设计
6. 实验结果可视化
7. 结果分析与讨论
8. 结论
9. 参考文献

本地编译：

```bash
latexmk -xelatex report.tex
```

> 注意：`.gitignore` 忽略了 `*.pdf` 和 `*.docx`，`report.pdf` 不会提交到仓库，需要本地自行编译。

## 注意事项

- **无噪声情形（σ=0）**：此时 `δ=0`，代码已做特殊处理，通过下压 `λ_min` 和 FISTA warm start 近似等式约束最小 L1 解。
- **L2 基线说明**：`l2_baseline.py` 使用的是等式约束最小范数解 `A^T(AA^T)^{-1}y`，它是约束问题的可行解，但不是严格约束最优解；作为不利用稀疏性的朴素对照足够公平，详见报告 4.4。
- **运行时间**：全量仿真为 `9 × 6 × 4 × 50 = 10800` 次 L1 求解，耗时可能较长，建议先运行 `--smoke` 或 `--trials 5` 估算时间。

## 主要结论

- L1 恢复存在明显的相变现象：观测维度存在临界值，超过后支撑集恢复成功率快速趋近 100%
- 稀疏度 `K` 越大，所需观测维度越高；噪声强度 `σ` 越大，相变临界右移、重建误差上升
- L1 凸松弛在支撑集恢复成功率和相对重建误差上均显著优于 L2 最小范数基线
