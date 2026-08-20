"""
Generate academic figures from experiment results.

Expected input files (in experiment_output/):
    experiment_params.npz
    experiment_results.npy

Figures are saved to experiment_output/figures/:
    fig1_heatmap_sigma0.png
    fig2_success_vs_M.png
    fig3_re_L1_vs_L2.png
    fig4_sample_recovery.png
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Make project modules importable when running from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lasso import LassoResidualSolver
from l2_baseline import solve_l2_baseline

OUTPUT_DIR = os.path.join(BASE_DIR, "experiment_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUTPUT_DIR, "experiment_results.npy")
PARAMS_PATH = os.path.join(OUTPUT_DIR, "experiment_params.npz")


def load_data():
    """Load parameter grids and result array."""
    params = np.load(PARAMS_PATH)
    M_list = np.asarray(params["M"], dtype=float)
    K_list = np.asarray(params["K"], dtype=int)
    sigma_list = np.asarray(params["sigma"], dtype=float)

    arr = np.load(RESULTS_PATH)  # shape (len(M), len(K), len(sigma), 4)
    return M_list, K_list, sigma_list, arr


def plot_heatmap_sigma0(M_list, K_list, arr):
    """Figure 1: phase-transition heatmap for sigma = 0 (L1 success)."""
    sigma_idx = 0  # sigma = 0
    data = arr[:, :, sigma_idx, 0].T  # shape (len(K), len(M))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    X, Y = np.meshgrid(M_list, K_list)
    mesh = ax.pcolormesh(
        X,
        Y,
        data,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )

    # Annotate each cell with the success percentage.
    for i, M in enumerate(M_list):
        for j, K in enumerate(K_list):
            val = data[j, i] * 100.0
            ax.text(
                M,
                K,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if val < 60 else "black",
            )

    ax.set_xlabel(r"Number of measurements $M$")
    ax.set_ylabel(r"Sparsity $K$")
    ax.set_title(r"L1 Support Recovery Success Rate ($\sigma=0$)")
    ax.set_xticks(M_list)
    ax.set_yticks(K_list)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_heatmap_sigma0.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_success_vs_M(M_list, K_list, sigma_list, arr):
    """Figure 2: L1 success rate vs M, one subplot per K, curves per sigma."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

    colors = plt.cm.plasma(np.linspace(0.0, 0.9, len(sigma_list)))

    for idx, K in enumerate(K_list):
        ax = axes.flat[idx]

        for si, sigma in enumerate(sigma_list):
            ax.plot(
                M_list,
                arr[:, idx, si, 0],
                marker="o",
                markersize=4,
                color=colors[si],
                label=rf"$\sigma={sigma:.2f}$",
            )

        ax.set_title(f"$K={K}$")
        ax.set_xlabel(r"$M$")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle("L1 Support Recovery Success Rate vs M", fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_success_vs_M.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_re_L1_vs_L2(M_list, K_list, sigma_list, arr, K_fixed=8):
    """Figure 3: mean relative error L1 vs L2, one subplot per sigma."""
    k_idx = int(np.where(K_list == K_fixed)[0][0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for si, sigma in enumerate(sigma_list):
        ax = axes.flat[si]

        ax.plot(
            M_list,
            arr[:, k_idx, si, 2],
            "o-",
            color="tab:blue",
            label="L1 (FISTA)",
        )
        ax.plot(
            M_list,
            arr[:, k_idx, si, 3],
            "s--",
            color="tab:red",
            label="L2 (pseudo-inverse)",
        )

        ax.set_title(rf"$\sigma={sigma:.2f}$, $K={K_fixed}$")
        ax.set_xlabel(r"$M$")
        ax.set_ylabel("Mean Relative Error")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle(
        rf"L1 vs L2 Mean Relative Error (fixed $K={K_fixed}$)",
        fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_re_L1_vs_L2.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sample_recovery(M=60, K=8, sigma=0.02, seed=123, n=128):
    """Figure 4: one example of sparse signal recovery."""
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((M, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    support = set(rng.choice(n, size=K, replace=False).tolist())
    x_true = np.zeros(n)
    x_true[list(support)] = rng.standard_normal(K)

    noise = sigma * rng.standard_normal(M)
    y = A @ x_true + noise

    delta = sigma * np.sqrt(M + 2.0 * np.sqrt(2.0 * M))

    # L1 recovery (moderate solver settings for the example figure).
    solver = LassoResidualSolver(
        A=A,
        y=y,
        delta=delta,
        n_grid=50,
        max_refine=20,
        max_iter=500,
        tol=1e-5,
        verbose=False,
    )
    _, x_l1, _ = solver.solve()

    # L2 baseline.
    x_l2, _ = solve_l2_baseline(A, y, delta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # Original sparse signal.
    ax = axes[0, 0]
    ax.stem(x_true, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.set_title(r"Original sparse signal $x_{\mathrm{true}}$")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)

    # Observation vector.
    ax = axes[0, 1]
    ax.plot(y, ".-", color="C1")
    ax.set_title(r"Observation vector $y=Ax_{\mathrm{true}}+e$")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)

    # L1 reconstruction.
    ax = axes[1, 0]
    ax.stem(x_l1, linefmt="C2-", markerfmt="C2o", basefmt="k-")
    ax.set_title(r"L1 recovery $\hat{x}_{L1}$ (FISTA)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)

    # L2 baseline reconstruction.
    ax = axes[1, 1]
    ax.stem(x_l2, linefmt="C3-", markerfmt="C3o", basefmt="k-")
    ax.set_title(r"L2 baseline $\hat{x}_{L2}$ (pseudo-inverse)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)

    fig.suptitle(
        rf"Example Recovery ($M={M}$, $K={K}$, $\sigma={sigma}$, "
        rf"$n={n}$)",
        fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_sample_recovery.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    M_list, K_list, sigma_list, arr = load_data()

    print("Loading data:")
    print(f"  M     = {M_list.tolist()}")
    print(f"  K     = {K_list.tolist()}")
    print(f"  sigma = {sigma_list.tolist()}")
    print(f"  shape = {arr.shape}")
    print()

    plot_heatmap_sigma0(M_list, K_list, arr)
    plot_success_vs_M(M_list, K_list, sigma_list, arr)
    plot_re_L1_vs_L2(M_list, K_list, sigma_list, arr, K_fixed=8)
    plot_sample_recovery()

    print()
    print("All figures saved to:", FIG_DIR)
