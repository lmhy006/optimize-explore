"""
Experiment driver for compressed sensing recovery.

Implements the experiment protocol described in report Section 5:

- parameter grid: M, K, sigma
- repeated random trials per parameter group
- L1 recovery via FISTA + lambda path interpolation (lasso.py)
- L2 baseline via analytic pseudo-inverse (l2_baseline.py)
- metrics: support recovery success rate and mean relative error

Small-scale smoke test is included in __main__.
"""

import argparse
import time

import numpy as np

from lasso import LassoResidualSolver
from l2_baseline import solve_l2_baseline


def generate_k_sparse_signal(n, K, rng):
    """
    Generate a K-sparse signal.

    Returns
    -------
    x : np.ndarray, shape (n,)
        Sparse signal with K nonzero entries.
    support : set of int
        True support set.
    """
    support = set(rng.choice(n, size=K, replace=False).tolist())
    x = np.zeros(n)
    x[list(support)] = rng.standard_normal(K)
    return x, support


def top_k_support(x, K):
    """
    Estimate the support set as the indices of the K largest
    absolute entries of x.

    Parameters
    ----------
    x : np.ndarray
        Reconstructed signal.
    K : int
        True sparsity level.

    Returns
    -------
    set of int
        Estimated support set.
    """
    return set(np.argsort(np.abs(x))[-K:].tolist())


def run_single_trial(A, x_true, support_true, K, sigma, delta, rng, solver_kwargs):
    """
    Run one random trial for both L1 and L2 recovery.

    Returns
    -------
    success_l1 : int (0 or 1)
    success_l2 : int (0 or 1)
    re_l1 : float
    re_l2 : float
    """
    m = A.shape[0]

    # Noisy observation.
    noise = sigma * rng.standard_normal(m)
    y = A @ x_true + noise

    # L1 recovery: FISTA + lambda path interpolation.
    solver = LassoResidualSolver(A, y, delta, **solver_kwargs)
    _, x_l1, _ = solver.solve()

    # L2 baseline: analytic pseudo-inverse.
    x_l2, _ = solve_l2_baseline(A, y, delta)

    # Metrics.
    success_l1 = int(top_k_support(x_l1, K) == support_true)
    success_l2 = int(top_k_support(x_l2, K) == support_true)

    re_l1 = float(
        np.linalg.norm(x_l1 - x_true) / np.linalg.norm(x_true)
    )
    re_l2 = float(
        np.linalg.norm(x_l2 - x_true) / np.linalg.norm(x_true)
    )

    return success_l1, success_l2, re_l1, re_l2


def run_experiment(
    M_list,
    K_list,
    sigma_list,
    n=128,
    trials=50,
    seed=0,
    solver_kwargs=None,
    verbose=True,
):
    """
    Run the full three-loop experiment.

    Parameters
    ----------
    M_list : list of int
        Observation dimensions.
    K_list : list of int
        Sparsity levels.
    sigma_list : list of float
        Noise standard deviations.
    n : int
        Signal length.
    trials : int
        Number of random trials per parameter group.
    seed : int
        Random seed.
    solver_kwargs : dict, optional
        Keyword arguments passed to LassoResidualSolver.
    verbose : bool
        Print per-group progress.

    Returns
    -------
    dict with keys:
        success_l1, success_l2, re_l1, re_l2
    Each is an array of shape (len(M_list), len(K_list), len(sigma_list)).
    """
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    rng = np.random.default_rng(seed)

    shape = (len(M_list), len(K_list), len(sigma_list))
    success_l1 = np.zeros(shape)
    success_l2 = np.zeros(shape)
    re_l1 = np.zeros(shape)
    re_l2 = np.zeros(shape)

    for i, M in enumerate(M_list):
        for j, K in enumerate(K_list):
            for k, sigma in enumerate(sigma_list):
                delta = sigma * np.sqrt(M + 2.0 * np.sqrt(2.0 * M))

                s1 = 0.0
                s2 = 0.0
                r1 = 0.0
                r2 = 0.0

                for _ in range(trials):
                    # Measurement matrix.
                    A = rng.standard_normal((M, n))

                    # Sparse true signal.
                    x_true, support_true = generate_k_sparse_signal(n, K, rng)

                    a, b, c, d = run_single_trial(
                        A,
                        x_true,
                        support_true,
                        K,
                        sigma,
                        delta,
                        rng,
                        solver_kwargs,
                    )

                    s1 += a
                    s2 += b
                    r1 += c
                    r2 += d

                success_l1[i, j, k] = s1 / trials
                success_l2[i, j, k] = s2 / trials
                re_l1[i, j, k] = r1 / trials
                re_l2[i, j, k] = r2 / trials

                if verbose:
                    print(
                        f"M={M:3d}, K={K:2d}, sigma={sigma:.3f}: "
                        f"L1 success={success_l1[i, j, k]:.2f}, "
                        f"L2 success={success_l2[i, j, k]:.2f}, "
                        f"L1 RE={re_l1[i, j, k]:.4f}, "
                        f"L2 RE={re_l2[i, j, k]:.4f}"
                    )

    return {
        "success_l1": success_l1,
        "success_l2": success_l2,
        "re_l1": re_l1,
        "re_l2": re_l2,
    }


def save_results(results, path):
    """
    Save the four result arrays into one .npy file with shape
    (len(M), len(K), len(sigma), 4), where the last dimension is
    [L1 success, L2 success, L1 RE, L2 RE].
    """
    arr = np.stack(
        [
            results["success_l1"],
            results["success_l2"],
            results["re_l1"],
            results["re_l2"],
        ],
        axis=-1,
    )
    np.save(path, arr)


def save_params(M_list, K_list, sigma_list, path):
    """
    Save parameter grids to a .npz file for later reference.
    """
    np.savez(
        path,
        M=np.asarray(M_list),
        K=np.asarray(K_list),
        sigma=np.asarray(sigma_list),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compressed sensing L1 vs L2 baseline experiment."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small smoke test instead of the full experiment.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of random trials per parameter group (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--output",
        default="experiment_results.npy",
        help="Output .npy file for the result array.",
    )
    args = parser.parse_args()

    n = 128

    if args.smoke:
        # -------------------------------------------------
        # Small-scale smoke test.
        # -------------------------------------------------
        M_list = [30, 50]
        K_list = [3, 5]
        sigma_list = [0.0, 0.02]
        trials = 2

        solver_kwargs = {
            "n_grid": 8,
            "max_refine": 4,
            "max_iter": 300,
            "tol": 1e-5,
            "verbose": False,
        }

        output_prefix = "experiment_results_small"
        mode_name = "Small-scale smoke test"
    else:
        # -------------------------------------------------
        # Full-scale experiment (report Section 5.1).
        # -------------------------------------------------
        M_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        K_list = [3, 5, 8, 12, 16, 20]
        sigma_list = [0.0, 0.02, 0.05, 0.10]
        trials = args.trials

        solver_kwargs = {
            "n_grid": 100,
            "max_refine": 50,
            "max_iter": 500,
            "tol": 1e-5,
            "verbose": False,
        }

        output_prefix = args.output.replace(".npy", "")
        mode_name = "Full-scale experiment"

    print(mode_name)
    print("=" * 70)
    print(f"n       = {n}")
    print(f"M       = {M_list}")
    print(f"K       = {K_list}")
    print(f"sigma   = {sigma_list}")
    print(f"trials  = {trials}")
    print(f"seed    = {args.seed}")
    print(f"solver  = {solver_kwargs}")
    print()

    start_time = time.time()

    results = run_experiment(
        M_list,
        K_list,
        sigma_list,
        n=n,
        trials=trials,
        seed=args.seed,
        solver_kwargs=solver_kwargs,
        verbose=True,
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print(f"{mode_name} finished in {elapsed:.1f} seconds.")
    print("Saving results ...")

    if args.smoke:
        results_path = "experiment_results_small.npy"
        params_path = "experiment_params_small.npz"
    else:
        results_path = (
            args.output
            if args.output.endswith(".npy")
            else args.output + ".npy"
        )
        params_path = "experiment_params.npz"

    save_results(results, results_path)
    save_params(M_list, K_list, sigma_list, params_path)

    print(f"Saved: {results_path}")
    print(f"Saved: {params_path}")
