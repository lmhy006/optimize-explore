"""
FISTA solver for the LASSO problem

    min_x  1/2 * ||A x - y||_2^2 + lambda * ||x||_1

This module implements:

- soft_threshold: the proximal operator of lambda * ||x||_1.
- fista_lasso: the Fast Iterative Shrinkage-Thresholding Algorithm.

Reference settings used in the project report (Section 4.3):

- step size: t = 1 / L, where L = ||A||_2^2 = lambda_max(A^T A).
- stopping criteria:
    relative iterate change < tol (default 1e-5)
    AND relative objective change < objective_tol (default 1e-8)
    OR max_iter (default 500) reached.
"""

import numpy as np


def soft_threshold(v, threshold):
    """
    Soft-thresholding operator applied element-wise:

        S_threshold(v) = sign(v) * max(|v| - threshold, 0).

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    threshold : float
        Nonnegative shrinkage threshold.

    Returns
    -------
    np.ndarray
        Soft-thresholded vector.
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def fista_lasso(
    A,
    y,
    lam,
    max_iter=500,
    tol=1e-5,
    objective_tol=1e-8,
    x0=None,
    verbose=False,
):
    """
    Solve

        min_x  1/2 * ||A x - y||_2^2 + lam * ||x||_1

    using FISTA (Nesterov-accelerated proximal gradient descent).

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Measurement matrix.
    y : np.ndarray, shape (m,)
        Observation vector.
    lam : float
        Nonnegative regularization parameter.
    max_iter : int
        Maximum number of FISTA iterations.
    tol : float
        Threshold for the relative iterate change.
    objective_tol : float
        Threshold for the relative objective change.
    x0 : np.ndarray, optional, shape (n,)
        Initial point. Defaults to the zero vector.
    verbose : bool
        If True, print convergence information.

    Returns
    -------
    x : np.ndarray, shape (n,)
        FISTA solution.
    residual : float
        Residual norm ||A x - y||_2.
    objective : float
        Final LASSO objective value.
    n_iter : int
        Number of iterations actually performed.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2-D array.")

    m, n = A.shape

    if y.size != m:
        raise ValueError(
            f"Dimension mismatch: A has {m} rows, "
            f"but y has length {y.size}."
        )

    lam = float(lam)
    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    if objective_tol <= 0:
        raise ValueError("objective_tol must be positive.")

    # Lipschitz constant of grad g(x) = A^T (A x - y):
    #     L = ||A||_2^2 = lambda_max(A^T A).
    L = float(np.linalg.norm(A, 2) ** 2)

    # Degenerate case: A = 0.  Then the problem is just
    # min_x lam * ||x||_1, whose minimizer is x = 0.
    if L <= 0:
        x = np.zeros(n)
        residual = float(np.linalg.norm(y))
        objective = 0.5 * residual * residual + lam * np.sum(np.abs(x))
        return x, residual, objective, 1

    # FISTA step size.
    t = 1.0 / L

    # Initialization: standard FISTA uses t_1 = 1, y_1 = x_0.
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    z = x.copy()          # extrapolation point y_k
    tk = 1.0              # auxiliary sequence t_k

    F_old = None
    n_iter = 0

    for k in range(max_iter):
        # Gradient of the smooth part at the extrapolation point.
        grad = A.T @ (A @ z - y)

        # Proximal gradient step:
        #     x_{k+1} = prox_{t * lam * ||.||_1}(z - t * grad)
        #             = S_{t * lam}(z - t * grad)
        x_new = soft_threshold(z - t * grad, t * lam)

        # Objective value for the stopping test.
        Ax_new = A @ x_new
        F_new = (
            0.5 * np.dot(Ax_new - y, Ax_new - y)
            + lam * np.sum(np.abs(x_new))
        )

        # Check stopping criteria after at least one full iteration.
        if F_old is not None:
            denom_x = max(1.0, float(np.linalg.norm(x)))
            rel_x = float(np.linalg.norm(x_new - x)) / denom_x

            denom_F = max(1.0, abs(F_old))
            rel_F = abs(F_new - F_old) / denom_F

            if verbose:
                print(
                    f"[FISTA] iter {k + 1:4d}: "
                    f"rel_x = {rel_x:.3e}, rel_F = {rel_F:.3e}"
                )

            if rel_x < tol and rel_F < objective_tol:
                x = x_new
                n_iter = k + 1
                break

        F_old = F_new

        # FISTA extrapolation update.
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
        z = x_new + ((tk - 1.0) / tk_new) * (x_new - x)

        x = x_new
        tk = tk_new
        n_iter = k + 1

    residual = float(np.linalg.norm(A @ x - y))
    objective = 0.5 * residual * residual + lam * np.sum(np.abs(x))

    return x, residual, objective, n_iter


# ----------------------------------------------------------------------
# Simple self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    m, n = 40, 80
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    # Sparse true signal.
    x_true = np.zeros(n)
    support = rng.choice(n, size=6, replace=False)
    x_true[support] = rng.standard_normal(6)

    y = A @ x_true + 0.01 * rng.standard_normal(m)

    lam = 0.01 * np.linalg.norm(A.T @ y, ord=np.inf)

    x_hat, residual, objective, n_iter = fista_lasso(
        A, y, lam, verbose=True
    )

    print()
    print(f"lambda      = {lam:.6e}")
    print(f"iterations  = {n_iter}")
    print(f"residual    = {residual:.6e}")
    print(f"objective   = {objective:.6e}")
    print(f"true sparsity      = {np.sum(np.abs(x_true) > 1e-6)}")
    print(f"recovered sparsity = {np.sum(np.abs(x_hat) > 1e-6)}")
    print(f"recovery error     = {np.linalg.norm(x_hat - x_true):.6e}")
