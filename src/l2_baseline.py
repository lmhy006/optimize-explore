"""
L2 baseline solver for compressed sensing.

Contrast model (project report, Section 4.4):

    min_x  ||x||_2
    s.t.   ||A x - y||_2 <= delta

In this project we use the analytic pseudo-inverse closed-form solution

    x_L2 = A^T (A A^T)^{-1} y,

which is the minimum-norm least-squares solution.  For a row-full-rank
under-determined measurement matrix A (the usual case in this project),
this solution satisfies Ax = y, hence its residual is 0 and it is always
feasible for any delta >= 0.

The optional `project=True` path is provided only as an extension point;
the project report states that the analytic solution is used directly in
the experiments.
"""

import numpy as np


def solve_l2_baseline(A, y, delta=None, project=False):
    """
    Compute the L2 baseline solution

        x_L2 = A^T (A A^T)^{-1} y.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Measurement matrix.
    y : np.ndarray, shape (m,)
        Observation vector.
    delta : float, optional
        Residual bound ||Ax - y||_2 <= delta.  Only used for the optional
        feasibility check / projection path; the default experiment uses
        the analytic solution directly.
    project : bool, optional
        If True and the analytic residual exceeds delta, a ValueError is
        raised because the least-squares residual is the smallest possible
        residual, so no feasible point exists.  This option is kept as an
        extension point and is not used in the default experiments.

    Returns
    -------
    x : np.ndarray, shape (n,)
        L2 baseline solution.
    residual : float
        Residual norm ||A x - y||_2.
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

    if delta is not None and delta < 0:
        raise ValueError("delta must be nonnegative.")

    # Analytic pseudo-inverse solution:
    #     x = A^T (A A^T)^{-1} y
    #
    # Use np.linalg.solve instead of computing the inverse explicitly.
    # If A is not row-full-rank, fall back to the Moore-Penrose pseudo-inverse.
    try:
        x = A.T @ np.linalg.solve(A @ A.T, y)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(A) @ y

    residual = float(np.linalg.norm(A @ x - y))

    if project and delta is not None and residual > delta:
        raise ValueError(
            "L2 baseline is infeasible: the least-squares residual "
            f"{residual:.6e} is larger than delta={delta:.6e}. "
            "No x can satisfy ||Ax - y||_2 <= delta."
        )

    return x, residual


# ----------------------------------------------------------------------
# Simple self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    m, n = 20, 50
    A = rng.standard_normal((m, n))
    A = A / np.linalg.norm(A, axis=0, keepdims=True)

    # Sparse true signal (only used to generate a consistent y).
    x_true = np.zeros(n)
    support = rng.choice(n, size=4, replace=False)
    x_true[support] = rng.standard_normal(4)

    # Noiseless observation: y lies in the column space of A^T,
    # so the pseudo-inverse solution has zero residual.
    y = A @ x_true

    x_l2, residual = solve_l2_baseline(A, y)

    # Reference: Moore-Penrose pseudo-inverse.
    x_ref = np.linalg.pinv(A) @ y

    print("L2 baseline self-test")
    print("=" * 60)
    print(f"A shape            = {A.shape}")
    print(f"residual           = {residual:.6e}")
    print(f"||x_l2 - x_ref||_2 = {np.linalg.norm(x_l2 - x_ref):.6e}")
    print(f"||x_l2||_2         = {np.linalg.norm(x_l2):.6e}")
    print(f"||x_true||_2       = {np.linalg.norm(x_true):.6e}")

    # Sanity check: the analytic solution should match pinv up to
    # numerical precision.
    assert np.allclose(x_l2, x_ref, atol=1e-10), (
        "Analytic pseudo-inverse solution does not match pinv."
    )

    print()
    print("Self-test passed.")
