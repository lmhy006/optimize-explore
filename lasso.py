import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


class LassoResidualSolver:
    """
    Solve the BPDN problem

        min_x ||x||_1
        s.t.  ||Ax - y||_2 <= delta

    by searching for a lambda such that the solution of

        min_x 1/2 ||Ax-y||_2^2 + lambda ||x||_1

    satisfies

        ||Ax_lambda-y||_2 ~= delta.

    Important:
    sklearn.linear_model.Lasso solves

        1/(2m) ||Ax-y||_2^2 + alpha ||x||_1,

    where m = number of rows of A. Therefore

        alpha = lambda / m.
    """

    def __init__(
        self,
        A,
        y,
        delta,
        n_grid=100,
        lambda_ratio=1e-6,
        max_iter=10000,
        tol=1e-8,
        interpolation_tol=1e-6,
        max_refine=50,
        verbose=True,
    ):
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.delta = float(delta)

        if self.A.ndim != 2:
            raise ValueError("A must be a 2-D array.")

        self.m, self.n = self.A.shape

        if self.y.size != self.m:
            raise ValueError(
                f"Dimension mismatch: A has {self.m} rows, "
                f"but y has length {self.y.size}."
            )

        if self.delta < 0:
            raise ValueError("delta must be nonnegative.")

        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")

        if not (0 < lambda_ratio < 1):
            raise ValueError("lambda_ratio must satisfy 0 < lambda_ratio < 1.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        if interpolation_tol <= 0:
            raise ValueError("interpolation_tol must be positive.")

        if max_refine <= 0:
            raise ValueError("max_refine must be positive.")

        self.n_grid = int(n_grid)
        self.lambda_ratio = float(lambda_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.interpolation_tol = float(interpolation_tol)
        self.max_refine = int(max_refine)
        self.verbose = bool(verbose)

        # sklearn 的 alpha = lambda / m
        self.n_samples = self.m

        # lambda_max = ||A^T y||_infinity.
        # For lambda >= lambda_max, x=0 is a Lasso minimizer.
        self.lambda_max = np.linalg.norm(
            self.A.T @ self.y,
            ord=np.inf
        )

        self.lambda_min = (
            self.lambda_ratio * self.lambda_max
        )

        # Initial residual at x=0.
        self.zero_residual = np.linalg.norm(self.y)

        # Least-squares residual gives the limiting residual as
        # lambda -> 0+ (up to possible numerical issues).
        x_ls, _, _, _ = np.linalg.lstsq(
            self.A, self.y, rcond=None
        )
        self.ls_residual = np.linalg.norm(
            self.A @ x_ls - self.y
        )

        self.lambdas = None
        self.residuals = None
        self.solutions = None

        self.lambda_star = None
        self.x_star = None
        self.residual_star = None

        # True if delta >= ||y||_2, in which case x=0 is
        # already the BPDN optimum.
        self.zero_solution = False

        # All points used during refinement.  They are kept so that
        # refinement does not rely on a strict monotonicity assumption.
        self._sampled_points = []

    # ---------------------------------------------------------
    # Solve one Lasso problem
    # ---------------------------------------------------------

    def solve_lasso(self, lam):
        """
        Solve

            min_x 1/2 ||Ax-y||_2^2 + lambda ||x||_1

        and return x and ||Ax-y||_2.
        """
        lam = float(lam)

        if lam < 0:
            raise ValueError("lambda must be nonnegative.")

        # At lambda >= lambda_max, x=0 is a Lasso minimizer.
        # Handling this explicitly also avoids unnecessary numerical work.
        if self.lambda_max > 0 and lam >= self.lambda_max:
            x = np.zeros(self.n)
            residual = self.zero_residual
            return x, residual

        if lam == 0:
            # Exact lambda=0 is least squares, not Lasso.
            x, _, _, _ = np.linalg.lstsq(
                self.A, self.y, rcond=None
            )
            residual = np.linalg.norm(
                self.A @ x - self.y
            )
            return x, residual

        # sklearn:
        # 1/(2m)||Ax-y||^2 + alpha ||x||_1
        #
        # Desired:
        # 1/2 ||Ax-y||^2 + lambda ||x||_1
        #
        # Hence alpha = lambda / m.
        alpha = lam / self.n_samples

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
        )

        model.fit(self.A, self.y)

        x = model.coef_

        residual = np.linalg.norm(
            self.A @ x - self.y
        )

        return x, residual

    # ---------------------------------------------------------
    # Generate geometric lambda grid
    # ---------------------------------------------------------

    def generate_lambda_grid(self):
        """
        Generate

            lambda_max > ... > lambda_min

        on a geometric scale.
        """
        if self.lambda_max == 0:
            self.lambdas = np.zeros(self.n_grid)
            return self.lambdas

        self.lambdas = np.geomspace(
            self.lambda_max,
            self.lambda_min,
            self.n_grid,
        )

        return self.lambdas

    # ---------------------------------------------------------
    # Compute the initial residual curve
    # ---------------------------------------------------------

    def compute_residual_curve(self):
        if self.lambdas is None:
            self.generate_lambda_grid()

        residuals = []
        solutions = []

        if self.verbose:
            print("=" * 70)
            print("Computing Lasso residual curve")
            print("=" * 70)

        for k, lam in enumerate(self.lambdas):
            x, r = self.solve_lasso(lam)

            solutions.append(x)
            residuals.append(r)

            if self.verbose:
                print(
                    f"[{k + 1:3d}/{len(self.lambdas)}] "
                    f"lambda = {lam:.6e}, "
                    f"residual = {r:.6e}"
                )

        self.residuals = np.asarray(residuals)
        self.solutions = solutions

        self._sampled_points = [
            (float(lam), float(r), x.copy())
            for lam, r, x in zip(
                self.lambdas,
                self.residuals,
                self.solutions
            )
        ]

        return self.lambdas, self.residuals

    # ---------------------------------------------------------
    # Find the best adjacent pair
    # ---------------------------------------------------------

    def find_bracket(self):
        """
        Find the best adjacent pair for interpolation.

        The residual curve is NOT assumed to be strictly monotone.

        Priority:
        1. Among adjacent pairs that straddle delta, choose the pair
           whose two residuals are closest to delta.
        2. If no pair straddles delta, choose the adjacent pair with
           the smallest residual-distance score.

        Returns:
            i, i+1
        """
        if self.lambdas is None or self.residuals is None:
            raise RuntimeError(
                "Residual curve has not been computed."
            )

        lambdas = self.lambdas
        residuals = self.residuals
        delta = self.delta

        n = len(lambdas)

        crossing_candidates = []
        all_candidates = []

        for i in range(n - 1):
            r1 = residuals[i]
            r2 = residuals[i + 1]

            d1 = abs(r1 - delta)
            d2 = abs(r2 - delta)

            # Sum of endpoint distances: smaller means the whole
            # pair is closer to delta.
            score = d1 + d2

            # A pair that contains delta between its residuals.
            crossing = (
                (r1 - delta) * (r2 - delta) <= 0
            )

            all_candidates.append((score, i))

            if crossing:
                crossing_candidates.append((score, i))

        if crossing_candidates:
            _, i = min(
                crossing_candidates,
                key=lambda item: item[0]
            )
        else:
            # No exact crossing was found, possibly because of a
            # coarse grid or numerical non-monotonicity.
            _, i = min(
                all_candidates,
                key=lambda item: item[0]
            )

        return i, i + 1

    # ---------------------------------------------------------
    # Log-linear interpolation
    # ---------------------------------------------------------

    @staticmethod
    def log_linear_interpolation(
        lambda1,
        r1,
        lambda2,
        r2,
        delta,
    ):
        """
        Interpolate linearly in t = log(lambda):

            t_new = t2 + (delta-r2)/(r1-r2) * (t1-t2)

        and return lambda_new = exp(t_new).

        If r1 and r2 are almost equal, return the geometric
        midpoint sqrt(lambda1*lambda2).
        """
        if lambda1 <= 0 or lambda2 <= 0:
            raise ValueError(
                "Log-linear interpolation requires positive lambda."
            )

        if abs(r1 - r2) < 1e-14:
            return np.sqrt(lambda1 * lambda2)

        t1 = np.log(lambda1)
        t2 = np.log(lambda2)

        t_new = (
            t2
            + (delta - r2)
            / (r1 - r2)
            * (t1 - t2)
        )

        return float(np.exp(t_new))

    # ---------------------------------------------------------
    # Store a new sampled point
    # ---------------------------------------------------------

    def _add_sampled_point(self, lam, r, x):
        """
        Add a point and keep all sampled points sorted by decreasing lambda.
        Duplicate lambda values are not added twice.
        """
        lam = float(lam)

        # Relative duplicate check.
        scale = max(1.0, abs(lam))

        for old_lam, _, _ in self._sampled_points:
            if abs(old_lam - lam) <= 1e-14 * scale:
                return

        self._sampled_points.append(
            (lam, float(r), x.copy())
        )

        self._sampled_points.sort(
            key=lambda p: p[0],
            reverse=True
        )

    # ---------------------------------------------------------
    # Rebuild arrays from all sampled points
    # ---------------------------------------------------------

    def _refresh_sampled_arrays(self):
        self.lambdas = np.asarray(
            [p[0] for p in self._sampled_points],
            dtype=float
        )

        self.residuals = np.asarray(
            [p[1] for p in self._sampled_points],
            dtype=float
        )

        self.solutions = [
            p[2].copy()
            for p in self._sampled_points
        ]

    # ---------------------------------------------------------
    # Refinement
    # ---------------------------------------------------------

    def refine_lambda(self):
        """
        Refine lambda without assuming strict monotonicity.

        At every iteration:
        1. Re-select the best adjacent pair from all sampled points.
        2. Perform log-lambda interpolation.
        3. Solve the new Lasso.
        4. Add the new point to the sampled set.
        5. Repeat.

        This avoids the previous update rule
            if r_new >= delta: ...
        which implicitly assumes monotonicity.
        """

        if self._sampled_points is None:
            raise RuntimeError(
                "Residual curve has not been computed."
            )

        # First check whether one of the existing points already
        # satisfies the target.
        best_existing = min(
            self._sampled_points,
            key=lambda p: abs(p[1] - self.delta)
        )

        if abs(best_existing[1] - self.delta) <= self.interpolation_tol:
            self.lambda_star = best_existing[0]
            self.x_star = best_existing[2].copy()
            self.residual_star = best_existing[1]

            if self.verbose:
                print()
                print("=" * 70)
                print("Target already reached on the initial lambda grid.")
                print("=" * 70)

            return (
                self.lambda_star,
                self.x_star,
                self.residual_star,
            )

        if self.verbose:
            print()
            print("=" * 70)
            print("Refining lambda by log-linear interpolation")
            print("=" * 70)

        best_lambda = best_existing[0]
        best_x = best_existing[2].copy()
        best_r = best_existing[1]
        best_error = abs(best_r - self.delta)

        for k in range(self.max_refine):

            self._refresh_sampled_arrays()

            i1, i2 = self.find_bracket()

            lam1 = self.lambdas[i1]
            r1 = self.residuals[i1]

            lam2 = self.lambdas[i2]
            r2 = self.residuals[i2]

            # -------------------------------------------------
            # Interpolate in log(lambda)
            # -------------------------------------------------
            lam_new = self.log_linear_interpolation(
                lam1,
                r1,
                lam2,
                r2,
                self.delta,
            )

            # Keep lambda strictly inside the selected interval.
            lam_low = min(lam1, lam2)
            lam_high = max(lam1, lam2)

            # Geometric midpoint is a safe fallback.
            lam_mid = np.sqrt(lam_low * lam_high)

            if (
                not np.isfinite(lam_new)
                or lam_new <= lam_low
                or lam_new >= lam_high
            ):
                lam_new = lam_mid

            # If interpolation is numerically too close to an
            # existing endpoint, use the geometric midpoint.
            endpoint_ratio = max(
                lam_new / lam_low,
                lam_high / lam_new
            )

            if endpoint_ratio < 1.0 + 1e-12:
                lam_new = lam_mid

            # -------------------------------------------------
            # Solve the new Lasso
            # -------------------------------------------------
            x_new, r_new = self.solve_lasso(lam_new)

            error = abs(r_new - self.delta)

            if self.verbose:
                print(
                    f"Iteration {k + 1:3d}: "
                    f"lambda = {lam_new:.10e}, "
                    f"residual = {r_new:.10e}, "
                    f"error = {error:.3e}"
                )

            # Save the best point encountered so far.
            if error < best_error:
                best_error = error
                best_lambda = lam_new
                best_x = x_new.copy()
                best_r = r_new

            # Add the new point and re-select the best pair next time.
            old_size = len(self._sampled_points)
            self._add_sampled_point(
                lam_new,
                r_new,
                x_new
            )
            new_size = len(self._sampled_points)

            # Convergence check.
            if error <= self.interpolation_tol:
                if self.verbose:
                    print()
                    print("Converged!")

                self.lambda_star = lam_new
                self.x_star = x_new
                self.residual_star = r_new

                self._refresh_sampled_arrays()

                return (
                    self.lambda_star,
                    self.x_star,
                    self.residual_star,
                )

            # If the new lambda was a duplicate and therefore no
            # new information was added, force a local geometric
            # midpoint once more. This prevents an infinite loop.
            if new_size == old_size:
                i1, i2 = self.find_bracket()

                lam1 = self.lambdas[i1]
                lam2 = self.lambdas[i2]

                lam_mid = np.sqrt(
                    min(lam1, lam2) * max(lam1, lam2)
                )

                # If even the midpoint is numerically identical to
                # an endpoint, further refinement is impossible.
                if (
                    np.isclose(
                        lam_mid,
                        lam1,
                        rtol=1e-14,
                        atol=0.0
                    )
                    or np.isclose(
                        lam_mid,
                        lam2,
                        rtol=1e-14,
                        atol=0.0
                    )
                ):
                    break

        # -----------------------------------------------------
        # Maximum refinement reached: return best point found.
        # -----------------------------------------------------
        if self.verbose:
            print()
            print(
                "Maximum refinement iterations reached."
            )
            print(
                f"Best residual error = {best_error:.6e}"
            )

        self.lambda_star = best_lambda
        self.x_star = best_x
        self.residual_star = best_r

        self._refresh_sampled_arrays()

        return (
            self.lambda_star,
            self.x_star,
            self.residual_star,
        )

    # ---------------------------------------------------------
    # Main solve
    # ---------------------------------------------------------

    def solve(self):
        """
        Complete solution procedure.

        Special cases:
        1. delta >= ||y||_2:
               x*=0 is feasible and has the minimum possible l1 norm.
        2. delta < least-squares residual:
               the BPDN constraint is infeasible.
        """

        # -----------------------------------------------------
        # Special case 1:
        # x=0 is feasible, therefore it is automatically optimal.
        # -----------------------------------------------------
        if self.delta >= self.zero_residual:
            self.zero_solution = True
            self.lambda_star = self.lambda_max
            self.x_star = np.zeros(self.n)
            self.residual_star = self.zero_residual

            if self.verbose:
                print("=" * 70)
                print("Special case: zero solution")
                print("=" * 70)
                print(
                    f"delta       = {self.delta:.10e}"
                )
                print(
                    f"||y||_2     = {self.zero_residual:.10e}"
                )
                print(
                    "Since delta >= ||y||_2, x*=0 is feasible "
                    "and is the BPDN optimum."
                )

            return (
                self.lambda_star,
                self.x_star,
                self.residual_star,
            )

        # -----------------------------------------------------
        # Special case 2:
        # The constraint cannot be satisfied even by least squares.
        # -----------------------------------------------------
        if self.delta < self.ls_residual - self.interpolation_tol:
            raise ValueError(
                "BPDN problem is infeasible: "
                f"delta={self.delta:.6e} is smaller than the "
                f"least-squares residual "
                f"{self.ls_residual:.6e}."
            )

        # -----------------------------------------------------
        # lambda_max = 0
        # -----------------------------------------------------
        if self.lambda_max == 0:
            raise ValueError(
                "||A^T y||_inf = 0, so lambda_max = 0. "
                "The Lasso lambda search is degenerate."
            )

        # -----------------------------------------------------
        # Initial geometric grid
        # -----------------------------------------------------
        self.generate_lambda_grid()

        # -----------------------------------------------------
        # Initial residual curve
        # -----------------------------------------------------
        self.compute_residual_curve()

        # -----------------------------------------------------
        # Refine
        # -----------------------------------------------------
        return self.refine_lambda()

    # ---------------------------------------------------------
    # Plot residual curve
    # ---------------------------------------------------------

    def plot_residual_curve(self):
        if self.lambdas is None or self.residuals is None:
            raise RuntimeError(
                "Please call solve() first."
            )

        plt.figure(figsize=(8, 5))

        plt.semilogx(
            self.lambdas,
            self.residuals,
            "o-",
            markersize=4,
            label="Lasso residual",
        )

        plt.axhline(
            self.delta,
            linestyle="--",
            label=r"$\delta$",
        )

        if self.lambda_star is not None:
            plt.scatter(
                [self.lambda_star],
                [self.residual_star],
                s=80,
                zorder=5,
                label=r"$\lambda^\star$",
            )

        plt.xlabel(r"$\lambda$")
        plt.ylabel(
            r"$r(\lambda)=\|Ax_\lambda-y\|_2$"
        )
        plt.title("Lasso Residual Curve")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =============================================================
# Example
# =============================================================

if __name__ == "__main__":

    np.random.seed(42)

    # ---------------------------------------------------------
    # Construct an underdetermined problem
    # ---------------------------------------------------------

    m = 50
    n = 100

    A = np.random.randn(m, n)

    # Normalize columns of A.
    A = A / np.linalg.norm(
        A,
        axis=0,
        keepdims=True
    )

    # ---------------------------------------------------------
    # Sparse true signal
    # ---------------------------------------------------------

    sparsity = 8

    x_true = np.zeros(n)

    support = np.random.choice(
        n,
        sparsity,
        replace=False
    )

    x_true[support] = np.random.randn(sparsity)

    # ---------------------------------------------------------
    # Noisy observation
    # ---------------------------------------------------------

    noise_level = 0.05
    noise = noise_level * np.random.randn(m)

    y = A @ x_true + noise

    # ---------------------------------------------------------
    # Choose delta
    # ---------------------------------------------------------

    delta = 1.05 * np.linalg.norm(noise)

    print("Problem information")
    print("=" * 70)
    print(f"A shape       = {A.shape}")
    print(f"sparsity      = {sparsity}")
    print(
        f"noise norm    = {np.linalg.norm(noise):.6e}"
    )
    print(
        f"least-squares residual = "
        f"{np.linalg.norm(A @ np.linalg.lstsq(A, y, rcond=None)[0] - y):.6e}"
    )
    print(f"delta         = {delta:.6e}")
    print()

    # ---------------------------------------------------------
    # Create solver
    # ---------------------------------------------------------

    solver = LassoResidualSolver(
        A=A,
        y=y,
        delta=delta,

        # Number of initial geometric lambda points.
        n_grid=100,

        # lambda_min / lambda_max.
        lambda_ratio=1e-6,

        # Lasso solver parameters.
        max_iter=20000,
        tol=1e-10,

        # Desired residual accuracy.
        interpolation_tol=1e-6,

        # Maximum refinement iterations.
        max_refine=50,

        verbose=True,
    )

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------

    lambda_star, x_star, residual_star = solver.solve()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("Final result")
    print("=" * 70)

    print(
        f"lambda_star = {lambda_star:.10e}"
    )

    print(
        f"residual    = {residual_star:.10e}"
    )

    print(
        f"target delta = {delta:.10e}"
    )

    print(
        f"residual error = "
        f"{abs(residual_star - delta):.6e}"
    )

    print(
        "recovered sparsity =",
        np.sum(np.abs(x_star) > 1e-6)
    )

    print(
        "true sparsity      =",
        np.sum(np.abs(x_true) > 1e-6)
    )

    print(
        "reconstruction error =",
        np.linalg.norm(x_star - x_true)
    )

    # ---------------------------------------------------------
    # Plot residual curve
    # ---------------------------------------------------------

    solver.plot_residual_curve()
