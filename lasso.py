import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

class LassoResidualSolver:
    def __init__(
        self,
        A,
        y,
        delta,
        n_grid=100,
        lambda_ratio=1e-6,#float, lambda_min / lambda_max
        max_iter=10000,#Lasso最大迭代次数
        tol=1e-8,#float, Lasso求解精度
        interpolation_tol=1e-6,#float, 最终要求residual-delta| <= interpolation_tol
        max_refine=50,#int, 最大插值迭代次数
        verbose=True#是否输出求解过程
    ):
        self.A=np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.delta = float(delta)

        self.m, self.n = self.A.shape

        self.n_grid = n_grid
        self.lambda_ratio = lambda_ratio

        self.max_iter = max_iter
        self.tol = tol

        self.interpolation_tol = interpolation_tol
        self.max_refine = max_refine

        self.verbose = verbose

        # sklearn 的 alpha = lambda / n
        self.n_samples = self.m

        # lambda_max
        self.lambda_max = np.linalg.norm(
            self.A.T @ self.y,
            ord=np.inf
        )

        self.lambda_min = (
            self.lambda_ratio * self.lambda_max
        )

        self.lambdas = None
        self.residuals = None
        self.solutions = None

        self.lambda_star = None
        self.x_star = None
        self.residual_star = None

    ##求解一个Lasso
    def solve_lasso(self, lam):
        alpha = lam / self.n_samples

        model = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic"
        )

        model.fit(self.A, self.y)

        x = model.coef_

        residual = np.linalg.norm(
            self.A @ x - self.y
        )

        return x, residual

    ##几何递减lambda序列
    def generate_lambda_grid(self):
        self.lambdas = np.geomspace(
            self.lambda_max,
            self.lambda_min,
            self.n_grid
        )

        return self.lambdas

    ##对所有lambda求Lasso
    def compute_residual_curve(self):
        if self.lambdas is None:
                self.generate_lambda_grid()

        residuals = []
        solutions = []

        if self.verbose:
                print("=" * 60)
                print("Computing Lasso residual curve")
                print("=" * 60)

        for k, lam in enumerate(self.lambdas):

            x, r = self.solve_lasso(lam)

            solutions.append(x)
            residuals.append(r)

            if self.verbose:
                print(
                    f"[{k+1:3d}/{len(self.lambdas)}] "
                    f"lambda = {lam:.6e}, "
                    f"residual = {r:.6e}"
                )

        self.residuals = np.asarray(residuals)
        self.solutions = solutions

        return self.lambdas, self.residuals

    ##找所有delta区间
    def find_bracket(self):
        lambdas = self.lambdas
        residuals = self.residuals

        delta = self.delta

        #检查边界
        if residuals[0] < delta:
            raise ValueError(
                "目标 delta 太大："
                "在 lambda_max 处残差已经小于 delta。"
            )

        if residuals[-1] > delta:
            raise ValueError(
                "目标 delta 太小："
                "在 lambda_min 处残差仍然大于 delta。"
            )

        for i in range(len(lambdas) - 1):

            r1 = residuals[i]
            r2 = residuals[i + 1]

            if (r1 - delta) * (r2 - delta) <= 0:

                return i, i + 1

        raise RuntimeError(
            "没有找到 delta 所在的 lambda 区间。"
        )

    ##插值
    @staticmethod
    def linear_interpolation(
        lambda1,
        r1,
        lambda2,
        r2,
        delta
    ):
        if abs(r1 - r2) < 1e-15:
            #如果两个 residual 几乎一样，直接取 lambda 中点
            return 0.5 * (lambda1 + lambda2)

        lambda_new = (
            lambda2
            + (delta - r2)
            / (r1 - r2)
            * (lambda1 - lambda2)
        )

        return lambda_new

    ##插值迭代
    def refine_lambda(self):

        i_low, i_high = self.find_bracket()

        lam_a = self.lambdas[i_low]
        r_a = self.residuals[i_low]

        lam_b = self.lambdas[i_high]
        r_b = self.residuals[i_high]

        delta = self.delta

        if self.verbose:
            print()
            print("=" * 60)
            print("Refining lambda by interpolation")
            print("=" * 60)

            print(
                f"Initial bracket:\n"
                f"lambda_1 = {lam_a:.6e}, "
                f"r_1 = {r_a:.6e}\n"
                f"lambda_2 = {lam_b:.6e}, "
                f"r_2 = {r_b:.6e}\n"
                f"delta    = {delta:.6e}"
            )

        best_lambda = None
        best_x = None
        best_r = None

        for k in range(self.max_refine):

            lam_new = self.linear_interpolation(
                lam_a,
                r_a,
                lam_b,
                r_b,
                delta
            )

            #防止插值跑出区间
            lam_new = np.clip(
                lam_new,
                min(lam_a, lam_b),
                max(lam_a, lam_b)
            )

            #求新的 Lasso

            x_new, r_new = self.solve_lasso(lam_new)

            error = abs(r_new - delta)

            if self.verbose:
                print(
                    f"Iteration {k+1:3d}: "
                    f"lambda = {lam_new:.10e}, "
                    f"residual = {r_new:.10e}, "
                    f"error = {error:.3e}"
                )

            #保存当前最好结果
            if (
                best_r is None
                or error < abs(best_r - delta)
            ):
                best_lambda = lam_new
                best_x = x_new.copy()
                best_r = r_new

            #判断是否收敛

            if error <= self.interpolation_tol:

                if self.verbose:
                    print()
                    print("Converged!")

                self.lambda_star = lam_new
                self.x_star = x_new
                self.residual_star = r_new

                return (
                    self.lambda_star,
                    self.x_star,
                    self.residual_star
                )

            #更新 bracket

            if r_new >= delta:

                lam_a = lam_new
                r_a = r_new

            else:

                lam_b = lam_new
                r_b = r_new

        #如果 max_refine 次仍没有严格达到 tolerance,返回最好的结果

        if self.verbose:
            print()
            print(
                "Maximum refinement iterations reached."
            )

        self.lambda_star = best_lambda
        self.x_star = best_x
        self.residual_star = best_r

        return (
            self.lambda_star,
            self.x_star,
            self.residual_star
        )

    ##主程序
    def solve(self):
        self.generate_lambda_grid()

        self.compute_residual_curve()

        result = self.refine_lambda()

        return result

    ##残差曲线
    def plot_residual_curve(self):

        if self.lambdas is None:
            raise RuntimeError(
                "请先调用 solve()"
            )

        plt.figure(figsize=(8, 5))

        plt.semilogx(
            self.lambdas,
            self.residuals,
            "o-",
            markersize=4,
            label="Lasso residual"
        )

        plt.axhline(
            self.delta,
            linestyle="--",
            label=r"$\delta$"
        )

        if self.lambda_star is not None:

            plt.scatter(
                [self.lambda_star],
                [self.residual_star],
                s=80,
                zorder=5,
                label=r"$\lambda^\star$"
            )

        plt.xlabel(r"$\lambda$")
        plt.ylabel(
            r"$r(\lambda)=\|Ax_\lambda-y\|_2$"
        )

        plt.title(
            "Lasso Residual Curve"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.show()
