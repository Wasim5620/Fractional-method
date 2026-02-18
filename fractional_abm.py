import numpy as np
from math import gamma


def caputo_abm_solver(rhs, t_grid, y0, alpha, args=()):
    t_grid = np.asarray(t_grid, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    alpha = float(alpha)

    if not (0 < alpha <= 1):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    n_steps = len(t_grid)
    dim = len(y0)
    solution = np.zeros((n_steps, dim), dtype=float)
    solution[0] = y0

    f_cache = np.zeros((n_steps, dim), dtype=float)
    f_cache[0] = rhs(t_grid[0], solution[0], *args)

    g1 = gamma(alpha + 1.0)
    g2 = gamma(alpha + 2.0)

    for n in range(n_steps - 1):
        h = t_grid[n + 1] - t_grid[n]

        predictor_sum = np.zeros(dim, dtype=float)
        for j in range(n + 1):
            weight = (n + 1 - j) ** alpha - (n - j) ** alpha if n - j >= 0 else 1.0
            predictor_sum += weight * f_cache[j]

        y_pred = y0 + (h ** alpha / g1) * predictor_sum
        f_pred = rhs(t_grid[n + 1], y_pred, *args)

        corrector_sum = np.zeros(dim, dtype=float)
        for j in range(1, n + 1):
            corrector_weight = (
                (n - j + 2) ** (alpha + 1)
                - 2 * (n - j + 1) ** (alpha + 1)
                + (n - j) ** (alpha + 1)
            )
            corrector_sum += corrector_weight * f_cache[j]

        y_next = y0 + (h ** alpha / g2) * (
            f_pred + ((n + 1) ** (alpha + 1) - (n + 1 - alpha) * (n + 1) ** alpha) * f_cache[0] + corrector_sum
        )

        solution[n + 1] = y_next
        f_cache[n + 1] = rhs(t_grid[n + 1], y_next, *args)

    return solution
