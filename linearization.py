import numpy as np


def linearize_system(dynamics_func, x_eq, u_eq, eps=1e-6):
    """
    Linearizace nelineárního systému:
        x_dot = f(x, u)

    Vrací matice A, B kolem pracovního bodu (x_eq, u_eq).
    """

    n = len(x_eq)
    m = len(u_eq)

    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    f0 = np.asarray(dynamics_func(x_eq, u_eq), dtype=float)

    # Jacobian podle stavu
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps

        f_plus = np.asarray(dynamics_func(x_eq + dx, u_eq), dtype=float)
        f_minus = np.asarray(dynamics_func(x_eq - dx, u_eq), dtype=float)

        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # Jacobian podle vstupu
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps

        f_plus = np.asarray(dynamics_func(x_eq, u_eq + du), dtype=float)
        f_minus = np.asarray(dynamics_func(x_eq, u_eq - du), dtype=float)

        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B