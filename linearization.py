import numpy as np


def linearize_system(dynamics_func, x_eq, u_eq, eps=1e-6):
    """
    Numerically linearize a continuous-time nonlinear system around an
    operating point.

    The nonlinear plant is expected in the standard form

        x_dot = f(x, u)

    where x is the full state vector and u is the input vector. Around the
    equilibrium/working point (x_eq, u_eq), the first-order approximation is

        delta_x_dot = A * delta_x + B * delta_u

    with

        A = df/dx evaluated at (x_eq, u_eq)
        B = df/du evaluated at (x_eq, u_eq)

    This function estimates those two Jacobians using central finite
    differences. Central differences evaluate both sides of the operating
    point and are usually more accurate than one-sided differences for the
    same perturbation size.

    Parameters
    ----------
    dynamics_func : callable
        Function with signature dynamics_func(x, u) returning x_dot.
    x_eq : array-like, shape (n,)
        State vector of the operating point.
    u_eq : array-like, shape (m,)
        Input vector of the operating point.
    eps : float
        Small perturbation used for the finite-difference derivative.

    Returns
    -------
    A : ndarray, shape (n, n)
        Continuous-time state matrix.
    B : ndarray, shape (n, m)
        Continuous-time input matrix.
    """

    # The number of states and inputs defines the size of the linear model.
    # In this crane project n = 8 and m = 2, but the implementation is kept
    # generic so it can be reused for other systems with the same interface.
    n = len(x_eq)
    m = len(u_eq)

    # Allocate the Jacobian matrices. Each column is filled by perturbing one
    # state or one input while keeping all other variables at the operating
    # point.
    A = np.zeros((n, n), dtype=float)
    B = np.zeros((n, m), dtype=float)

    # Evaluate the nominal derivative once. The central-difference formula
    # below does not need f0 directly, but this checks early that dynamics_func
    # returns a numeric vector of the expected length.
    _f0 = np.asarray(dynamics_func(x_eq, u_eq), dtype=float)

    # Build A = df/dx column by column.
    # For column i, only x[i] is perturbed:
    #
    #     A[:, i] ~= (f(x_eq + eps*e_i, u_eq)
    #                 - f(x_eq - eps*e_i, u_eq)) / (2*eps)
    #
    # The result tells how every state derivative changes when the i-th state
    # is moved slightly away from the operating point.
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps

        f_plus = np.asarray(dynamics_func(x_eq + dx, u_eq), dtype=float)
        f_minus = np.asarray(dynamics_func(x_eq - dx, u_eq), dtype=float)

        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # Build B = df/du column by column.
    # For column j, only u[j] is perturbed. This measures how each state
    # derivative reacts to a small change in the j-th actuator/input.
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps

        f_plus = np.asarray(dynamics_func(x_eq, u_eq + du), dtype=float)
        f_minus = np.asarray(dynamics_func(x_eq, u_eq - du), dtype=float)

        B[:, j] = (f_plus - f_minus) / (2 * eps)

    # A and B describe the local continuous-time linear dynamics. They are used
    # directly by the LQR code and discretized before the MPC code.
    return A, B
