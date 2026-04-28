from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import numpy as np

from constants import params, x0
from symbolic_model import build_crane_model
from mat_model import crane_dynamics
from linearization import linearize_system
from visualization import (
    payload_polar_position,
    plot_angles,
    plot_positions,
    plot_velocities,
    plot_control_inputs,
    plot_payload_trajectory,
    plot_payload_position_error,
)
from meshcat_crane_viz import visualize_crane_meshcat

np.set_printoptions(precision=5, suppress=True)

# ============================================================
# 1) Symbolicky model -> numericke funkce
# ============================================================
M_func, forcing_func = build_crane_model()


# ============================================================
# 2) Nelinearni plant
# ============================================================
def plant_dynamics(x, u, d=None):
    if d is None:
        d = np.zeros(3, dtype=float)

    return crane_dynamics(
        0.0, x, u, d, params, M_func, forcing_func
    )


# ============================================================
# 3) Pracovni bod
# ============================================================
r0 = 3.0
phi0 = 0.0

x_eq = np.array([
    r0,    # r
    phi0,  # phi
    0.0,   # alpha_t
    0.0,   # alpha_n
    0.0,   # dr
    0.0,   # dphi
    0.0,   # dalpha_t
    0.0    # dalpha_n
], dtype=float)

u_eq = np.array([
    0.0,   # F_r
    0.0    # M_phi
], dtype=float)


# ============================================================
# 4) Linearizace
# ============================================================
A, B = linearize_system(plant_dynamics, x_eq, u_eq)

print("Matice A:")
print(A)
print("\nMatice B:")
print(B)


# ============================================================
# 5) Reference a pomocne funkce pro polohu bremene
# ============================================================
x_ref = np.array([
    4.0,   # r_target
    0.4,   # phi_target
    0.0,   # alpha_t
    0.0,   # alpha_n
    0.0,   # dr
    0.0,   # dphi
    0.0,   # dalpha_t
    0.0    # dalpha_n
], dtype=float)


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def linearize_payload_position(x_ref, params, eps=1e-6):
    C = np.zeros((2, len(x_ref)), dtype=float)

    for i in range(len(x_ref)):
        dx = np.zeros(len(x_ref), dtype=float)
        dx[i] = eps

        y_plus = payload_polar_position(x_ref + dx, params)
        y_minus = payload_polar_position(x_ref - dx, params)

        dy = y_plus - y_minus
        dy[1] = wrap_angle(dy[1])
        C[:, i] = dy / (2.0 * eps)

    return C


# ============================================================
# 6) Vahy LQR
# ============================================================
Q_state = np.diag([5.0, 5.0, 80.0, 80.0, 5.0, 2.0, 30.0, 30.0])
Q_payload = np.diag([1500.0, 2500.0])

C_payload = linearize_payload_position(x_ref, params)

Q = Q_state + C_payload.T @ Q_payload @ C_payload
R = np.diag([1.6e-5, 2.0e-6])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("\nMatice K (LQR zisk):")
print(K)


# ============================================================
# 7) Vstupni funkce LQR
# ============================================================
def control_input(x, K, params):
    dx = x - x_eq
    dx[1] = wrap_angle(dx[1])

    u = u_eq - K @ dx
    u[0] = np.clip(u[0], -params["F_r_max"], params["F_r_max"])
    u[1] = np.clip(u[1], -params["M_phi_max"], params["M_phi_max"])
    return u


# ============================================================
# 8) Porucha
# ============================================================
rng = np.random.default_rng(seed=42)


def generate_wind_samples(t_grid):
    wr = rng.normal(0.0, params["wind_std_wr"], size=len(t_grid))
    wt = rng.normal(0.0, params["wind_std_wt"], size=len(t_grid))
    wz = rng.normal(0.0, params["wind_std_wz"], size=len(t_grid))
    return wr, wt, wz


# ============================================================
# 9) Simulace
# ============================================================
t0 = 0.0
tf = 100.0
t_eval = np.linspace(t0, tf, 3000)

wind_wr_samples, wind_wt_samples, wind_wz_samples = generate_wind_samples(t_eval)


def disturbance_input(t):
    wr = np.interp(t, t_eval, wind_wr_samples)
    wt = np.interp(t, t_eval, wind_wt_samples)
    wz = np.interp(t, t_eval, wind_wz_samples)
    return np.array([wr, wt, wz], dtype=float)


def rhs_open_loop(t, x):
    u = np.array([0.0, 0.0], dtype=float)
    d = disturbance_input(t)
    return plant_dynamics(x, u, d)


def rhs_closed_loop(t, x):
    u = control_input(x, K, params)
    d = disturbance_input(t)
    return plant_dynamics(x, u, d)


sol_open = solve_ivp(
    rhs_open_loop,
    (t0, tf),
    x0,
    t_eval=t_eval,
    method="RK45"
)

sol_lqr = solve_ivp(
    rhs_closed_loop,
    (t0, tf),
    x0,
    t_eval=t_eval,
    method="RK45"
)

if not sol_open.success:
    raise RuntimeError(sol_open.message)

if not sol_lqr.success:
    raise RuntimeError(sol_lqr.message)


# ============================================================
# 10) Ploty
# ============================================================
control_method = "LQR"
payload_ref = payload_polar_position(x_eq, params)

plot_angles(sol_open, sol_lqr, control_method=control_method, x_ref=x_eq)
plot_positions(sol_open, sol_lqr, control_method=control_method, x_ref=x_eq)
plot_velocities(sol_open, sol_lqr, control_method=control_method, x_ref=x_eq)
plot_payload_trajectory(sol_open, sol_lqr, params, control_method=control_method, x_ref=x_eq)
plot_payload_position_error(sol_open, sol_lqr, params, payload_ref, control_method=control_method)
plot_control_inputs(
    sol_lqr,
    control_input,
    K,
    params,
    control_method=control_method,
    u_ref=u_eq
)

visualize_crane_meshcat(sol_lqr, params, t_eval)
