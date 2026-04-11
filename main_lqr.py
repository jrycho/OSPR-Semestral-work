from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.signal import cont2discrete
from constants import params, x0
from symbolic_model import build_crane_model
from mat_model import crane_dynamics
from linearization import linearize_system
from visualization import (
    plot_angles,
    plot_positions,
    plot_velocities,
    plot_control_inputs,
    plot_payload_trajectory
)
from meshcat_crane_viz import visualize_crane_meshcat
import numpy as np
import matplotlib.pyplot as plt

# symbolické odvození modelu
M_func, forcing_func = build_crane_model()

# random number generator for disturbances
rng = np.random.default_rng(seed=42)


# vstupní funkce (kontrolér) 
def control_input(t, x, x_eq, u_eq, K, params):
    dx = x - x_eq
    u = u_eq - K @ dx
    u[0] = np.clip(u[0], -params["F_r_max"], params["F_r_max"])
    u[1] = np.clip(u[1], -params["M_phi_max"], params["M_phi_max"])
    return u

def disturbance_input(t, x):
    wr = np.interp(t, t_eval, wind_wr_samples)
    wt = np.interp(t, t_eval, wind_wt_samples)
    wz = np.interp(t, t_eval, wind_wz_samples)
    return np.array([wr, wt, wz], dtype=float)

def rhs_closed_loop(t, x):
    u = control_input(t, x, x_eq, u_eq, K, params)
    d = disturbance_input(t, x)
    return crane_dynamics(t, x, u, d, params, M_func, forcing_func)

def rhs_open_loop(t, x):
    u = np.array([0.0, 0.0], dtype=float)
    d = disturbance_input(t, x)
    return crane_dynamics(t, x, u, d, params, M_func, forcing_func)


import numpy as np
from scipy.signal import cont2discrete

def discretize_system(A, B, Ts):
    """
    Diskretizace spojité lineární soustavy:
        x_dot = A x + B u

    na:
        x[k+1] = Ad x[k] + Bd u[k]
    """
    nx = A.shape[0]
    nu = B.shape[1]

    C = np.eye(nx)
    D = np.zeros((nx, nu))

    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), Ts)
    return Ad, Bd
"""
# pravá strana ODE pro integraci
def rhs(t, x):
    u = control_input(t, x)
    return crane_dynamics(t, x, u, params, M_func, forcing_func)

# 4) Nastavení simulace
t_start = 0.0
t_end = 10.0
n_points = 1000
t_eval = np.linspace(t_start, t_end, n_points)


# 5) Numerická integrace
sol = solve_ivp(
    rhs,
    (t_start, t_end),
    x0,
    t_eval=t_eval,
    method="RK45"
)


# 6) Kontrola úspěšnosti
if not sol.success:
    raise RuntimeError(f"Simulace selhala: {sol.message}")


# 7) Rozbalení výsledků
t = sol.t
r = sol.y[0]
phi = sol.y[1]
alpha = sol.y[2]
beta = sol.y[3]
dr = sol.y[4]
dphi = sol.y[5]
dalpha = sol.y[6]
dbeta = sol.y[7]


# 8) Vykreslení
plt.figure(figsize=(10, 6))
plt.plot(t, r, label="r [m]")
plt.plot(t, phi, label="phi [rad]")
plt.plot(t, alpha, label="alpha [rad]")
plt.plot(t, beta, label="beta [rad]")
plt.xlabel("čas [s]")
plt.ylabel("stav")
plt.title("Průběh stavů jeřábu")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(t, dr, label="dr [m/s]")
plt.plot(t, dphi, label="dphi [rad/s]")
plt.plot(t, dalpha, label="dalpha [rad/s]")
plt.plot(t, dbeta, label="dbeta [rad/s]")
plt.xlabel("čas [s]")
plt.ylabel("rychlost")
plt.title("Průběh rychlostí")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""

r0 = 3.0
phi0 = 0.0
x_eq = np.array([r0, phi0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u_eq = np.array([0.0, 0.0])

def f_xu(x, u):
    d = np.array([0.0, 0.0, 0.0], dtype=float)
    return crane_dynamics(0.0, x, u, d, params, M_func, forcing_func)


A, B = linearize_system(f_xu, x_eq, u_eq)

np.set_printoptions(precision=5, suppress=True)

Q = np.diag([800.0, 800.0, 3000.0, 3000.0, 150.0, 20.0, 1000.0, 1000.0])
R = np.diag([0.001, 0.001])
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("Matice A:")
print(A)
print("\nMatice B:")
print(B)
print("\nMatice K (LQR zisk):")
print(K)

t0 = 0.0
tf = 100.0
t_eval = np.linspace(t0, tf, 3000)

# náhodný vítr pro poruchy
rng = np.random.default_rng(seed=42)

wind_wr_samples = rng.normal(0.0, params["wind_std_wr"], size=len(t_eval))
wind_wt_samples = rng.normal(0.0, params["wind_std_wt"], size=len(t_eval))
wind_wz_samples = rng.normal(0.0, params["wind_std_wz"], size=len(t_eval))

sol_open = solve_ivp(
    rhs_open_loop,
    (t0, tf),
    x0,
    t_eval=t_eval,
    method="RK45"
)

sol_closed = solve_ivp(
    rhs_closed_loop,
    (t0, tf),
    x0,
    t_eval=t_eval,
    method="RK45"
)

if not sol_open.success:
    raise RuntimeError(sol_open.message)

if not sol_closed.success:
    raise RuntimeError(sol_closed.message)

control_method = "LQR"

plot_angles(sol_open, sol_closed, control_method=control_method, x_ref=x_eq)
plot_positions(sol_open, sol_closed, control_method=control_method, x_ref=x_eq)
plot_velocities(sol_open, sol_closed, control_method=control_method, x_ref=x_eq)
plot_control_inputs(
    sol_closed,
    control_input,
    x_eq,
    u_eq,
    K,
    params,
    control_method=control_method,
    u_ref=u_eq
)
plot_payload_trajectory(sol_open, sol_closed, params, control_method=control_method, x_ref=x_eq)

# Meshcat visualization
visualize_crane_meshcat(sol_closed, params, t_eval)
