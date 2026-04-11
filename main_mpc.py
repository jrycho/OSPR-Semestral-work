from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

from constants import params, x0
from symbolic_model import build_crane_model
from mat_model import crane_dynamics
from linearization import linearize_system
from visualization import (
    plot_angles,
    plot_positions,
    plot_velocities,
    plot_payload_trajectory
)
from meshcat_crane_viz import visualize_crane_meshcat

np.set_printoptions(precision=5, suppress=True)

# ============================================================
# 1) Symbolický model -> numerické funkce
# ============================================================
M_func, forcing_func = build_crane_model()

# ============================================================
# 2) Nelineární plant
# ============================================================
def plant_dynamics(x, u, d=None):
    if d is None:
        d = np.zeros(3, dtype=float)

    return crane_dynamics(
        0.0, x, u, d, params, M_func, forcing_func
    )

# ============================================================
# 3) Pracovní bod
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
def f_xu(x, u):
    d = np.zeros(3, dtype=float)
    return plant_dynamics(x, u, d)

A, B = linearize_system(f_xu, x_eq, u_eq)

print("Matice A:")
print(A)
print("\nMatice B:")
print(B)

# ============================================================
# 5) Diskretizace
# ============================================================
def discretize_system(A, B, Ts):
    nx = A.shape[0]
    nu = B.shape[1]
    C = np.eye(nx)
    D = np.zeros((nx, nu))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), Ts)
    return Ad, Bd

Ts = 0.01
Ad, Bd = discretize_system(A, B, Ts)

nx = A.shape[0]
nu = B.shape[1]

print("\nMatice Ad:")
print(Ad)
print("\nMatice Bd:")
print(Bd)

# ============================================================
# 6) Reference
# ============================================================
x_ref = np.array([
    3.0,   # r_target
    0.0,   # phi_target
    0.0,   # alpha_t
    0.0,   # alpha_n
    0.0,   # dr
    0.0,   # dphi
    0.0,   # dalpha_t
    0.0    # dalpha_n
], dtype=float)

x_ref_tilde = x_ref - x_eq

# ============================================================
# 7) Váhy MPC
# ============================================================
Q = np.diag([800.0, 800.0, 3000.0, 3000.0, 150.0, 150.0, 1000.0, 1000.0])
R = np.diag([0.001, 0.001])
P = Q.copy()

# ============================================================
# 8) Neomezené MPC přes Riccatiho rekurzi
# ============================================================
class UnconstrainedMPC:
    def __init__(self, Ad, Bd, Q, R, P, N):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]

    def solve(self, x0_tilde, x_ref_tilde):
        """
        Vrátí první optimální vstup pro neomezené konečněhorizontové MPC.
        Řeší tracking přes augmentaci na chybu e = x - x_ref.
        """
        A = self.Ad
        B = self.Bd
        Q = self.Q
        R = self.R
        P = self.P
        N = self.N

        e0 = x0_tilde - x_ref_tilde

        # zpětná Riccatiho rekurze
        S = [None] * (N + 1)
        K = [None] * N
        S[N] = P

        for k in range(N - 1, -1, -1):
            BtSB = B.T @ S[k + 1] @ B
            inv_term = np.linalg.inv(R + BtSB)
            K[k] = inv_term @ (B.T @ S[k + 1] @ A)
            S[k] = Q + A.T @ S[k + 1] @ A - A.T @ S[k + 1] @ B @ K[k]

        # první krok
        u0_tilde = -K[0] @ e0
        return u0_tilde

# ============================================================
# 9) Pomocné převody
# ============================================================
def state_to_deviation(x):
    return x - x_eq

def input_from_deviation(u_tilde):
    return u_tilde + u_eq

# ============================================================
# 10) Porucha
# ============================================================
rng = np.random.default_rng(seed=42)

def generate_wind_samples(t_grid):
    wr = rng.normal(0.0, params["wind_std_wr"], size=len(t_grid))
    wt = rng.normal(0.0, params["wind_std_wt"], size=len(t_grid))
    wz = rng.normal(0.0, params["wind_std_wz"], size=len(t_grid))
    return wr, wt, wz

# ============================================================
# 11) Open-loop simulace
# ============================================================
t0 = 0.0
tf = 30.0
t_eval = np.arange(t0, tf + Ts, Ts)

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

sol_open = solve_ivp(
    rhs_open_loop,
    (t0, tf),
    x0,
    t_eval=t_eval,
    method="RK45"
)

if not sol_open.success:
    raise RuntimeError(sol_open.message)

# ============================================================
# 12) Closed-loop simulace s neomezeným MPC
# ============================================================
N = 30

mpc = UnconstrainedMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    P=P,
    N=N
)

def simulate_nonlinear_step(x, u, Ts, d):
    def rhs(t, xx):
        return plant_dynamics(xx, u, d)

    sol = solve_ivp(
        rhs,
        (0.0, Ts),
        x,
        t_eval=[Ts],
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[:, -1]

x_hist = np.zeros((nx, len(t_eval)))
u_hist = np.zeros((nu, len(t_eval) - 1))

x_current = x0.copy()
x_hist[:, 0] = x_current

for k in range(len(t_eval) - 1):
    d_k = np.array([
        wind_wr_samples[k],
        wind_wt_samples[k],
        wind_wz_samples[k]
    ], dtype=float)

    x_tilde = state_to_deviation(x_current)

    u_tilde = mpc.solve(
        x0_tilde=x_tilde,
        x_ref_tilde=x_ref_tilde
    )

    u = input_from_deviation(u_tilde)

    # volitelná fyzikální saturace plantu
    u[0] = np.clip(u[0], -params["F_r_max"], params["F_r_max"])
    u[1] = np.clip(u[1], -params["M_phi_max"], params["M_phi_max"])

    u_hist[:, k] = u
    x_next = simulate_nonlinear_step(x_current, u, Ts, d_k)

    x_hist[:, k + 1] = x_next
    x_current = x_next

sol_mpc = SimpleNamespace(t=t_eval, y=x_hist)

# ============================================================
# 13) Plot vstupů
# ============================================================
def plot_control_inputs_mpc(t, u_hist, control_method="MPC", u_ref=None):
    plt.figure(figsize=(10, 6))
    plt.plot(t[:-1], u_hist[0, :], label=f"F_r ({control_method}) [N]")
    plt.plot(t[:-1], u_hist[1, :], label=f"M_phi ({control_method}) [Nm]")
    if u_ref is not None:
        plt.hlines(
            u_ref[0],
            t[0],
            t[-2],
            colors="0.35",
            linestyles=":",
            linewidth=1.5,
            label=f"F_r target = {u_ref[0]:.3g} N"
        )
        plt.hlines(
            u_ref[1],
            t[0],
            t[-2],
            colors="0.25",
            linestyles=":",
            linewidth=1.5,
            label=f"M_phi target = {u_ref[1]:.3g} Nm"
        )
    plt.xlabel("čas [s]")
    plt.ylabel("vstupy")
    plt.title("Průběh řídicích vstupů - neomezené MPC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 14) Ploty
# ============================================================
control_method = "MPC"

plot_angles(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_positions(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_velocities(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_payload_trajectory(sol_open, sol_mpc, params, control_method=control_method, x_ref=x_ref)
plot_control_inputs_mpc(t_eval, u_hist, control_method=control_method, u_ref=u_eq)

visualize_crane_meshcat(sol_mpc, params, t_eval)
