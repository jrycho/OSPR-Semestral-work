from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from types import SimpleNamespace
import cvxpy as cp
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
    plot_payload_trajectory,
    plot_payload_position_error
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

N = 20
Ts = 0.1
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
    4.0,   # r_target
    0.4,   # phi_target
    0.0,   # alpha_t
    0.0,   # alpha_n
    0.0,   # dr
    0.0,   # dphi
    0.0,   # dalpha_t
    0.0    # dalpha_n
], dtype=float)

x_ref_tilde = x_ref - x_eq


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# ============================================================
# 7) Váhy MPC
# ============================================================
def payload_polar_position(x, params):
    r, phi, alpha, beta = x[:4]
    l = params["l"]

    radial_offset = r + l * np.sin(alpha) * np.cos(beta)
    tangential_offset = l * np.sin(beta)

    rho = np.hypot(radial_offset, tangential_offset)
    theta = phi + np.arctan2(tangential_offset, radial_offset)

    return np.array([rho, theta], dtype=float)


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


Q_state = np.diag([800.0, 800.0, 800.0, 800.0, 150.0, 150.0, 1000.0, 1000.0])
Q_payload = np.diag([2000.0, 2000.0])

C_payload = linearize_payload_position(x_ref, params)

Q = Q_state + C_payload.T @ Q_payload @ C_payload
R = np.diag([0.001, 0.001])
Rd = np.diag([0.05, 0.05])
P = Q.copy()


# ============================================================
# 8) MPC pres celou posloupnost stavu a vstupu
# ============================================================
# Explicit constrained full-horizon MPC solved as a QP.
class SequenceMPC:
    def __init__(self, Ad, Bd, Q, R, Rd, P, N, u_min, u_max):
        self.Ad = Ad
        self.Bd = Bd
        self.Q = Q
        self.R = R
        self.Rd = Rd
        self.P = P
        self.N = N
        self.nx = Ad.shape[0]
        self.nu = Bd.shape[1]

        self.x = cp.Variable((self.nx, N + 1))
        self.u = cp.Variable((self.nu, N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_prev_param = cp.Parameter(self.nu)

        cost = 0
        constraints = [self.x[:, 0] == self.x0_param]

        for k in range(N):
            x_error = self.x[:, k] - self.x_ref_param
            cost += cp.quad_form(x_error, cp.psd_wrap(Q))
            cost += cp.quad_form(self.u[:, k], cp.psd_wrap(R))

            if k == 0:
                du = self.u[:, k] - self.u_prev_param
            else:
                du = self.u[:, k] - self.u[:, k - 1]
            cost += cp.quad_form(du, cp.psd_wrap(Rd))

            constraints += [
                self.x[:, k + 1] == Ad @ self.x[:, k] + Bd @ self.u[:, k],
                self.u[:, k] >= u_min,
                self.u[:, k] <= u_max
            ]

        terminal_error = self.x[:, N] - self.x_ref_param
        cost += cp.quad_form(terminal_error, cp.psd_wrap(P))

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x0_tilde, x_ref_tilde, u_prev_tilde):
        """
        Solves the full finite-horizon QP:
        x[:,0], ..., x[:,N] and u[:,0], ..., u[:,N-1].
        The simulation loop applies only u[:,0].
        """
        self.x0_param.value = x0_tilde
        self.x_ref_param.value = x_ref_tilde
        self.u_prev_param.value = u_prev_tilde

        self.problem.solve(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            max_iter=50000,
            eps_abs=1e-3,
            eps_rel=1e-3,
            polish=False
        )

        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.problem.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False,
                max_iter=2000
            )

        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC QP solve failed: {self.problem.status}")

        return self.u.value[:, 0], self.x.value, self.u.value

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
# 12) Closed-loop simulace s omezenym QP MPC
# ============================================================


mpc = SequenceMPC(
    Ad=Ad,
    Bd=Bd,
    Q=Q,
    R=R,
    Rd=Rd,
    P=P,
    N=N,
    u_min=np.array([-params["F_r_max"], -params["M_phi_max"]], dtype=float) - u_eq,
    u_max=np.array([params["F_r_max"], params["M_phi_max"]], dtype=float) - u_eq
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
x_pred_hist = np.zeros((nx, N + 1, len(t_eval) - 1))
u_pred_hist = np.zeros((nu, N, len(t_eval) - 1))

x_current = x0.copy()
x_hist[:, 0] = x_current
u_prev_tilde = np.zeros(nu, dtype=float)

for k in range(len(t_eval) - 1):
    d_k = np.array([
        wind_wr_samples[k],
        wind_wt_samples[k],
        wind_wz_samples[k]
    ], dtype=float)

    x_tilde = state_to_deviation(x_current)

    u_tilde, x_pred, u_pred = mpc.solve(
        x0_tilde=x_tilde,
        x_ref_tilde=x_ref_tilde,
        u_prev_tilde=u_prev_tilde
    )
    x_pred_hist[:, :, k] = x_pred
    u_pred_hist[:, :, k] = u_pred

    u = input_from_deviation(u_tilde)
    u_prev_tilde = u_tilde

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
    plt.title("Prubeh ridicich vstupu - omezene QP MPC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 14) Ploty
# ============================================================
control_method = "MPC"
payload_ref = payload_polar_position(x_ref, params)

plot_angles(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_positions(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_velocities(sol_open, sol_mpc, control_method=control_method, x_ref=x_ref)
plot_payload_trajectory(sol_open, sol_mpc, params, control_method=control_method, x_ref=x_ref)
plot_payload_position_error(sol_open, sol_mpc, params, payload_ref, control_method=control_method)
plot_control_inputs_mpc(t_eval, u_hist, control_method=control_method, u_ref=u_eq)

visualize_crane_meshcat(sol_mpc, params, t_eval)
