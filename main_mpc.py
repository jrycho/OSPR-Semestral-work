from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from types import SimpleNamespace
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from constants import params, x0
from symbolic_model import build_crane_model
from mat_model import crane_dynamics
from linearization import linearize_system
from visualization import (
    payload_polar_position,
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
A, B = linearize_system(plant_dynamics, x_eq, u_eq)

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

N = 50
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


Q_state = np.diag([5.0, 5.0, 80.0, 80.0, 5.0, 2.0, 30.0, 30.0])
Q_payload = np.diag([1500.0, 2500.0])

C_payload = linearize_payload_position(x_ref, params)

Q = Q_state + C_payload.T @ Q_payload @ C_payload
R = np.diag([400.0, 200.0])
u_scale = np.array([params["F_r_max"], params["M_phi_max"]], dtype=float)
Bd_mpc = Bd @ np.diag(u_scale)
P = solve_discrete_are(Ad, Bd_mpc, Q, R)


# ============================================================
# 8) MPC s omezenim vstupu
# ============================================================
class CvxpyMPC:
    """
    Finite-horizon MPC written as a CVXPY optimization problem.

    The constraints are the discrete prediction model and actuator limits:

        x[k+1] = Ad*x[k] + Bd_scaled*v[k]
        -1 <= v[k] <= 1

    v[k] is normalized actuator usage. The physical input sent to the
    nonlinear plant is:

        u[k] = u_scale * v[k]

    The minimized cost function uses payload-focused Q and normalized input R:

        min_U J(U)

    where:

        J(U) =
            sum from k = 0 to N-1:
                x_error[k].T Q x_error[k] + v[k].T R v[k]
            + x_error[N].T P x_error[N]

        U = [v[0], v[1], ..., v[N-1]]
    """

    def __init__(self, Ad, Bd_scaled, Q, R, P, N, u_scale):
        self.Ad = Ad
        self.Bd = Bd_scaled
        self.Q = Q
        self.R = R
        self.P = P
        self.N = N
        self.u_scale = u_scale
        self.nx = Ad.shape[0]
        self.nu = Bd_scaled.shape[1]

        self.x0_param = cp.Parameter(self.nx)
        self.xr_param = cp.Parameter(self.nx)

        self.x = cp.Variable((self.nx, N + 1))
        self.v = cp.Variable((self.nu, N))

        cost = 0
        constraints = [self.x[:, 0] == self.x0_param]

        for k in range(N):
            x_error = self.x[:, k] - self.xr_param
            cost += cp.quad_form(x_error, cp.psd_wrap(Q))
            cost += cp.quad_form(self.v[:, k], cp.psd_wrap(R))

            constraints += [
                self.x[:, k + 1] == Ad @ self.x[:, k] + Bd_scaled @ self.v[:, k],
                self.v[:, k] >= -1.0,
                self.v[:, k] <= 1.0
            ]

        terminal_error = self.x[:, N] - self.xr_param
        cost += cp.quad_form(terminal_error, cp.psd_wrap(P))

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x0_tilde, xr_tilde):
        self.x0_param.value = x0_tilde
        self.xr_param.value = xr_tilde

        solver_attempts = [

            (
                cp.CLARABEL,
                {
                    "warm_start": True,
                    "verbose": False,
                    "max_iter": 10000
                }
            ),      
            (
                cp.SCS,
                {
                    "warm_start": True,
                    "verbose": False,
                    "max_iters": 20000,
                    "eps": 1e-3
                }
            ),
            (
                cp.OSQP,
                {
                    "warm_start": True,
                    "verbose": False,
                    "max_iter": 200000,
                    "eps_abs": 1e-2,
                    "eps_rel": 1e-2,
                    "polish": False
                }
            )

        ]

        for solver, options in solver_attempts:
            try:
                self.problem.solve(solver=solver, **options)
            except cp.SolverError:
                continue

            if self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return self.u_scale * self.v[:, 0].value

        return None

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
tf = 100.0
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
# 12) Closed-loop simulace s MPC
# ============================================================
mpc = CvxpyMPC(
    Ad=Ad,
    Bd_scaled=Bd_mpc,
    Q=Q,
    R=R,
    P=P,
    N=N,
    u_scale=u_scale
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
    u_tilde = mpc.solve(x_tilde, x_ref_tilde)

    if u_tilde is None:
        raise RuntimeError(f"MPC solve failed: {mpc.problem.status}")

    u = input_from_deviation(u_tilde)
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
    plt.title("Prubeh ridicich vstupu - MPC s omezenim vstupu")
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
