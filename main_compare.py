from scipy.signal import cont2discrete
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, solve_discrete_are
from types import SimpleNamespace
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from constants import params, x0
from symbolic_model import build_crane_model
from mat_model import crane_dynamics
from linearization import linearize_system
from visualization import payload_polar_position, payload_position

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
# 3) Pracovni bod a reference
# ============================================================
r0 = 3.0
phi0 = 0.0

x_eq = np.array([
    r0,    # r
    phi0,  # phi
    0.0,   # alpha
    0.0,   # beta
    0.0,   # dr
    0.0,   # dphi
    0.0,   # dalpha
    0.0    # dbeta
], dtype=float)

u_eq = np.array([
    0.0,   # F_r
    0.0    # M_phi
], dtype=float)

# Both controllers regulate to the same point for a fair comparison.
x_ref = x_eq.copy()
x_ref_tilde = x_ref - x_eq


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
# 4) Linearizace a diskretizace
# ============================================================
A, B = linearize_system(plant_dynamics, x_eq, u_eq)

print("Matice A:")
print(A)
print("\nMatice B:")
print(B)


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


# ============================================================
# 5) Spolecne vahy
# ============================================================
Q_state = np.diag([5.0, 5.0, 80.0, 80.0, 5.0, 2.0, 30.0, 30.0])
Q_payload = np.diag([1500.0, 2500.0])

C_payload = linearize_payload_position(x_ref, params)
Q = Q_state + C_payload.T @ Q_payload @ C_payload

# MPC penalizes normalized actuator usage v = u / u_scale.
R_mpc = np.diag([400.0, 200.0])
u_scale = np.array([params["F_r_max"], params["M_phi_max"]], dtype=float)
Bd_mpc = Bd @ np.diag(u_scale)

# LQR uses physical inputs u = [F_r, M_phi], so convert the normalized MPC
# input penalty back to physical input units.
R_lqr = np.diag([
    R_mpc[0, 0] / params["F_r_max"]**2,
    R_mpc[1, 1] / params["M_phi_max"]**2
])


# ============================================================
# 6) LQR
# ============================================================
P_lqr = solve_continuous_are(A, B, Q, R_lqr)
K_lqr = np.linalg.inv(R_lqr) @ B.T @ P_lqr

print("\nMatice K (LQR zisk):")
print(K_lqr)


def control_input_lqr(x):
    dx = x - x_eq
    dx[1] = wrap_angle(dx[1])

    u = u_eq - K_lqr @ dx
    u[0] = np.clip(u[0], -params["F_r_max"], params["F_r_max"])
    u[1] = np.clip(u[1], -params["M_phi_max"], params["M_phi_max"])
    return u


# ============================================================
# 7) MPC
# ============================================================
P_mpc = solve_discrete_are(Ad, Bd_mpc, Q, R_mpc)


class CvxpyMPC:
    """
    Finite-horizon MPC with normalized actuator constraints.

    Prediction:
        x[k+1] = Ad*x[k] + Bd_scaled*v[k]

    Constraints:
        -1 <= v[k] <= 1

    Cost:
        sum x_error[k].T Q x_error[k] + v[k].T R v[k]
        + x_error[N].T P x_error[N]
    """

    def __init__(self, Ad, Bd_scaled, Q, R, P, N, u_scale):
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
            (cp.CLARABEL, {"warm_start": True, "verbose": False, "max_iter": 10000}),
            (cp.OSQP, {
                "warm_start": True,
                "verbose": False,
                "max_iter": 200000,
                "eps_abs": 1e-2,
                "eps_rel": 1e-2,
                "polish": False
            }),
            (cp.SCS, {"warm_start": True, "verbose": False, "max_iters": 20000, "eps": 1e-3})
        ]

        for solver, options in solver_attempts:
            try:
                self.problem.solve(solver=solver, **options)
            except cp.SolverError:
                continue

            if self.problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return self.u_scale * self.v[:, 0].value

        return None


mpc = CvxpyMPC(
    Ad=Ad,
    Bd_scaled=Bd_mpc,
    Q=Q,
    R=R_mpc,
    P=P_mpc,
    N=N,
    u_scale=u_scale
)


def state_to_deviation(x):
    dx = x - x_eq
    dx[1] = wrap_angle(dx[1])
    return dx


def input_from_deviation(u_tilde):
    return u_tilde + u_eq


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
t_eval = np.arange(t0, tf + Ts, Ts)

wind_wr_samples, wind_wt_samples, wind_wz_samples = generate_wind_samples(t_eval)


def disturbance_input(t):
    wr = np.interp(t, t_eval, wind_wr_samples)
    wt = np.interp(t, t_eval, wind_wt_samples)
    wz = np.interp(t, t_eval, wind_wz_samples)
    return np.array([wr, wt, wz], dtype=float)


def rhs_lqr(t, x):
    u = control_input_lqr(x)
    d = disturbance_input(t)
    return plant_dynamics(x, u, d)


def simulate_nonlinear_step(x, u, Ts, d):
    def rhs(t, xx):
        return plant_dynamics(xx, u, d)

    sol = solve_ivp(rhs, (0.0, Ts), x, t_eval=[Ts], method="RK45")

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y[:, -1]


sol_lqr = solve_ivp(rhs_lqr, (t0, tf), x0, t_eval=t_eval, method="RK45")

if not sol_lqr.success:
    raise RuntimeError(sol_lqr.message)

x_mpc_hist = np.zeros((nx, len(t_eval)))
u_mpc_hist = np.zeros((nu, len(t_eval) - 1))

x_current = x0.copy()
x_mpc_hist[:, 0] = x_current

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
    u_mpc_hist[:, k] = u

    x_next = simulate_nonlinear_step(x_current, u, Ts, d_k)
    x_mpc_hist[:, k + 1] = x_next
    x_current = x_next

sol_mpc = SimpleNamespace(t=t_eval, y=x_mpc_hist)

u_lqr_hist = np.array([
    control_input_lqr(x)
    for x in sol_lqr.y.T
]).T


# ============================================================
# 10) Porovnavaci ploty
# ============================================================
def plot_state_compare(indexes, labels, title, ylabel):
    plt.figure(figsize=(10, 6))

    for idx, label in zip(indexes, labels):
        plt.plot(sol_lqr.t, sol_lqr.y[idx], "-", label=f"{label} LQR")
        plt.plot(sol_mpc.t, sol_mpc.y[idx], "--", label=f"{label} MPC")
        plt.hlines(x_ref[idx], t_eval[0], t_eval[-1], colors="0.35", linestyles=":")

    plt.xlabel("cas [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def payload_error(sol):
    payload = np.array([
        payload_polar_position(x, params)
        for x in sol.y.T
    ])
    error = payload - payload_polar_position(x_ref, params)
    error[:, 1] = wrap_angle(error[:, 1])
    return error


def plot_payload_error_compare():
    lqr_error = payload_error(sol_lqr)
    mpc_error = payload_error(sol_mpc)

    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, lqr_error[:, 0], "-", label="rho LQR")
    plt.plot(t_eval, mpc_error[:, 0], "--", label="rho MPC")
    plt.hlines(0.0, t_eval[0], t_eval[-1], colors="0.35", linestyles=":")
    plt.xlabel("cas [s]")
    plt.ylabel("rho error [m]")
    plt.title("Chyba radialni polohy bremene")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, lqr_error[:, 1], "-", label="theta LQR")
    plt.plot(t_eval, mpc_error[:, 1], "--", label="theta MPC")
    plt.hlines(0.0, t_eval[0], t_eval[-1], colors="0.35", linestyles=":")
    plt.xlabel("cas [s]")
    plt.ylabel("theta error [rad]")
    plt.title("Chyba uhlove polohy bremene")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_control_compare():
    plt.figure(figsize=(10, 6))
    plt.plot(sol_lqr.t, u_lqr_hist[0, :], "-", label="F_r LQR")
    plt.plot(t_eval[:-1], u_mpc_hist[0, :], "--", label="F_r MPC")
    plt.hlines(0.0, t_eval[0], t_eval[-1], colors="0.35", linestyles=":")
    plt.xlabel("cas [s]")
    plt.ylabel("F_r [N]")
    plt.title("Porovnani radialni sily")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sol_lqr.t, u_lqr_hist[1, :], "-", label="M_phi LQR")
    plt.plot(t_eval[:-1], u_mpc_hist[1, :], "--", label="M_phi MPC")
    plt.hlines(0.0, t_eval[0], t_eval[-1], colors="0.35", linestyles=":")
    plt.xlabel("cas [s]")
    plt.ylabel("M_phi [Nm]")
    plt.title("Porovnani momentu otaceni")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_cost_summary(sol, u_hist, method_name):
    state_error = sol.y.T - x_ref
    state_error[:, 1] = np.array([
        wrap_angle(angle)
        for angle in state_error[:, 1]
    ])

    payload = np.array([
        payload_polar_position(x, params)
        for x in sol.y.T
    ])
    payload_ref = payload_polar_position(x_ref, params)
    payload_error = payload - payload_ref
    payload_error[:, 1] = np.array([
        wrap_angle(angle)
        for angle in payload_error[:, 1]
    ])

    state_cost_samples = np.einsum("ij,jk,ik->i", state_error, Q, state_error)
    payload_cost_samples = np.einsum(
        "ij,jk,ik->i",
        payload_error,
        Q_payload,
        payload_error
    )

    normalized_u = u_hist.T / u_scale
    input_cost_samples = np.einsum(
        "ij,jk,ik->i",
        normalized_u,
        R_mpc,
        normalized_u
    )

    state_cost = np.trapz(state_cost_samples, sol.t)
    payload_cost = np.trapz(payload_cost_samples, sol.t)
    input_cost = np.sum(input_cost_samples) * Ts
    total_cost = state_cost + input_cost

    return {
        "method": method_name,
        "state_cost": state_cost,
        "payload_cost": payload_cost,
        "input_cost": input_cost,
        "total_cost": total_cost,
        "max_abs_Fr": np.max(np.abs(u_hist[0, :])),
        "max_abs_Mphi": np.max(np.abs(u_hist[1, :]))
    }


def print_cost_summary():
    lqr_summary = calculate_cost_summary(sol_lqr, u_lqr_hist, "LQR")
    mpc_summary = calculate_cost_summary(sol_mpc, u_mpc_hist, "MPC")

    print("\nPorovnani nakladu:")
    print("method | total_cost | state_cost | payload_cost | input_cost | max |F_r| | max |M_phi|")

    for summary in (lqr_summary, mpc_summary):
        print(
            f"{summary['method']:>6} | "
            f"{summary['total_cost']:>10.3f} | "
            f"{summary['state_cost']:>10.3f} | "
            f"{summary['payload_cost']:>12.3f} | "
            f"{summary['input_cost']:>10.3f} | "
            f"{summary['max_abs_Fr']:>9.3f} | "
            f"{summary['max_abs_Mphi']:>11.3f}"
        )


def plot_payload_trajectory_compare():
    traj_lqr = np.array([
        payload_position(x, params)
        for x in sol_lqr.y.T
    ])
    traj_mpc = np.array([
        payload_position(x, params)
        for x in sol_mpc.y.T
    ])

    fig, ax = plt.subplots(figsize=(7, 7))

    points_lqr = traj_lqr[:, :2]
    segments_lqr = np.array([
        points_lqr[i:i + 2]
        for i in range(len(points_lqr) - 1)
    ])
    lc_lqr = LineCollection(
        segments_lqr,
        cmap="viridis",
        norm=plt.Normalize(t_eval.min(), t_eval.max()),
        linestyle="-"
    )
    lc_lqr.set_array(t_eval[:-1])
    ax.add_collection(lc_lqr)

    points_mpc = traj_mpc[:, :2]
    segments_mpc = np.array([
        points_mpc[i:i + 2]
        for i in range(len(points_mpc) - 1)
    ])
    lc_mpc = LineCollection(
        segments_mpc,
        cmap="plasma",
        norm=plt.Normalize(t_eval.min(), t_eval.max()),
        linestyle="--"
    )
    lc_mpc.set_array(t_eval[:-1])
    ax.add_collection(lc_mpc)

    target_payload = payload_position(x_ref, params)
    ax.plot(
        target_payload[0],
        target_payload[1],
        "x",
        color="black",
        markersize=8,
        label=f"target payload ({target_payload[0]:.3g}, {target_payload[1]:.3g}) m"
    )

    ax.autoscale()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajektorie bremene - LQR vs MPC")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()

    plt.colorbar(lc_lqr, ax=ax, shrink=0.8, label="cas [s] - LQR")
    plt.colorbar(lc_mpc, ax=ax, shrink=0.8, label="cas [s] - MPC")

    plt.tight_layout()
    plt.show()


plot_state_compare([0, 1], ["r", "phi"], "Porovnani poloh", "poloha")
plot_state_compare([2, 3], ["alpha", "beta"], "Porovnani kyvu bremene", "uhel [rad]")
plot_state_compare([6, 7], ["dalpha", "dbeta"], "Porovnani uhlovych rychlosti", "rychlost [rad/s]")
plot_payload_error_compare()
plot_payload_trajectory_compare()
plot_control_compare()
print_cost_summary()
