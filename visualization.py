import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


def _method_label(control_method):
    return str(control_method).strip() if control_method else "controller"


def _target_label(name, value, unit):
    return f"{name} target = {value:.3g} {unit}".rstrip()


def _plot_target_line(t, value, name, unit):
    plt.hlines(
        value,
        t[0],
        t[-1],
        colors="0.35",
        linestyles=":",
        linewidth=1.5,
        label=_target_label(name, value, unit),
    )


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def payload_polar_position(state, params):
    r, phi, alpha, beta = state[:4]
    l = params["l"]

    radial_offset = r + l * np.sin(alpha) * np.cos(beta)
    tangential_offset = l * np.sin(beta)

    rho = np.hypot(radial_offset, tangential_offset)
    theta = phi + np.arctan2(tangential_offset, radial_offset)

    return np.array([rho, theta], dtype=float)


# =========================================================
# 1) Uhly kyvu (alpha, beta)
# =========================================================
def plot_angles(sol_open, sol_closed, control_method="LQR", x_ref=None):
    method = _method_label(control_method)

    plt.figure(figsize=(10, 6))

    plt.plot(sol_open.t, sol_open.y[2], label=f"alpha bez {method}")
    plt.plot(sol_open.t, sol_open.y[3], label=f"beta bez {method}")

    plt.plot(sol_closed.t, sol_closed.y[2], "--", label=f"alpha s {method}")
    plt.plot(sol_closed.t, sol_closed.y[3], "--", label=f"beta s {method}")
    if x_ref is not None:
        _plot_target_line(sol_closed.t, x_ref[2], "alpha", "rad")
        _plot_target_line(sol_closed.t, x_ref[3], "beta", "rad")

    plt.xlabel("cas [s]")
    plt.ylabel("uhel [rad]")
    plt.title("Kyv bremene")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 2) Polohy (r, phi)
# =========================================================
def plot_positions(sol_open, sol_closed, control_method="LQR", x_ref=None):
    method = _method_label(control_method)

    plt.figure(figsize=(10, 6))

    plt.plot(sol_open.t, sol_open.y[0], label=f"r bez {method}")
    plt.plot(sol_open.t, sol_open.y[1], label=f"phi bez {method}")

    plt.plot(sol_closed.t, sol_closed.y[0], "--", label=f"r s {method}")
    plt.plot(sol_closed.t, sol_closed.y[1], "--", label=f"phi s {method}")
    if x_ref is not None:
        _plot_target_line(sol_closed.t, x_ref[0], "r", "m")
        _plot_target_line(sol_closed.t, x_ref[1], "phi", "rad")

    plt.xlabel("cas [s]")
    plt.ylabel("poloha")
    plt.title("Polohy systemu")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 3) Rychlosti
# =========================================================
def plot_velocities(sol_open, sol_closed, control_method="LQR", x_ref=None):
    method = _method_label(control_method)

    plt.figure(figsize=(10, 6))

    plt.plot(sol_open.t, sol_open.y[6], label=f"dalpha bez {method}")
    plt.plot(sol_open.t, sol_open.y[7], label=f"dbeta bez {method}")

    plt.plot(sol_closed.t, sol_closed.y[6], "--", label=f"dalpha s {method}")
    plt.plot(sol_closed.t, sol_closed.y[7], "--", label=f"dbeta s {method}")
    if x_ref is not None:
        _plot_target_line(sol_closed.t, x_ref[6], "dalpha", "rad/s")
        _plot_target_line(sol_closed.t, x_ref[7], "dbeta", "rad/s")

    plt.xlabel("cas [s]")
    plt.ylabel("rychlost [rad/s]")
    plt.title("Uhlove rychlosti")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 4) Ridici vstupy
# =========================================================
def plot_control_inputs(sol_closed, control_input, x_eq, u_eq, K, params, control_method="LQR", u_ref=None):
    method = _method_label(control_method)
    u_hist = np.array([
        control_input(t, x, x_eq, u_eq, K, params)
        for t, x in zip(sol_closed.t, sol_closed.y.T)
    ])

    plt.figure(figsize=(10, 6))
    plt.plot(sol_closed.t, u_hist[:, 0], label=f"F_r ({method})")
    plt.plot(sol_closed.t, u_hist[:, 1], label=f"M_phi ({method})")
    if u_ref is not None:
        _plot_target_line(sol_closed.t, u_ref[0], "F_r", "N")
        _plot_target_line(sol_closed.t, u_ref[1], "M_phi", "Nm")

    plt.xlabel("cas [s]")
    plt.ylabel("vstup")
    plt.title(f"Ridici zasahy {method}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 5) Poloha bremene v prostoru
# =========================================================
def payload_position(state, params):
    r, phi, alpha, beta = state[:4]
    l = params["l"]
    h = params["h"]

    e_r = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_t = np.array([-np.sin(phi), np.cos(phi), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])

    p_v = r * e_r + h * e_z

    s = l * (
        np.sin(alpha) * np.cos(beta) * e_r
        + np.sin(beta) * e_t
        - np.cos(alpha) * np.cos(beta) * e_z
    )

    return p_v + s


# =========================================================
# 6) Trajektorie bremene (x-y)
# =========================================================
def plot_payload_trajectory(sol_open, sol_closed, params, control_method="LQR", x_ref=None):
    method = _method_label(control_method)
    traj_open = np.array([payload_position(x, params) for x in sol_open.y.T])
    traj_closed = np.array([payload_position(x, params) for x in sol_closed.y.T])

    fig, ax = plt.subplots(figsize=(7, 7))

    points_open = traj_open[:, :2]
    segments_open = np.array([points_open[i:i + 2] for i in range(len(points_open) - 1)])
    lc_open = LineCollection(
        segments_open,
        cmap="viridis",
        norm=plt.Normalize(sol_open.t.min(), sol_open.t.max()),
    )
    lc_open.set_array(sol_open.t[:-1])
    ax.add_collection(lc_open)

    points_closed = traj_closed[:, :2]
    segments_closed = np.array([points_closed[i:i + 2] for i in range(len(points_closed) - 1)])
    lc_closed = LineCollection(
        segments_closed,
        cmap="viridis",
        norm=plt.Normalize(sol_closed.t.min(), sol_closed.t.max()),
        linestyle="--",
    )
    lc_closed.set_array(sol_closed.t[:-1])
    ax.add_collection(lc_closed)

    if x_ref is not None:
        target_payload = payload_position(x_ref, params)
        ax.plot(
            target_payload[0],
            target_payload[1],
            "x",
            color="black",
            markersize=8,
            label=f"target payload ({target_payload[0]:.3g}, {target_payload[1]:.3g}) m",
        )

    ax.autoscale()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajektorie bremene (pudorys)")
    ax.grid(True)
    ax.axis("equal")
    if x_ref is not None:
        ax.legend()

    plt.colorbar(lc_open, ax=ax, shrink=0.8, label=f"cas [s] - bez {method}")
    plt.colorbar(lc_closed, ax=ax, shrink=0.8, label=f"cas [s] - s {method}")

    plt.tight_layout()
    plt.show()


def plot_payload_position_error(sol_open, sol_closed, params, payload_ref, control_method="MPC"):
    method = _method_label(control_method)
    payload_open = np.array([
        payload_polar_position(x, params)
        for x in sol_open.y.T
    ])
    payload_closed = np.array([
        payload_polar_position(x, params)
        for x in sol_closed.y.T
    ])

    open_error = payload_open - payload_ref
    closed_error = payload_closed - payload_ref
    open_error[:, 1] = _wrap_angle(open_error[:, 1])
    closed_error[:, 1] = _wrap_angle(closed_error[:, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(sol_open.t, open_error[:, 0], label=f"rho error bez {method}")
    plt.plot(sol_closed.t, closed_error[:, 0], "--", label=f"rho error s {method}")
    plt.hlines(
        0.0,
        sol_closed.t[0],
        sol_closed.t[-1],
        colors="0.35",
        linestyles=":",
        linewidth=1.5,
        label="rho target error = 0 m",
    )
    plt.xlabel("cas [s]")
    plt.ylabel("rho error [m]")
    plt.title(f"Chyba polohy bremene rho - {method}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(sol_open.t, open_error[:, 1], label=f"theta error bez {method}")
    plt.plot(sol_closed.t, closed_error[:, 1], "--", label=f"theta error s {method}")
    plt.hlines(
        0.0,
        sol_closed.t[0],
        sol_closed.t[-1],
        colors="0.35",
        linestyles=":",
        linewidth=1.5,
        label="theta target error = 0 rad",
    )
    plt.xlabel("cas [s]")
    plt.ylabel("theta error [rad]")
    plt.title(f"Chyba polohy bremene theta - {method}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
