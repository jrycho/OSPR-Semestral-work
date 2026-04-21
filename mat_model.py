import numpy as np

"""
Computes the state-space dynamics of a crane system.

The function represents the nonlinear equations of motion of a crane with a suspended load.
It takes the current state, control inputs, external disturbances, and system parameters,
and returns the time derivative of the state vector.

State vector:
    x = [r, phi, alpha, beta, dr, dphi, dalpha, dbeta]
        r      - radial position (trolley position)
        phi    - rotation angle of the crane
        alpha  - load swing angle in one plane
        beta   - load swing angle in the other plane
        dr, dphi, dalpha, dbeta - corresponding velocities

Inputs:
    u = [F_r, M_phi]
        F_r    - force applied to the trolley (radial direction)
        M_phi  - torque applied to crane rotation

Disturbances:
    disturbance = [Fwr, Fwt, Fwz]
        External forces acting on the load (e.g., wind)

The dynamics are defined in the standard form:
    M(q) * ddq = forcing(q, dq, u, disturbance)

where:
    M        - mass/inertia matrix (provided by M_func)
    forcing  - generalized forces vector (provided by forcing_func)

The function solves for accelerations (ddq) and returns:
    dx/dt = [dr, dphi, dalpha, dbeta, ddr, ddphi, ddalpha, ddbeta]

This function is suitable for numerical integration (e.g., scipy.integrate.solve_ivp)
and control system design.
"""


def crane_dynamics(t, x, u, disturbance, params, M_func, forcing_func):
    r, phi, alpha, beta, dr, dphi, dalpha, dbeta = x
    F_r, M_phi = u
    Fwr, Fwt, Fwz = disturbance

    mv = params["m_v"]
    mb = params["m_b"]
    l = params["l"]
    h = params["h"]
    g = params["g"]
    Jphi = params["J_phi"]

    d_r, d_phi, d_alpha, d_beta = params["d_r"], params["d_phi"], params["d_alpha"], params["d_beta"]
    
    M = M_func(
        r, phi, alpha, beta,
        dr, dphi, dalpha, dbeta,
        F_r, M_phi,
        Fwr, Fwt, Fwz,
        mv, mb, l, h, g, Jphi,
        d_r, d_phi, d_alpha, d_beta
    )


    forcing = forcing_func(
        r, phi, alpha, beta,
        dr, dphi, dalpha, dbeta,
        F_r, M_phi,
        Fwr, Fwt, Fwz,
        mv, mb, l, h, g, Jphi,
        d_r, d_phi, d_alpha, d_beta
    )

    M = np.asarray(M, dtype=float)
    forcing = np.asarray(forcing, dtype=float).reshape(4,)

    ddq = np.linalg.solve(M, forcing)

    ddr, ddphi, ddalpha, ddbeta = ddq

    return np.array([
        dr,
        dphi,
        dalpha,
        dbeta,
        ddr,
        ddphi,
        ddalpha,
        ddbeta
    ], dtype=float)