import numpy as np


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