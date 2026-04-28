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

Matrix form used in this file:

    q =
        [ r      ]
        [ phi    ]
        [ alpha  ]
        [ beta   ]

    dq =
        [ dr      ]
        [ dphi    ]
        [ dalpha  ]
        [ dbeta   ]

    ddq =
        [ ddr      ]
        [ ddphi    ]
        [ ddalpha  ]
        [ ddbeta   ]

    M(q, dq) =
        [
          [mb + mv,
           -l*mb*sin(beta),
           l*mb*cos(alpha)*cos(beta),
           -l*mb*sin(alpha)*sin(beta)]

          [-l*mb*sin(beta),
           Jphi
             + mb*r**2 + mv*r**2
             + 2*l*mb*r*sin(alpha)*cos(beta)
             + l**2*mb*(1 - cos(alpha)**2*cos(beta)**2),
           -l**2*mb*cos(alpha)*sin(beta)*cos(beta),
           l*mb*(l*sin(alpha) + r*cos(beta))]

          [l*mb*cos(alpha)*cos(beta),
           -l**2*mb*cos(alpha)*sin(beta)*cos(beta),
           l**2*mb*cos(beta)**2,
           0]

          [-l*mb*sin(alpha)*sin(beta),
           l*mb*(l*sin(alpha) + r*cos(beta)),
           0,
           l**2*mb]
        ]

    forcing(q, dq, u, disturbance) =
        [ Qr      ]
        [ Qphi    ]
        [ Qalpha  ]
        [ Qbeta   ]

    The acceleration vector is obtained from:

        [mb + mv, ...                       ] [ ddr     ]   [ Qr     ]
        [-l*mb*sin(beta), ...               ] [ ddphi   ] = [ Qphi   ]
        [l*mb*cos(alpha)*cos(beta), ...     ] [ ddalpha ]   [ Qalpha ]
        [-l*mb*sin(alpha)*sin(beta), ...    ] [ ddbeta  ]   [ Qbeta  ]

    In code this is solved as:

        ddq = np.linalg.solve(M, forcing)

The function solves for accelerations (ddq) and returns:
    dx/dt = [dr, dphi, dalpha, dbeta, ddr, ddphi, ddalpha, ddbeta]

State-space output as a column vector:

    x_dot =
        [ dr       ]
        [ dphi     ]
        [ dalpha   ]
        [ dbeta    ]
        [ ddr      ]
        [ ddphi    ]
        [ ddalpha  ]
        [ ddbeta   ]

This function is suitable for numerical integration (e.g., scipy.integrate.solve_ivp)
and control system design.
"""


def crane_dynamics(t, x, u, disturbance, params, M_func, forcing_func):
    # Split the state vector into generalized coordinates q and velocities dq:
    # x = [q, dq] = [r, phi, alpha, beta, dr, dphi, dalpha, dbeta].
    r, phi, alpha, beta, dr, dphi, dalpha, dbeta = x

    # Control input vector:
    # u = [F_r, M_phi].
    F_r, M_phi = u

    # Disturbance force vector resolved in the crane basis:
    # disturbance = [Fwr, Fwt, Fwz].
    Fwr, Fwt, Fwz = disturbance

    mv = params["m_v"]
    mb = params["m_b"]
    l = params["l"]
    h = params["h"]
    g = params["g"]
    Jphi = params["J_phi"]

    d_r, d_phi, d_alpha, d_beta = params["d_r"], params["d_phi"], params["d_alpha"], params["d_beta"]

    # M is the 4x4 mass/inertia matrix multiplying the acceleration vector:
    #
    #     M * [ddr, ddphi, ddalpha, ddbeta]^T = forcing
    #
    # M_func is generated from the symbolic model in symbolic_model.py.
    M = M_func(
        r, phi, alpha, beta,
        dr, dphi, dalpha, dbeta,
        F_r, M_phi,
        Fwr, Fwt, Fwz,
        mv, mb, l, h, g, Jphi,
        d_r, d_phi, d_alpha, d_beta
    )

    # forcing is the 4x1 generalized force vector:
    #
    #     forcing = [Qr, Qphi, Qalpha, Qbeta]^T
    #
    # It already includes actuation, gravity, damping, Coriolis
    # terms, and external disturbance contributions from the symbolic model.
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

    # Solve the matrix equation M * ddq = forcing for the unknown
    # generalized accelerations ddq.
    ddq = np.linalg.solve(M, forcing)

    ddr, ddphi, ddalpha, ddbeta = ddq

    # Convert from the second-order mechanical model to first-order
    # state-space form for numerical integration:
    #
    #     x_dot = [dq, ddq].
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
