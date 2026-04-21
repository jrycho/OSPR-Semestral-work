import sympy as sp


def build_crane_model():
    t = sp.symbols('t', real=True)

    # Generalized coordinates: trolley radius, crane rotation,
    # radial load swing, and tangential load swing.
    r = sp.Function('r')(t)
    phi = sp.Function('phi')(t)
    alpha = sp.Function('alpha')(t)
    beta = sp.Function('beta')(t)

    # First and second time derivatives of the generalized coordinates.
    q = sp.Matrix([r, phi, alpha, beta])
    dq = q.diff(t)
    ddq = dq.diff(t)

    dr, dphi, dalpha, dbeta = dq

    # Cylindrical basis fixed to the crane rotation angle phi.
    e_r = sp.Matrix([sp.cos(phi), sp.sin(phi), 0])
    e_t = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
    e_z = sp.Matrix([0, 0, 1])

    # Physical parameters.
    mv, mb, l, h, g, Jphi = sp.symbols('mv mb l h g Jphi', positive=True, real=True)

    # Linear damping coefficients for each generalized coordinate.
    d_r, d_phi, d_alpha, d_beta = sp.symbols(
        'd_r d_phi d_alpha d_beta', nonnegative=True, real=True
    )

    # Control inputs: radial force on the trolley and tower rotation torque.
    F_r, M_phi = sp.symbols('F_r M_phi', real=True)

    # External disturbance force, resolved in the rotating crane basis.
    Fwr, Fwt, Fwz = sp.symbols('Fwr Fwt Fwz', real=True)
    F_dist = Fwr * e_r + Fwt * e_t + Fwz * e_z

    # Trolley position.
    p_v = r * e_r + h * e_z

    # Load position relative to the trolley plus the trolley position.
    s = l * (
        sp.sin(alpha) * sp.cos(beta) * e_r
        + sp.sin(beta) * e_t
        - sp.cos(alpha) * sp.cos(beta) * e_z
    )
    p_b = p_v + s

    # Trolley and load velocities.
    v_v = p_v.diff(t)
    v_b = p_b.diff(t)

    # Squared speeds used in the kinetic energy.
    vv2 = v_v.dot(v_v)
    vb2 = v_b.dot(v_b)

    # Kinetic energy of the trolley, load, and crane rotation.
    T_v = sp.Rational(1, 2) * mv * vv2
    T_b = sp.Rational(1, 2) * mb * vb2
    T_r = sp.Rational(1, 2) * Jphi * dphi**2

    # Lagrangian: kinetic energy minus gravitational potential energy.
    T = T_v + T_b + T_r
    V = mb * g * p_b[2]
    L = T - V

    # Generalized forces from the control inputs.
    Q_act = sp.Matrix([F_r, M_phi, 0, 0])

    # Viscous damping forces.
    Q_damp = sp.Matrix([
        -d_r * dr,
        -d_phi * dphi,
        -d_alpha * dalpha,
        -d_beta * dbeta
    ])

    # Project the external Cartesian disturbance onto generalized coordinates.
    Q_dist = sp.Matrix([
        F_dist.dot(p_b.diff(r)),
        F_dist.dot(p_b.diff(phi)),
        F_dist.dot(p_b.diff(alpha)),
        F_dist.dot(p_b.diff(beta))
    ])

    # Total generalized forces.
    Q = Q_act + Q_damp + Q_dist

    # Euler-Lagrange equations in implicit form: M(q) * ddq = forcing(q, dq, u).
    lag_eqs = sp.Matrix([
        sp.diff(sp.diff(L, dq[i]), t) - sp.diff(L, q[i]) - Q[i]
        for i in range(4)
    ])

    # Mass matrix: coefficients multiplying the generalized accelerations.
    # Extracted by deriving everything by ddq (not time) rest goe -> 0
    M = lag_eqs.jacobian(ddq)

    # Remaining terms after removing acceleration-dependent parts.
    forcing = -lag_eqs.subs({
        ddq[0]: 0,
        ddq[1]: 0,
        ddq[2]: 0,
        ddq[3]: 0
    })

    # Replace time-dependent SymPy functions with plain symbols for lambdify.
    r_s, phi_s, alpha_s, beta_s = sp.symbols('r phi alpha beta', real=True)
    dr_s, dphi_s, dalpha_s, dbeta_s = sp.symbols('dr dphi dalpha dbeta', real=True)
    Fwr_s, Fwt_s, Fwz_s = sp.symbols('Fwr Fwt Fwz', real=True)
    d_r_s, d_phi_s, d_alpha_s, d_beta_s = sp.symbols(
        'd_r d_phi d_alpha d_beta', real=True, nonnegative=True
    )

    subs_dict = {
        r: r_s,
        phi: phi_s,
        alpha: alpha_s,
        beta: beta_s,
        dq[0]: dr_s,
        dq[1]: dphi_s,
        dq[2]: dalpha_s,
        dq[3]: dbeta_s,
        Fwr: Fwr_s,
        Fwt: Fwt_s,
        Fwz: Fwz_s,
        d_r: d_r_s,
        d_phi: d_phi_s,
        d_alpha: d_alpha_s,
        d_beta: d_beta_s,
    }

    M = M.subs(subs_dict)
    forcing = forcing.subs(subs_dict)

    # Numeric functions used by the simulation/model code.
    # Convert the symbolic expressions into fast numerical functions using lambdify.
    M_func = sp.lambdify(
        (r_s, phi_s, alpha_s, beta_s,
         dr_s, dphi_s, dalpha_s, dbeta_s,
         F_r, M_phi,
         Fwr_s, Fwt_s, Fwz_s,
         mv, mb, l, h, g, Jphi,
         d_r_s, d_phi_s, d_alpha_s, d_beta_s),
        M,
        modules="numpy"
    )

    forcing_func = sp.lambdify(
        (r_s, phi_s, alpha_s, beta_s,
         dr_s, dphi_s, dalpha_s, dbeta_s,
         F_r, M_phi,
         Fwr_s, Fwt_s, Fwz_s,
         mv, mb, l, h, g, Jphi,
         d_r_s, d_phi_s, d_alpha_s, d_beta_s),
        forcing,
        modules="numpy"
    )

    return M_func, forcing_func
