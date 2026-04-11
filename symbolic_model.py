import sympy as sp

def build_crane_model():
    t = sp.symbols('t', real=True)

    # zobecnÄ›nĂ© souĹ™adnice
    r = sp.Function('r')(t)
    phi = sp.Function('phi')(t)
    alpha = sp.Function('alpha')(t)
    beta = sp.Function('beta')(t)

    q = sp.Matrix([r, phi, alpha, beta])
    dq = q.diff(t)
    ddq = dq.diff(t)

    dr, dphi, dalpha, dbeta = dq

    # bĂˇze
    e_r = sp.Matrix([sp.cos(phi), sp.sin(phi), 0])
    e_t = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
    e_z = sp.Matrix([0, 0, 1])

    # parametry
    mv, mb, l, h, g, Jphi = sp.symbols('mv mb l h g Jphi', positive=True, real=True)

    # tlumenĂ­
    d_r, d_phi, d_alpha, d_beta = sp.symbols(
        'd_r d_phi d_alpha d_beta', nonnegative=True, real=True
    )

    # vstupy
    F_r, M_phi = sp.symbols('F_r M_phi', real=True)

    # poruchovĂ© veliÄŤiny vÄ›tru
    Fwr, Fwt, Fwz = sp.symbols('Fwr Fwt Fwz', real=True)
    F_dist = Fwr * e_r + Fwt * e_t + Fwz * e_z



    # poloha vozĂ­ku
    p_v = r * e_r + h * e_z

    # poloha bĹ™emene
    s = l * (
        sp.sin(alpha) * sp.cos(beta) * e_r
        + sp.sin(beta) * e_t
        - sp.cos(alpha) * sp.cos(beta) * e_z
    )
    p_b = p_v + s

    # rychlosti
    v_v = p_v.diff(t)
    v_b = p_b.diff(t)

    # bez expand
    vv2 = v_v.dot(v_v)
    vb2 = v_b.dot(v_b)

    # energie
    T_v = sp.Rational(1, 2) * mv * vv2
    T_b = sp.Rational(1, 2) * mb * vb2
    T_r = sp.Rational(1, 2) * Jphi * dphi**2

    T = T_v + T_b + T_r
    V = mb * g * p_b[2]
    L = T - V

    # zobecnÄ›nĂ© sĂ­ly
    Q_act = sp.Matrix([F_r, M_phi, 0, 0])

    # tlumĂ­cĂ­ tĹ™ecĂ­ sĂ­ly
    Q_damp = sp.Matrix([
        -d_r * dr,
        -d_phi * dphi,
        -d_alpha * dalpha,
        -d_beta * dbeta
    ])

    # poruchovĂ© sĂ­ly
    Q_dist = sp.Matrix([
    F_dist.dot(p_b.diff(r)),
    F_dist.dot(p_b.diff(phi)),
    F_dist.dot(p_b.diff(alpha)),
    F_dist.dot(p_b.diff(beta))
])
    # celkovĂ© zobecnÄ›nĂ© sĂ­ly
    Q = Q_act + Q_damp + Q_dist

    # Lagrangeovy rovnice
    lag_eqs = sp.Matrix([
        sp.diff(sp.diff(L, dq[i]), t) - sp.diff(L, q[i]) - Q[i]
        for i in range(4)
    ])

    # matice u zrychlenĂ­
    M = lag_eqs.jacobian(ddq)

    # vĹˇe ostatnĂ­
    forcing = -lag_eqs.subs({
        ddq[0]: 0,
        ddq[1]: 0,
        ddq[2]: 0,
        ddq[3]: 0
    })

    # obyÄŤejnĂ© symboly pro lambdify
    r_s, phi_s, alpha_s, beta_s = sp.symbols('r phi alpha beta', real=True)
    dr_s, dphi_s, dalpha_s, dbeta_s = sp.symbols('dr dphi dalpha dbeta', real=True)
    Fwr_s, Fwt_s, Fwz_s = sp.symbols('Fwr Fwt Fwz', real=True)
    d_r_s, d_phi_s, d_alpha_s, d_beta_s = sp.symbols('d_r d_phi d_alpha d_beta', real=True, nonnegative=True)

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
