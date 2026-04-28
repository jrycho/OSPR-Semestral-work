import numpy as np

# =========================
# FYZIKÁLNÍ KONSTANTY
# =========================

g = 9.81  # gravitační zrychlení [m/s^2]


# =========================
# PARAMETRY JEŘÁBU
# =========================

# Hmotnosti
m_v = 500.0     # hmotnost vozíku [kg]
m_b = 1000.0    # hmotnost břemene [kg]

# Geometrie
l = 5.0         # délka lana [m]
h = 10.0        # výška závěsu nad zemí [m]

# Setrvačnost ramene
J_phi = 3000.0  # moment setrvačnosti kolem osy otáčení [kg·m^2]

# Tlumení (volitelné, pro realističtější model)
d_r = 50.0      # tlumení v radiálním směru
d_phi = 100.0   # tlumení rotace
d_alpha = 10.0  # tlumení kyvu (α)
d_beta = 10.0   # tlumení kyvu (β)


# =========================
# VĚTRNÉ PORUCHY
# =========================

wind_std_wr = 1000.0   # směrodatná odchylka radiální větrné síly [N]
wind_std_wt = 1000.0   # směrodatná odchylka tangenciální větrné síly [N]
wind_std_wz = 0.0   # směrodatná odchylka vertikální větrné síly [N]


# =========================
# VSTUPY (max hodnoty / limity)
# =========================

F_r_max = 5000.0     # max síla vozíku [N]
M_phi_max = 10000.0  # max moment [Nm]


# =========================
# POČÁTEČNÍ PODMÍNKY
# =========================

# stav: [r, phi, alpha, beta, dr, dphi, dalpha, dbeta]
x0 = np.array([
    4.0,    # r [m]
    0.4,    # phi [rad]
    0.1,    # alpha [rad] (malý kyv)
    0.05,   # beta [rad]
    0.0,    # dr
    0.0,    # dphi
    0.0,    # dalpha
    0.0     # dbeta
])

params = {
    "m_v": m_v,
    "m_b": m_b,
    "l": l,
    "h": h,
    "g": g,
    "J_phi": J_phi,
    "d_r": d_r,
    "d_phi": d_phi,
    "d_alpha": d_alpha,
    "d_beta": d_beta,
    "F_r_max": F_r_max,
    "M_phi_max": M_phi_max,
    "wind_std_wr": wind_std_wr,
    "wind_std_wt": wind_std_wt,
    "wind_std_wz": wind_std_wz,
}