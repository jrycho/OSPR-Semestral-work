import numpy as np

def basis_vectors(phi):
    """
    Lokální báze spojená s ramenem
    """
    e_r = np.array([np.cos(phi), np.sin(phi), 0.0])
    e_t = np.array([-np.sin(phi), np.cos(phi), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])

    return e_r, e_t, e_z