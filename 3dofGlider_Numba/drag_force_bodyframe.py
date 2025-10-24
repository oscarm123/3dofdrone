import numpy as np
from numba import njit

@njit
def drag_force_bodyframe(cone_radius: float,
                         air_density: float,
                         v_eci: np.ndarray,
                         R_eci2body: np.ndarray,
                         C_D: float = 0.8) -> np.ndarray:
    # 1) transform velocity into body frame
    v_body = R_eci2body @ v_eci
    V = np.sqrt(v_body[0]**2 + v_body[1]**2 + v_body[2]**2)
    if V == 0.0:
        return np.zeros(3, dtype=np.float64)

    # 2) frontal area of the cone
    A_ref = np.pi * cone_radius**2

    # 3) scalar drag magnitude
    F_D = 0.5 * air_density * V**2 * C_D * A_ref

    # 4) vector drag, opposite v_body
    return -F_D * (v_body / V)
