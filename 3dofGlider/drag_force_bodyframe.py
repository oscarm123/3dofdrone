import numpy as np

def drag_force_bodyframe(cone_radius: float,
                         air_density: float,
                         v_eci: np.ndarray,
                         R_eci2body: np.ndarray,
                         C_D: float = 0.8) -> np.ndarray:
    """
    Compute the drag force in the *body* frame for a cone.

    Parameters
    ----------
    cone_radius : float
        Base radius of the cone (m).
    air_density : float
        Local air density (kg/m^3).
    v_eci : ndarray, shape (3,)
        Velocity vector in the ECI frame (m/s).
    R_eci2body : ndarray, shape (3,3)
        Rotation matrix from ECI to body frame.
    C_D : float, optional
        Drag coefficient (default 0.8).

    Returns
    -------
    F_drag_body : ndarray, shape (3,)
        Drag force vector in the body frame (N).
    """
    # 1) transform velocity into body frame
    v_body = R_eci2body @ v_eci
    V = np.linalg.norm(v_body)
    if V == 0.0:
        return np.zeros(3)

    # 2) frontal area of the cone
    A_ref = np.pi * cone_radius**2

    # 3) scalar drag magnitude
    F_D = 0.5 * air_density * V**2 * C_D * A_ref

    # 4) vector drag, opposite v_body
    return -F_D * (v_body / V)