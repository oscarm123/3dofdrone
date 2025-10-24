import numpy as np

def drag_force(cone_radius: float,
               air_density: float,
               velocity_vec: np.ndarray,
               C_D: float = 0.8) -> np.ndarray:
    """
    Compute the aerodynamic drag force on a cone.
    
    Parameters
    ----------
    cone_radius : float
        Base radius of the cone (m).
    air_density : float
        Local air density (kg/m^3).
    velocity_vec : ndarray, shape (3,)
        Velocity vector of the cone in body or inertial frame (m/s).
    C_D : float, optional
        Drag coefficient, default 0.8.
    
    Returns
    -------
    F_drag : ndarray, shape (3,)
        Drag force vector (N), opposite to velocity_vec.
    """
    V = np.linalg.norm(velocity_vec)
    if V == 0:
        return np.zeros(3)
    # reference (frontal) area
    A_ref = np.pi * cone_radius**2
    # magnitude of drag
    F_D_mag = 0.5 * air_density * V**2 * C_D * A_ref
    # force vector: opposite direction of motion
    return -F_D_mag * (velocity_vec / V)
