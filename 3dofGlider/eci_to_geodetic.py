import numpy as np
from numba import njit

# Earth constants
a = 6378137.0                 # WGS-84 equatorial radius (m)
f = 1.0 / 298.257223563       # WGS-84 flattening
e2 = f * (2 - f)              # square of eccentricity
Ω_e = 7.2921150e-5            # Earth rotation rate (rad/s)

@njit
def eci_to_geodetic(r_eci: np.ndarray,
                    t_ut1: float
                   ) -> tuple[float, float, float]:
    """
    Convert ECI position to geodetic latitude, longitude, and altitude.

    Parameters
    ----------
    r_eci : ndarray, shape (3,)
        Position vector in ECI frame (m).
    t_ut1 : float
        Seconds since ECI epoch (UTC−based), used to rotate into ECEF.

    Returns
    -------
    lat : float
        Geodetic latitude (rad).
    lon : float
        Longitude (rad).
    h   : float
        Height above WGS-84 ellipsoid (m).
    """
    # 1) Rotate ECI → ECEF
    θ = Ω_e * t_ut1
    c, s = np.cos(θ), np.sin(θ)
    R_eci2ecef = np.array([[ c,  s, 0],
                           [-s,  c, 0],
                           [ 0,  0, 1]])
    x, y, z = R_eci2ecef @ r_eci

    # 2) Longitude
    lon = np.arctan2(y, x)

    # 3) Iterative latitude solution (Bowring’s method)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))  # initial guess
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * (N / (N + h))))

    # 4) Final height
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N

    return lat, lon, h