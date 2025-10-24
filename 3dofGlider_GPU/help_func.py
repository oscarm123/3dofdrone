import numpy as np
# Earth constants
mu = 3.986004418e14  # Earth's gravitational parameter, m^3/s^2
Omega_e = 7.2921150e-5  # Earth's rotation rate, rad/s
R_EARTH = 6378137.0  # mean Earth radius [m]


# Rotation matrices
def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# Coordinate transforms
def ecef_to_eci(vec_ecef, t):
    """Rotate from ECEF to ECI at time t (s)."""
    return rot_z(Omega_e * t).dot(vec_ecef)

def eci_to_ecef(vec_eci, t):
    """Rotate from ECI to ECEF at time t (s)."""
    return rot_z(-Omega_e * t).dot(vec_eci)

def eci_to_ecef_vel(v_eci: np.ndarray, r_eci: np.ndarray, t: float) -> np.ndarray:
    """
    Convert a velocity vector from ECI to ECEF at time t.

    Args:
        v_eci: 3-element velocity in ECI frame [m/s]
        r_eci: 3-element position in ECI frame [m]
        t:     Time since epoch (t=0) in seconds

    Returns:
        3-element velocity in ECEF frame [m/s]
    """
    # 1) rotate the inertial velocity into the Earth-fixed frame
    v_rot = rot_z(-Omega_e * t).dot(v_eci)
    # 2) compute ECEF position for the cross-term
    r_ecef = eci_to_ecef(r_eci, t)
    # 3) subtract the Coriolis term Ω × r_ecef
    omega = np.array([0.0, 0.0, Omega_e])
    return v_rot - np.cross(omega, r_ecef)


def geodetic_to_ecef(lat, lon, h):
    """Convert geodetic lat, lon (rad) and altitude (m) to ECEF."""
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + h) * np.sin(lat)
    return np.array([x, y, z])

def body_to_ned(vec_b, euler):
    """Transform vector from body to NED frame given Euler (roll, pitch, yaw)."""
    phi, theta, psi = euler
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    C_n_b = np.array([
        [ cth*cpsi,                 cth*spsi,                -sth    ],
        [ sphi*sth*cpsi - cphi*spsi, sphi*sth*spsi + cphi*cpsi, sphi*cth ],
        [ cphi*sth*cpsi + sphi*spsi, cphi*sth*spsi - sphi*cpsi, cphi*cth ]
    ])
    return C_n_b.dot(vec_b)

def ned_to_ecef(vec_n, lat, lon):
    """Transform vector from NED to ECEF given geodetic lat, lon."""
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    C_e_n = np.array([
        [-slat*clon, -slat*slon,  clat],
        [-slon,       clon,        0   ],
        [-clat*clon, -clat*slon, -slat]
    ])
    return C_e_n.dot(vec_n)

def get_lat_lon_from_eci(r_eci, t):
    """Estimate geodetic lat, lon from ECI position at time t."""
    r_ecef = eci_to_ecef(r_eci, t)
    x, y, z = r_ecef
    lat = np.arcsin(z / np.linalg.norm(r_ecef))
    lon = np.arctan2(y, x)
    return lat, lon

def guidance_command(ecef_chaser, ecef_target, ecef_v_chaser, ecef_v_target):
    gain = 0.6*2    
    a_command = (ecef_v_target - ecef_v_chaser)*1.2 + gain*(ecef_target - ecef_chaser)
    # a_command[2] = a_command[2] - 9.81
    return a_command

def ecef_to_ned_matrix(lat, lon):
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    # from NED to ECEF:
    C_e_n = np.array([
        [-slat*clon, -slat*slon,  clat],
        [    -slon,      clon,     0 ],
        [-clat*clon, -clat*slon, -slat]
    ])
    # invert for ECEF→NED:
    return C_e_n.T

def flight_path_euler(r_eci, v_eci, t):
    # 1) get lat/lon
    lat, lon = get_lat_lon_from_eci(r_eci, t)
    # 2) velocity in ECEF
    v_ecef   = eci_to_ecef_vel(v_eci, r_eci, t)
    # 3) to NED
    C_n_e    = ecef_to_ned_matrix(lat, lon)
    v_ned    = C_n_e.dot(v_ecef)
    # 4) angles
    north, east, down = v_ned
    speed = np.linalg.norm(v_ned)
    psi   = np.arctan2(east, north)
    theta = np.arcsin(-down / speed)
    phi   = 0.0
    return phi, theta, psi