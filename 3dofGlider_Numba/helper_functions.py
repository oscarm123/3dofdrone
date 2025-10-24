import numpy as np
from numba import njit
from numba.extending import register_jitable

# Earth constants
mu = 3.986004418e14  # Earth's gravitational parameter, m^3/s^2
Omega_e = 7.2921150e-5  # Earth's rotation rate, rad/s
R_EARTH = 6378137.0  # mean Earth radius [m]
a = 6378137.0          # WGS-84 semi-major axis (m)
f = 1 / 298.257223563  # WGS-84 flattening

e2 = f * (2 - f)       # square of eccentricity


# Rotation matrices
@njit
def rot_x(phi):
    """
    Rotation about the x-axis by angle phi.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    M = np.empty((3, 3), dtype=np.float64)
    # first row
    M[0, 0] = 1.0; M[0, 1] = 0.0; M[0, 2] = 0.0
    # second row
    M[1, 0] = 0.0; M[1, 1] = c;   M[1, 2] = -s
    # third row
    M[2, 0] = 0.0; M[2, 1] = s;   M[2, 2] =  c
    return M

@njit
def rot_y(theta):
    """
    Rotation about the y-axis by angle theta.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    M = np.empty((3, 3), dtype=np.float64)
    # first row
    M[0, 0] =  c;   M[0, 1] = 0.0; M[0, 2] = s
    # second row
    M[1, 0] = 0.0;  M[1, 1] = 1.0; M[1, 2] = 0.0
    # third row
    M[2, 0] = -s;   M[2, 1] = 0.0; M[2, 2] = c
    return M

@njit
def rot_z(psi):
    """
    Rotation about the z-axis by angle psi.
    """
    c = np.cos(psi)
    s = np.sin(psi)
    M = np.empty((3, 3), dtype=np.float64)
    # first row
    M[0, 0] =  c;  M[0, 1] = -s;  M[0, 2] = 0.0
    # second row
    M[1, 0] =  s;  M[1, 1] =  c;  M[1, 2] = 0.0
    # third row
    M[2, 0] = 0.0; M[2, 1] = 0.0; M[2, 2] = 1.0
    return M


# Coordinate transforms
@njit
def ecef_to_eci(vec_ecef, t):
    """Rotate from ECEF to ECI at time t (s)."""
    return rot_z(Omega_e * t).dot(vec_ecef)

@njit
def eci_to_ecef(vec_eci, t):
    """Rotate from ECI to ECEF at time t (s)."""
    return rot_z(-Omega_e * t).dot(vec_eci)

@njit
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


def ecef_to_eci_vel(v_ecef: np.ndarray,
                    r_ecef: np.ndarray,
                    t: float) -> np.ndarray:
    """
    Convert a velocity vector from ECEF back to ECI at time t.

    Args:
        v_ecef: 3-element velocity in ECEF frame [m/s]
        r_ecef: 3-element position in ECEF frame [m]
        t:       Time since epoch (t=0) in seconds

    Returns:
        3-element velocity in ECI frame [m/s]
    """
    # Earth’s spin vector (rad/s about z-axis)
    omega = np.array([0.0, 0.0, Omega_e])

    # 1) put back the frame-rotation (Coriolis) term:
    #    v_rot = v_ecef + ω × r_ecef
    v_rot = v_ecef + np.cross(omega, r_ecef)

    # 2) rotate from the Earth-fixed axes back into inertial axes:
    #    R = rot_z(+Omega_e * t) is the transpose of rot_z(-Omega_e * t)
    return rot_z(Omega_e * t).dot(v_rot)


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

@njit
def eci_to_geodetic(r_eci, t_ut1):
    """
    Convert an ECI position vector to geodetic latitude, longitude, and height.

    Parameters
    ----------
    r_eci : array-like, shape (3,)
        ECI position vector [m].
    t_ut1 : float
        Time since epoch in seconds.

    Returns
    -------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    h   : float
        Height above WGS-84 ellipsoid in meters.
    """
    # 1) ECI -> ECEF rotation about Z-axis
    theta = Omega_e * t_ut1
    c = np.cos(theta)
    s = np.sin(theta)

    x_ec, y_ec, z_ec = r_eci[0], r_eci[1], r_eci[2]
    x = c * x_ec + s * y_ec
    y = -s * x_ec + c * y_ec
    z = z_ec

    # 2) Longitude
    lon = np.arctan2(y, x)

    # 3) Bowring's method for latitude
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))  # initial guess
    for _ in range(5):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
        h_temp = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * (N / (N + h_temp))))

    # 4) Final height
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
    h = p / np.cos(lat) - N

    return lat, lon, h

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


@njit
def guidance_command(ecef_chaser, ecef_target, ecef_v_chaser, ecef_v_target, rgain=0, vgain=0):
    """
    Compute a guidance acceleration command in ECEF frame.

    Parameters
    ----------
    ecef_chaser    : array-like, shape (3,)
    ecef_target    : array-like, shape (3,)
    ecef_v_chaser  : array-like, shape (3,)
    ecef_v_target  : array-like, shape (3,)

    Returns
    -------
    a_command      : ndarray, shape (3,)
        Acceleration command [m/s^2].
    """
    # allocate output vector
    a_command = np.empty(3, dtype=np.float64)

    # velocity difference term and proportional gain term
    a_command[0] = (ecef_v_target[0] - ecef_v_chaser[0]) * vgain + rgain * (ecef_target[0] - ecef_chaser[0])
    a_command[1] = (ecef_v_target[1] - ecef_v_chaser[1]) * vgain + rgain * (ecef_target[1] - ecef_chaser[1])
    a_command[2] = (ecef_v_target[2] - ecef_v_chaser[2]) * vgain + rgain * (ecef_target[2] - ecef_chaser[2]) - 9.81
    # (vtarget - vChaser) + k * (posTarget - posChaser) - aGravity; %Gravity compensated
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

def ecei_to_ecef_states(states_eci, times):
    states_eci = np.asarray(states_eci)
    times      = np.asarray(times)
    assert states_eci.shape[0] == times.shape[0], "Mismatch in number of time stamps vs. states"
    # constants
    omega_e  = 7.2921150e-5   # rad/s

    # states_eci: (N,3) arrays
    r_eci = states_eci[:, :3]
    v_eci = states_eci[:, 3:]

    # 1) build your “angle” array and its sin/cos:
    psi = -omega_e  * times               # shape (N,)
    c, s = np.cos(psi), np.sin(psi)   # each (N,)

    # 2) rotate positions:
    x, y, z = r_eci.T
    r_ecef = np.column_stack((c*x - s*y,
                            s*x + c*y,
                            z))

    # 3) rotate velocities _and_ add the ω×r term:
    vx, vy, vz = v_eci.T

    v_rot = np.column_stack((c*vx - s*vy,
                            s*vx + c*vy,
                            vz))
    temp =  np.column_stack((-omega_e  * r_ecef[:,1],
                                    omega_e  * r_ecef[:,0],
                                    np.zeros_like(z)))

    # ω vector is [0,0,omega_e ], so ω×r = [ -omega_e *y', omega_e *x', 0 ]
    v_ecef = v_rot - temp

    # 4) stitch back together
    states_ecef = np.hstack((r_ecef, v_ecef))
    return states_ecef


def batch_eci_to_ecef(r_eci: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    r_eci : (N,3) array of ECI positions
    times : (N,)   array of time stamps
    returns: (N,3) array of ECEF positions
    """
    psi     = -Omega_e * times             # (N,)
    cos_psi = np.cos(psi)                  # (N,)
    sin_psi = np.sin(psi)                  # (N,)

    x, y, z = r_eci.T                      # each (N,)
    x_e = cos_psi * x - sin_psi * y
    y_e = sin_psi * x + cos_psi * y
    z_e = z                                # unchanged

    return np.column_stack((x_e, y_e, z_e))


def batch_eci_to_ecef_vel(r_eci: np.ndarray,
                          v_eci: np.ndarray,
                          times: np.ndarray) -> np.ndarray:
    """
    r_eci : (N,3) array of ECI positions
    v_eci : (N,3) array of ECI velocities
    times : (N,)   array of time stamps
    returns: (N,3) array of ECEF velocities
    """
    psi     = -Omega_e * times
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    vx, vy, vz = v_eci.T
    v_x = cos_psi * vx - sin_psi * vy
    v_y = sin_psi * vx + cos_psi * vy
    v_z = vz

    # rotate r here to compute omega_cross
    x, y, _ = r_eci.T
    x_e = cos_psi * x - sin_psi * y
    y_e = sin_psi * x + cos_psi * y

    omega_cross = np.column_stack((
        -Omega_e * y_e,
         Omega_e * x_e,
         np.zeros_like(z)
    ))

    return np.column_stack((v_x, v_y, v_z)) + omega_cross


def batch_eci_states_to_ecef(states: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    states : (N,6) array [x,y,z,vx,vy,vz]
    times  : (N,)
    returns : (N,6) array [x_e,y_e,z_e,vx_e,vy_e,vz_e]
    """
    r_eci = states[:, :3]
    v_eci = states[:, 3:]
    r_ecef = batch_eci_to_ecef(r_eci, times)
    v_ecef = batch_eci_to_ecef_vel(r_eci, v_eci, times)
    return np.hstack((r_ecef, v_ecef))


def batch_ecef_to_geodetic(r_ecef: np.ndarray,
                           tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    r_ecef : (N,3) array of ECEF positions
    returns : (lat, lon, alt) each of shape (N,)
    """
    a  = 6378137.0
    f  = 1/298.257223563
    e2 = f * (2 - f)

    x = r_ecef[:, 0]
    y = r_ecef[:, 1]
    z = r_ecef[:, 2]

    lon = np.arctan2(y, x)
    p   = np.hypot(x, y)

    # initial latitude guess
    lat      = np.arctan2(z, p * (1 - e2))
    lat_prev = np.zeros_like(lat)

    # iterate to convergence
    while np.max(np.abs(lat - lat_prev)) > tol:
        lat_prev = lat
        sin_lat  = np.sin(lat_prev)
        N        = a / np.sqrt(1 - e2 * sin_lat**2)
        lat      = np.arctan2(z + e2 * N * sin_lat, p)

    N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt


@njit
def eci_to_ecef_acc(a_eci, v_eci, r_eci, t):
    """
    Convert acceleration from ECI to ECEF frame at time t.
    """
    # rotation angle
    theta = -Omega_e * t
    c = np.cos(theta)
    s = np.sin(theta)

    # rotate position
    x_ec, y_ec, z_ec = r_eci[0], r_eci[1], r_eci[2]
    x_e = c * x_ec - s * y_ec
    y_e = s * x_ec + c * y_ec
    z_e = z_ec

    # rotate velocity
    vx_ec, vy_ec, vz_ec = v_eci[0], v_eci[1], v_eci[2]
    vx_e = c * vx_ec - s * vy_ec
    vy_e = s * vx_ec + c * vy_ec
    vz_e = vz_ec

    # rotate acceleration
    ax_ec, ay_ec, az_ec = a_eci[0], a_eci[1], a_eci[2]
    ax_e = c * ax_ec - s * ay_ec
    ay_e = s * ax_ec + c * ay_ec
    az_e = az_ec

    # Coriolis term: 2 * omega x v_e
    ocv0 = -Omega_e * vy_e
    ocv1 =  Omega_e * vx_e
    cor0 = 2.0 * ocv0
    cor1 = 2.0 * ocv1
    cor2 = 0.0

    # Centrifugal term: omega x (omega x r_e)
    or0 = -Omega_e * y_e
    or1 =  Omega_e * x_e
    cen0 = -Omega_e * or1
    cen1 =  Omega_e * or0
    cen2 = 0.0

    # sum components
    a_e = np.empty(3, dtype=np.float64)
    a_e[0] = ax_e + cor0 + cen0
    a_e[1] = ay_e + cor1 + cen1
    a_e[2] = az_e + cor2 + cen2
    return a_e


@njit
def ecef_to_eci_acc(a_ecef, v_ecef, r_ecef, t):
    """
    Convert acceleration from ECEF back to ECI frame at time t.
    """
    # rotation angle
    theta = Omega_e * t
    c = np.cos(theta)
    s = np.sin(theta)

    # rotate position back
    x_ec, y_ec, z_ec = r_ecef[0], r_ecef[1], r_ecef[2]
    x_i = c * x_ec - s * y_ec
    y_i = s * x_ec + c * y_ec
    z_i = z_ec

    # rotate velocity back
    vx_ec, vy_ec, vz_ec = v_ecef[0], v_ecef[1], v_ecef[2]
    vx_i = c * vx_ec - s * vy_ec
    vy_i = s * vx_ec + c * vy_ec
    vz_i = vz_ec

    # rotate acceleration back
    ax_ec, ay_ec, az_ec = a_ecef[0], a_ecef[1], a_ecef[2]
    ax_r = c * ax_ec - s * ay_ec
    ay_r = s * ax_ec + c * ay_ec
    az_r = az_ec

    # Coriolis: 2 * omega x v_i
    ovi0 = -Omega_e * vy_i
    ovi1 =  Omega_e * vx_i
    cor0 = 2.0 * ovi0
    cor1 = 2.0 * ovi1
    cor2 = 0.0

    # Centrifugal: omega x (omega x r_i)
    ori0 = -Omega_e * y_i
    ori1 =  Omega_e * x_i
    cen0 = -Omega_e * ori1
    cen1 =  Omega_e * ori0
    cen2 = 0.0

    # combine
    a_i = np.empty(3, dtype=np.float64)
    a_i[0] = ax_r - cor0 - cen0
    a_i[1] = ay_r - cor1 - cen1
    a_i[2] = az_r - cor2 - cen2
    return a_i

from numba import njit
import numpy as np

# -----------------------------------------------------------------------------
# ECEF <-> NED
# -----------------------------------------------------------------------------

@njit
def ecef_to_ned_jit(vec_ecef, lat, lon):
    """
    Map a vector in ECEF to NED at given geodetic lat,lon.
    vec_ecef: array-like [x_ecef, y_ecef, z_ecef]
    """
    x, y, z = vec_ecef[0], vec_ecef[1], vec_ecef[2]
    slat = np.sin(lat);  clat = np.cos(lat)
    slon = np.sin(lon);  clon = np.cos(lon)
    # C_e_n (NED→ECEF).  We need its transpose:
    ned0 = -slat*clon * x - slon       * y - clat*clon * z
    ned1 = -slat*slon * x + clon       * y - clat*slon * z
    ned2 =  clat       * x + 0.0       * y - slat       * z
    return np.array((ned0, ned1, ned2), dtype=np.float64)

@njit
def ned_to_ecef_jit(vec_ned, lat, lon):
    """
    Map a vector in NED to ECEF at given geodetic lat,lon.
    vec_ned: array-like [north, east, down]
    """
    nx, ny, nz = vec_ned[0], vec_ned[1], vec_ned[2]
    slat = np.sin(lat);  clat = np.cos(lat)
    slon = np.sin(lon);  clon = np.cos(lon)
    # C_e_n (NED→ECEF)
    x = -slat*clon * nx - slat*slon * ny + clat       * nz
    y = -slon       * nx + clon       * ny + 0.0       * nz
    z = -clat*clon * nx - clat*slon * ny - slat       * nz
    return np.array((x, y, z), dtype=np.float64)

# -----------------------------------------------------------------------------
# BODY <-> NED
# -----------------------------------------------------------------------------

@njit
def body_to_ned_jit(vec_b, phi, theta, psi):
    """
    Map a vector in the BODY frame to NED via 3-2-1 Euler (φ-roll, θ-pitch, ψ-yaw).
    vec_b: array-like [b_x, b_y, b_z]
    """
    bx, by, bz = vec_b[0], vec_b[1], vec_b[2]
    cphi = np.cos(phi);   sphi = np.sin(phi)
    cth  = np.cos(theta); sth  = np.sin(theta)
    cpsi = np.cos(psi);   spsi = np.sin(psi)

    ned0 =   cth*cpsi*bx +   cth*spsi*by -    sth*bz
    ned1 = (sphi*sth*cpsi - cphi*spsi)*bx + \
           (sphi*sth*spsi + cphi*cpsi)*by + \
            sphi*cth*bz
    ned2 = (cphi*sth*cpsi + sphi*spsi)*bx + \
           (cphi*sth*spsi - sphi*cpsi)*by + \
            cphi*cth*bz

    return np.array((ned0, ned1, ned2), dtype=np.float64)

@njit
def ned_to_body_jit(vec_ned, phi, theta, psi):
    """
    Map a vector in NED back to BODY frame (inverse of body_to_ned_jit).
    """
    nx, ny, nz = vec_ned[0], vec_ned[1], vec_ned[2]
    cphi = np.cos(phi);   sphi = np.sin(phi)
    cth  = np.cos(theta); sth  = np.sin(theta)
    cpsi = np.cos(psi);   spsi = np.sin(psi)

    bx =   cth*cpsi*nx +   (sphi*sth*cpsi - cphi*spsi)*ny +   (cphi*sth*cpsi + sphi*spsi)*nz
    by =   cth*spsi*nx +   (sphi*sth*spsi + cphi*cpsi)*ny +   (cphi*sth*spsi - sphi*cpsi)*nz
    bz =  -sth*    nx +     sphi*cth*      ny +               cphi*cth*      nz

    return np.array((bx, by, bz), dtype=np.float64)

# -----------------------------------------------------------------------------
# BODY <-> ECEF (via NED)
# -----------------------------------------------------------------------------

@njit
def body_to_ecef_jit(vec_b, phi, theta, psi, lat, lon):
    """
    BODY → NED → ECEF
    """
    ned = body_to_ned_jit(vec_b, phi, theta, psi)
    return ned_to_ecef_jit(ned, lat, lon)

@njit
def ecef_to_body_jit(vec_ecef, phi, theta, psi, lat, lon):
    """
    ECEF → NED → BODY
    """
    ned = ecef_to_ned_jit(vec_ecef, lat, lon)
    return ned_to_body_jit(ned, phi, theta, psi)


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF (x,y,z) to geodetic coordinates (lat, lon, alt).
    
    Inputs:
      x, y, z : float or array_like
        ECEF coordinates in meters.
    Returns:
      lat, lon, alt :
        lat, lon in radians, altitude in meters above the WGS-84 ellipsoid.
    """
    # WGS-84 ellipsoid constants
    a  = 6378137.0                   # semi-major axis, meters
    f  = 1/298.257223563            # flattening
    b  = a * (1 - f)                # semi-minor axis
    e2 = f * (2 - f)                # first eccentricity squared
    ep2 = (a**2 - b**2) / b**2      # second eccentricity squared

    # Longitude
    lon = np.arctan2(y, x)

    # Distance from Z-axis
    p = np.hypot(x, y)

    # Bowring’s formula for latitude
    theta = np.arctan2(z * a, p * b)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + ep2 * b * st**3,
                     p - e2 * a * ct**3)

    # Radius of curvature in prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    # Altitude above ellipsoid
    alt = p / np.cos(lat) - N

    return lat, lon, alt