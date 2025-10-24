# trajectory_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Earth rotation rate (rad/s)
Omega_e = 7.2921150e-5  

def rot_z(psi: float) -> np.ndarray:
    """Rotation about the Z-axis by angle psi."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def eci_to_ecef(vec_eci: np.ndarray, t: float) -> np.ndarray:
    """Convert ECI vector to ECEF at time t (seconds)."""
    return rot_z(-Omega_e * t).dot(vec_eci)

def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF vector to geodetic latitude, longitude (radians) and altitude (m).
    Uses an iterative solution for latitude.
    """
    x, y, z = r_ecef
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    lat_prev = 0.0

    # Iteratively refine latitude
    while abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat_prev)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat_prev), p)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return lat, lon, alt

def plot_lla_and_flat(times: np.ndarray, states: np.ndarray) -> None:
    """
    Plot latitude, longitude, altitude vs time and a flat-earth ground track.
    
    Args:
        times: 1D array of time stamps [s]
        states: Nx6 array, each row = [r_eci_x, r_eci_y, r_eci_z, ...]
    """
    base_fontsize = 22
    plt.rcParams.update({
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize + 2,
        'axes.labelsize': base_fontsize,
        'xtick.labelsize': base_fontsize - 2,
        'ytick.labelsize': base_fontsize - 2,
        'legend.fontsize': base_fontsize - 2,
    })
    # Preallocate
    lats, lons, alts = [], [], []

    # Convert each ECI position into LLA
    for t, state in zip(times, states):
        r_eci = state[:3]
        r_ecef = eci_to_ecef(r_eci, t)
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)

    lats = np.array(lats)
    lons = np.array(lons)
    alts = np.array(alts)

    # Flat-earth local tangent plane (meters)
    lat0, lon0 = lats[0], lons[0]
    R = 6378137.0
    x = (lons - lon0) * np.cos(lat0) * R  # East
    y = (lats - lat0) * R                 # North

    #--- Plot LLA vs Time
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(times, np.degrees(lats)); axs[0].set_ylabel('Latitude (°)')
    axs[1].plot(times, np.degrees(lons)); axs[1].set_ylabel('Longitude (°)')
    axs[2].plot(times, alts);             axs[2].set_ylabel('Altitude (m)')
    axs[2].set_xlabel('Time (s)')
    fig.tight_layout()

    #--- Plot Flat-Earth Ground Track
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, '-')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Flat-Earth Ground Track')
    plt.axis('equal')

    # matplotlib.use('gtk4agg')
    plt.show()
