import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Earth rotation rate (rad/s)
Omega_e = 7.2921150e-5  

def rot_z(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def eci_to_ecef(vec_eci: np.ndarray, t: float) -> np.ndarray:
    return rot_z(-Omega_e * t).dot(vec_eci)

def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = r_ecef
    a = 6378137.0
    f = 1/298.257223563
    e2 = f * (2 - f)
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    lat_prev = 0.0
    while abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat_prev)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat_prev), p)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return lat, lon, alt

def plot_lla_and_flat_plotly(times: np.ndarray, states: np.ndarray) -> None:
    """
    Plot LLA vs time, 2D flat-earth track, and 3D flat-earth trajectory using Plotly.
    Args:
        times:   1D array of time stamps [s]
        states:  Nx6 array, rows = [r_eci_x, r_eci_y, r_eci_z, ...]
    """
    
    fontsize = 20
    # --- compute LLA arrays ---
    lats, lons, alts = [], [], []
    for t, st in zip(times, states):
        r_ecef = eci_to_ecef(st[:3], t)
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        lats.append(lat); lons.append(lon); alts.append(alt)
    lats = np.degrees(lats)
    lons = np.degrees(lons)
    alts = np.array(alts)

    # --- flat-earth projection ---
    lat0, lon0 = np.radians(lats[0]), np.radians(lons[0])
    R = 6378137.0
    x = (np.radians(lons) - lon0) * np.cos(lat0) * R
    y = (np.radians(lats) - lat0) * R

    # --- LLA vs Time subplots ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Latitude (°)", "Longitude (°)", "Altitude (m)"))
    fig.add_trace(go.Scatter(x=times, y=lats, mode='lines', line=dict(width=5), name='Latitude'), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=lons, mode='lines', line=dict(width=5), name='Longitude'), row=2, col=1)
    fig.add_trace(go.Scatter(x=times, y=alts, mode='lines', line=dict(width=5), name='Altitude'), row=3, col=1)
    fig.update_layout(
        height=800*2, width=700*2,
        title_text="LLA vs Time",
        font=dict(size=fontsize),
        margin=dict(t=50, b=50)
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.show()

    # --- 2D Flat-Earth Ground Track ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=5), name='Ground Track'))
    fig2.update_layout(
        title="Flat-Earth Ground Track",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        font=dict(size=fontsize),
        width=600*2, height=700*2,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig2.show()

    # --- 3D Flat-Earth Trajectory ---
    fig3 = go.Figure(
        data=[go.Scatter3d(
            x=x, y=y, z=alts,
            mode='lines',
            line=dict(width=5),
            name='3D Trajectory'
        )]
    )
    fig3.update_layout(
        title="3D Flat-Earth Trajectory",
        scene=dict(
            xaxis_title='East (m)',
            yaxis_title='North (m)',
            zaxis_title='Altitude (m)',
            aspectmode='auto'
        ),
        font=dict(size=fontsize),
        margin=dict(t=50)
    )
    fig3.show()
