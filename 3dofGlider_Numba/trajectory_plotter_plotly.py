import numpy as np
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from helper_functions import *

def plot_states_plotly(times, states_ecef: np.array, ecef_target: np.array, html_file="glider.html", fontsize=16):    
  
    figure_width = 1200
    figure_height = 600
    
    lat, lon, alt = batch_ecef_to_geodetic(states_ecef[:, :3])
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0]); R=6378137.0
    x     = (np.radians(lon)-lon0)*np.cos(lat0)*R
    y     = (np.radians(lat)-lat0)*R

    
    lat_targ, lon_targ, alt_targ = batch_ecef_to_geodetic(ecef_target)
    # LLA vs Time
    fig1 = make_subplots(rows=3, cols=1, shared_xaxes='all',
                         subplot_titles=("Latitude (°)","Longitude (°)","Altitude (m)"),
                         vertical_spacing=0.02)
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*lat,      mode='lines', line=dict(width=4)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*lat_targ, mode='lines', line=dict(width=4)), row=1, col=1)
    
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*lon,      mode='lines', line=dict(width=4)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*lon_targ, mode='lines', line=dict(width=4)), row=2, col=1)
    
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*alt,      mode='lines', line=dict(width=4)), row=3, col=1)
    fig1.add_trace(go.Scatter(x=times, y=180/3.14*alt_targ, mode='lines', line=dict(width=4)), row=3, col=1)
    
    fig1.update_layout(width=figure_width, height=figure_height, title="LLA vs Time",
                       font=dict(size=fontsize))
    fig1.update_xaxes(title="Time (s)", row=3, col=1)

    # 2D ground track
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=4)))
    fig2.update_layout(title="Flat‐Earth Track", xaxis_title="East (m)",
                       yaxis_title="North (m)", font=dict(size=fontsize),
                       width=figure_width, height=figure_height)
    fig2.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # spped
    speed=np.linalg.vector_norm(states_ecef[:, 3:6], axis=1)
    print(times)
    print("---------------------")
    print(speed)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=speed, mode='lines', line=dict(width=4)))
    fig3.update_layout(title="Speed", xaxis_title="Time (s)",
                       yaxis_title="Speed (m/s)", font=dict(size=fontsize),
                       width=figure_width, height=figure_height)

    # 3D trajectory
    fig4 = go.Figure(go.Scatter3d(x=x, y=y, z=alt, mode='lines', line=dict(width=4)))
    fig4.update_layout(title="3D Flat‐Earth Trajectory",
                       scene=dict(xaxis_title='East (m)',
                                  yaxis_title='North (m)',
                                  zaxis_title='Altitude (m)'),
                       font=dict(size=fontsize),
                       width=1500, height=1500)

    # --- 2) Convert each to an HTML <div> snippet ---
    div1 = pyo.plot(fig1, include_plotlyjs='cdn', output_type='div')
    div2 = pyo.plot(fig2, include_plotlyjs=False, output_type='div')
    div3 = pyo.plot(fig3, include_plotlyjs=False, output_type='div')
    div4 = pyo.plot(fig4, include_plotlyjs=False, output_type='div')

    # --- 3) Combine & write a single HTML file (overwrites existing) ---
    html = f"""<!DOCTYPE html>
        <html><head><meta charset="utf-8"></head><body>
        {div1}
        {div2}
        {div3}
        {div4}
        </body></html>"""

    with open(html_file, 'w') as f:
        f.write(html)

    print(f"Wrote updated trajectory to {html_file}. Open it in your browser and refresh to see the new plots.")
