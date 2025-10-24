import numpy as np
import jax.numpy as jnp
from integrator_gpu import propagate_eci_gpu
from datetime import datetime
from trajectory_plotter_plotly  import plot_states_plotly
from help_func import *
from jax import device_get

def main():
    tick = datetime.now()
    # 1) define time grid on CPU
    t0, tf, dt = 0.0, 60.0, 1e-2 
     
    n     = int((tf - t0)/dt) + 1         # pure Python int
    times = jnp.linspace(t0, tf, n)       # shape (n,), fixed at compile time

    pi = 3.14
    
    ecef_target = geodetic_to_ecef(lat=pi/180*0, lon=pi/180*0.01*15, h=30e3)
    
    # Initial geodetic conditions (45°N,  0°E, 1000 m)
    lat0, lon0, h0 = np.deg2rad(0), np.deg2rad(0), 50e3
    r_ecef0 = geodetic_to_ecef(lat0, lon0, h0)
    r_eci0 = ecef_to_eci(r_ecef0, 0.0)

    # Initial NED velocity (northward 400 m/s, eastward 0, down 0)
    v_ned0 = np.array([400.0, 0.0, 0.0])
    v_ecef0 = ned_to_ecef(v_ned0, lat0, lon0)
    v_eci0 = ecef_to_eci(v_ecef0, 0.0)

    # State vector
    state0 = np.hstack((r_eci0, v_eci0))
    
    # 2) initial ECI state → GPU
    state0     = jnp.array(state0)

    # 3) call JIT’d integrator
    states = propagate_eci_gpu(state0, times)
    print(type(states))

    # 4) bring back for plotting/analysis
    states_cpu = np.array(states)
    print("Done! states shape:", states_cpu.shape)
    tock = datetime.now()
    diff = tock - tick
    print( "Simulation took "+ str(diff.total_seconds())+" seconds" )
    
    # split pos/vel
    r_eci = states_cpu[:, :3]   # shape (N,3)
    v_eci = states_cpu[:, 3:6]  # shape (N,3)
    r_ecef = np.vstack([eci_to_ecef(r, t) for r, t in zip(r_eci, times)])
    v_ecef = np.vstack([eci_to_ecef_vel(r, v, t)
                    for r, v, t in zip(r_eci, v_eci, times)])
    states_ecef = np.hstack((r_ecef, v_ecef))
    
    plot_states_plotly(times, states_ecef, html_file="glider.html", fontsize=13)

if __name__ == "__main__":
    main()
