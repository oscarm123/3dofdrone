import numpy as np
from trajectory_plotter_plotly  import plot_states_plotly
import concurrent.futures
from datetime import datetime
from drag_force_bodyframe import drag_force_bodyframe
from eci_to_geodetic import eci_to_geodetic
from helper_functions import *



# Glider model
class Glider3DOF:
    def __init__(self, mass, ecef_target):
        self.mass = mass
        self.target_ecef = ecef_target

    def aerodynamic_forces(self, r_eci, v_eci, t):
        lat, lon, alt = eci_to_geodetic(r_eci, t)
        
        # altitude-based density (optional)
        rho0 = 1.225           # kg/m^3 at sea level
        scale_height = 8000.0  # m
        rho = rho0 * np.exp(-alt / scale_height)
        
        phi, theta, psi = flight_path_euler(r_eci, v_eci, t)

        # compute drag
        # Body 3-2-1 sequence: first yaw (ψ), then pitch (θ), then roll (φ)
        R_eci2body = rot_x(phi) @ rot_y(theta) @ rot_z(psi)
        F_drag_body = drag_force_bodyframe(cone_radius=0.5,
                                           air_density=rho, 
                                           v_eci=v_eci,
                                           R_eci2body=R_eci2body,
                                           C_D=0.8)
            
        F_body = np.zeros(3)
        euler = (phi, theta, psi)
        return F_body, euler

    def dynamics(self, t, state):
        r_eci = state[0:3]
        v_eci = state[3:6]


        r_ecef = eci_to_ecef(r_eci, t)
        v_ecef = eci_to_ecef_vel(v_eci, r_eci, t)
        # Aerodynamic forces in body frame + orientation
        F_body, euler = self.aerodynamic_forces(r_eci, v_eci, t)
        # Convert to NED
        F_ned = body_to_ned(F_body, euler)
        # Get lat, lon for NED->ECEF
        lat, lon = get_lat_lon_from_eci(r_eci, t)
        F_ecef = ned_to_ecef(F_ned, lat, lon)
        # To ECI
        F_eci = ecef_to_eci(F_ecef, t)
        # Gravity in ECI
        a_gravity = -mu * r_eci / np.linalg.norm(r_eci)**3


        a_command = guidance_command(r_ecef, self.target_ecef, v_ecef, v_ecef*0) 
        # Total acceleration
        a_eci = a_gravity + F_eci / self.mass + a_command
        return np.hstack((v_eci, a_eci))

# Integrator (RK4)
def rk4_step(fun, t, y, dt):
    k1 = fun(t, y)
    k2 = fun(t + dt/2, y + dt/2 * k1)
    k3 = fun(t + dt/2, y + dt/2 * k2)
    k4 = fun(t + dt,   y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def propagate_eci(state0, t0, tf, dt, glider):
    times = np.arange(t0, tf + dt, dt)
    states = np.zeros((len(times), len(state0)))
    state = state0.copy()

    for i, t in enumerate(times):
        # record state
        states[i] = state

        # compute current altitude
        r_eci = state[0:3]
        r_ecef = eci_to_ecef(r_eci, t)
        alt = np.linalg.norm(r_ecef) - R_EARTH

        if alt < 0:
            print(f"Simulation ended: glider hit the ground at t = {t:.2f} s (alt = {alt:.1f} m)")
            # truncate arrays up to and including this step
            return times[:i+1], states[:i+1]

        # advance state
        state = rk4_step(lambda tt, yy: glider.dynamics(tt, yy), t, state, dt)

    return times, states

# Example usage
if __name__ == "__main__":
    
    pi = 3.14
    dt = 1e-3
    t0 = 0
    tf = 60
    
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
    
    tick = datetime.now()
    # Create glider model and propagate for 60 seconds
    glider = Glider3DOF(mass=100.0, ecef_target = ecef_target)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # times, states = propagate_eci(state0, 0.0, 60.0, 1e-3, glider)
        future = executor.submit(
            propagate_eci,
            state0,      # initial state
            t0,         # t0
            tf,        # tf
            dt,        # dt
            glider       # your Glider3DOF instance
        )
        times, states = future.result()
    tock = datetime.now()
    diff = tock - tick
    # Print final ECI position
    print("Final ECI Position (m):", states[-1, :3])
    print( "Simulation took "+ str(diff.total_seconds())+" seconds" )
    
    # states is an (N,6) array: [x,y,z,vx,vy,vz]
    r_eci = states[:,  :3]   # shape (N,3)
    v_eci = states[:, 3: ]  # shape (N,3)
    r_ecef = np.vstack([eci_to_ecef(r, t) for r, t in zip(r_eci, times)])
    v_ecef = np.vstack([eci_to_ecef_vel(r, v, t)
                    for r, v, t in zip(r_eci, v_eci, times)])
    states_ecef = np.hstack((r_ecef, v_ecef))
    
    plot_states_plotly(times, states_ecef, html_file="glider.html", fontsize=13)