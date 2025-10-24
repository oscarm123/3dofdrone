import numpy as np
from trajectory_plotter_plotly  import plot_states_plotly
from datetime import datetime
from drag_force_bodyframe import drag_force_bodyframe
from numba import njit
from helper_functions import *


# Glider model
class Glider3DOF:
    def __init__(self, mass, ecef_target):
        self.mass = mass
        self.ecef_target = ecef_target

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
            
        F_body = F_drag_body*0 
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

        phi = euler[0]
        theta = euler[1]
        psi = euler[2]
        
        a_command_ecef = guidance_command(r_ecef, self.ecef_target, v_ecef, v_ecef*0, rgain=0.5, vgain=5)
        a_command_body = ecef_to_body_jit(a_command_ecef, phi, theta, psi, lat, lon)
        maxaccel = 100
        a_command_body[0] = np.clip(a_command_body[0], -maxaccel, maxaccel)
        a_command_body[1] = np.clip(a_command_body[1], -maxaccel, maxaccel)
        a_command_body[2] = np.clip(a_command_body[2], -maxaccel, maxaccel)
        a_command_ecef = body_to_ecef_jit(a_command_body, phi, theta, psi, lat, lon)
        a_command_eci = ecef_to_eci_acc(a_command_ecef, v_ecef, r_ecef, t)
        # Total acceleration
        a_eci = a_gravity + F_eci / self.mass + a_command_eci
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
        distance_target = glider.ecef_target - r_ecef
        if np.linalg.norm(distance_target) <= 20:
            print(f"Simulation ended: glider intercerpted target at t = {t:.2f} s (alt = {alt:.1f} m)")
            return times[:i+1], states[:i+1]

        # advance state
        state = rk4_step(lambda tt, yy: glider.dynamics(tt, yy), t, state, dt)

    return times, states

# --- JIT-compile your RK4 integrator and the full propagate loop ---
@njit
def rk4_step_jit(dyn, t, state, dt, mass, target_ecef):
    k1 = dyn(t, state, mass, target_ecef)
    k2 = dyn(t + dt/2, state + dt/2*k1, mass, target_ecef)
    k3 = dyn(t + dt/2, state + dt/2*k2, mass, target_ecef)
    k4 = dyn(t + dt,   state + dt*k3, mass, target_ecef)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

@njit
def glider_dynamics(t, state, mass, target_ecef):
    # inline the minimal bits of Glider3DOF.dynamics()
    # unpack
    r_eci = state[0:3]
    v_eci = state[3:6]
    
    r_ecef = eci_to_ecef(r_eci, t)
    v_ecef = eci_to_ecef_vel(v_eci, r_eci, t)

    # gravity
    norm_r = np.sqrt(r_eci[0]**2 + r_eci[1]**2 + r_eci[2]**2)
    a_grav = -mu * r_eci / norm_r**3  # make mu a global constant

    # aerodynamic drag only (we skip full NED/ECEF transforms here for brevity)
    lat, lon, alt = eci_to_geodetic(r_eci, t)
    rho = 1.225 * np.exp(-alt / 8000.0)
    # assume body axes aligned with velocity for demo
    # so R_eci2body = identity; you can JIT your own flight_path_euler and rot_mats too
    F_body = drag_force_bodyframe(0.5, rho, v_eci, np.eye(3), 0.8)
    a_drag = F_body / mass
    phi, theta, psi = flight_path_euler(r_eci, v_eci, t)

    # simple guidance term zeroed out
    a_command_ecef = guidance_command(r_ecef, target_ecef, v_ecef, v_ecef)
    a_command_body = ecef_to_body_jit(a_command_ecef, phi, theta, psi, lat, lon)
    a_command_body[0] = 0
    a_command_ecef = body_to_ecef_jit(a_command_body, phi, theta, psi, lat, lon)
    a_command_eci = ecef_to_eci_acc(a_command_ecef, v_ecef, r_ecef, t)
    a_cmd = a_command_eci

    a_total = a_grav + a_drag*0 + a_cmd
    return np.array([v_eci[0], v_eci[1], v_eci[2],
                     a_total[0], a_total[1], a_total[2]])

@njit
def propagate_eci_jit(state0, t0, tf, dt, mass, target_ecef):
    n = int((tf - t0)/dt) + 1
    times = np.linspace(t0, tf, n)
    states = np.zeros((n, 6))
    state = state0.copy()
    for i in range(n):
        states[i] = state
        state = rk4_step_jit(glider_dynamics, times[i], state, dt, mass, target_ecef)
    return times, states


# Example usage
if __name__ == "__main__":
    tick = datetime.now()
    pi = 3.14
    dt = 1e-2
    t0 = 0
    tf = 200
    
    ecef_target = geodetic_to_ecef(lat=pi/180*0, lon=pi/180*0.01*10, h=30e3)
    
    # Initial geodetic conditions (45°N,  0°E, 1000 m)
    lat0, lon0, h0 = np.deg2rad(0), np.deg2rad(0), 50e3
    r_ecef0 = geodetic_to_ecef(lat0, lon0, h0)
    r_eci0 = ecef_to_eci(r_ecef0, 0.0)

    # Initial NED velocity (northward 400 m/s, eastward 0, down 0)
    v_ned0 = np.array([000, 400, 0])
    v_ecef0 = ned_to_ecef(v_ned0, lat0, lon0)
    v_eci0 = ecef_to_eci_vel(v_ecef0,r_ecef0, 0.0)

    # State vector
    state0 = np.hstack((r_eci0, v_eci0))
    
    # Create glider model and propagate for 60 seconds
    glider = Glider3DOF(mass=100.0, ecef_target = ecef_target)
    # times, states = propagate_eci_jit(state0, t0, tf, dt,
    #                                   glider.mass, glider.target_ecef)
    # use the pure-Python propagator that has your stopping logic built in
    times, states = propagate_eci(state0, t0, tf, dt, glider)
    tock = datetime.now()
    diff = tock - tick
    # Print final ECI position
    print("Final ECI Position (m):", states[-1, :3])
    print( "Simulation took "+ str(diff.total_seconds())+" seconds" )
    
    # states is an (N,6) array: [x,y,z,vx,vy,vz]
    tick = datetime.now()
    states_eci = np.array(states)
    states_ecef = ecei_to_ecef_states(states_eci, times)
    print('ECI to ECEF transform took ' + str( (datetime.now() - tick).total_seconds() ) )
    
    ecef_target = np.atleast_2d(ecef_target)
    print(ecef_target.shape)
    print(ecef_to_geodetic(ecef_target[0,0], ecef_target[0,1], ecef_target[0,2]))
    plot_states_plotly(times, states_ecef, ecef_target, html_file="glider.html", fontsize=13)