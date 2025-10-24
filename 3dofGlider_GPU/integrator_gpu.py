import jax
import jax.numpy as jnp
from dynamics import accel

@jax.jit
def propagate_eci_gpu(
    state0: jnp.ndarray,
    times: jnp.ndarray
) -> jnp.ndarray:
    """
    JIT‐compiled RK4 integrator on GPU.
    Inputs:
      state0: (6,) initial [x,y,z,vx,vy,vz]
      times:  (N,) time stamps
    Returns:
      states: (N,6) array of [x,y,z,vx,vy,vz] at each time
    """
    dt = times[1] - times[0]

    def rk4_step(state, t):
        r, v = state[:3], state[3:]

        # k1
        a1       = accel(r, v, t)
        k1_r, k1_v = v, a1

        # k2
        r2       = r + 0.5 * (times[1] - times[0]) * k1_r
        v2       = v + 0.5 * (times[1] - times[0]) * k1_v
        a2       = accel(r2, v2, t + 0.5*(times[1] - times[0]))
        k2_r, k2_v = v2, a2

        # k3
        r3       = r + 0.5 * (times[1] - times[0]) * k2_r
        v3       = v + 0.5 * (times[1] - times[0]) * k2_v
        a3       = accel(r3, v3, t + 0.5*(times[1] - times[0]))
        k3_r, k3_v = v3, a3

        # k4
        r4       = r + (times[1] - times[0]) * k3_r
        v4       = v + (times[1] - times[0]) * k3_v
        a4       = accel(r4, v4, t + (times[1] - times[0]))
        k4_r, k4_v = v4, a4

        dt = times[1] - times[0]
        r_next = r + (dt/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_next = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        return jnp.concatenate([r_next, v_next])

    # this step function returns (new_carry, y_i=new_carry)
    def scan_step(carry, t):
        next_state = rk4_step(carry, t)
        return next_state, next_state

    # run the scan: carry_final is dropped, `states` is shape (N,6)
    _, states = jax.lax.scan(scan_step, state0, times)
    return states
