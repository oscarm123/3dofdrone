import jax
import jax.numpy as jnp

# Physical constants
mu           = 3.986004418e14    # Earth’s gravitational parameter (m^3/s^2)
cone_radius  = 0.5               # Cone base radius (m)
C_D          = 0.8               # Drag coefficient
rho0         = 1.225             # Sea‐level air density (kg/m^3)
scale_height = 8000.0            # Atmospheric scale height (m)

@jax.jit
def accel(r: jnp.ndarray, v: jnp.ndarray, t: float) -> jnp.ndarray:
    """
    Total acceleration = gravity + drag, JIT‐compiled for GPU.
    r: position (3,)
    v: velocity (3,)
    t: time (scalar)
    """
    # 1) Gravity
    norm_r = jnp.linalg.norm(r)
    a_grav = -mu * r / norm_r**3

    # 2) (Optional) altitude‐dependent density:
    # Here we keep altitude = 0 for simplicity; you can plug in a geodetic altitude
    rho = rho0 * jnp.exp(-0.0 / scale_height)

    # 3) Quadratic drag (body‐axes aligned with v)
    V     = jnp.linalg.norm(v)
    A_ref = jnp.pi * cone_radius**2
    F_D   = 0.5 * rho * V**2 * C_D * A_ref
    a_drag = jnp.where(
        V > 0.0,
        - (F_D / V) * (v / V),
        jnp.zeros_like(v),
    )

    return a_grav + a_drag
