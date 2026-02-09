from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def compute_index_mismatch_phase(
    na: float,
    n_oil: float,
    n_glass: float,
    n_medium: float,
    wavelength: float,
    z_focal: float,
    t_oil_ideal: float,
    z_particle: float,
    fx: Array,
    fy: Array,
) -> Array:
    """Compute multi-layer index mismatch phase for a single particle depth."""
    _ = n_glass
    f = jnp.sqrt(fx**2 + fy**2)
    input_sin = f * wavelength / na
    input_sin = jnp.clip(input_sin, 0, 1 - 1e-6)
    theta = jnp.arcsin(input_sin)

    cos_theta = jnp.cos(theta)
    t_oil = z_particle - z_focal + n_oil * (t_oil_ideal / n_oil - z_particle / n_medium)

    aberr_phase = (
        z_particle * n_medium * (cos_theta - 1)
        + n_oil * (t_oil - t_oil_ideal) * (cos_theta - 1)
        - z_focal * cos_theta
        - 0.5 * jnp.pi
    )

    phase = (2 * jnp.pi / wavelength) * aberr_phase

    tir_mask = input_sin > (n_medium / n_oil)
    evanescent_decay = jnp.where(tir_mask, -jnp.sqrt(input_sin**2 - (n_medium / n_oil) ** 2), 0j)
    phase += evanescent_decay * z_particle
    return phase
