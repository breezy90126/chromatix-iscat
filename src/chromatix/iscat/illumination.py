from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from chromatix.field import VectorField


@jax.jit
def apply_oblique_illumination(
    field: VectorField,
    theta_inc: float,
    phi: float,
    n_medium: float,
    wavelength: float,
) -> VectorField:
    """Apply oblique illumination phase ramp to a vector field."""
    k0 = 2 * jnp.pi * n_medium / wavelength
    kx = k0 * jnp.sin(theta_inc) * jnp.cos(phi)
    ky = k0 * jnp.sin(theta_inc) * jnp.sin(phi)

    ny, nx = field.spatial_shape
    y = (jnp.arange(ny) - ny // 2) * field._dx[0, 0]
    x = (jnp.arange(nx) - nx // 2) * field._dx[1, 0]
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    tilt_phase = jnp.exp(1j * (kx * xx + ky * yy))
    tilt_phase = tilt_phase.reshape((1,) * (field.ndim - 4) + tilt_phase.shape + (1, 1))
    return field.replace(u=field.u * tilt_phase)
