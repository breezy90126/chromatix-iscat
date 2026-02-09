from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from chromatix.field import VectorField

from .field_utils import _split_xy, _with_xy


@jax.jit
def apply_fresnel_reference(
    field: VectorField, theta: Array, n1: float, n2: float
) -> VectorField:
    """Apply Fresnel reflection coefficients as a Jones matrix."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    rs = (n1 * cos_theta - n2 * jnp.sqrt(1 - (n1 / n2 * sin_theta) ** 2)) / (
        n1 * cos_theta + n2 * jnp.sqrt(1 - (n1 / n2 * sin_theta) ** 2)
    )
    rp = (n2 * cos_theta - n1 * jnp.sqrt(1 - (n1 / n2 * sin_theta) ** 2)) / (
        n2 * cos_theta + n1 * jnp.sqrt(1 - (n1 / n2 * sin_theta) ** 2)
    )

    _, ey, ex = _split_xy(field)
    ex_ref = rs * ex
    ey_ref = rp * ey
    return _with_xy(field, ex_ref, ey_ref)
