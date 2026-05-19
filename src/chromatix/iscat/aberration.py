"""Gibson-Lanni index-mismatch aberration phase."""

import jax.numpy as jnp
from chromatix.core import VectorField


def index_mismatch_phase(
    field: VectorField,
    na: float,
    n_oil: float,
    n_medium: float,
    wavelength: float,
    z_focal: float,
    t_oil_ideal: float,
    z_particle: float,
):
    """Compute Gibson-Lanni aberration phase for index mismatch.

    Returns phase array of shape (y, x, 1) in radians.
    """
    k = 2.0 * jnp.pi / wavelength

    fy = field.f_grid[..., 0, 0]
    fx = field.f_grid[..., 0, 1]

    f_na = na / wavelength
    rho2 = (fy ** 2 + fx ** 2) / (f_na ** 2 + 1e-30)
    rho2 = jnp.clip(rho2, 0.0, 1.0)

    sin2_na = na ** 2 * rho2  # sin^2(theta) at each pupil point

    cos_med = jnp.sqrt(jnp.maximum(n_medium ** 2 - sin2_na, 0.0)) / n_medium

    # OPD from axial shift between z_particle and z_focal
    opd = (z_particle - z_focal) * cos_med

    phase = k * n_medium * opd
    return phase[..., jnp.newaxis]  # (y, x, 1)
