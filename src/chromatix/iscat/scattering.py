"""Pupil-plane scattering models for iSCAT."""

import jax.numpy as jnp
from chromatix.core import VectorField
from .mie import mie_coefficients, s1_s2, default_n_max


def rayleigh_polarizability(radius: float, m) -> complex:
    """Rayleigh polarizability: 4pi R^3 (m^2-1)/(m^2+2)."""
    m = jnp.asarray(m, dtype=jnp.complex64)
    r = jnp.asarray(radius, dtype=jnp.float32)
    return 4.0 * jnp.pi * r ** 3 * (m ** 2 - 1.0) / (m ** 2 + 2.0)


def _sp_couple(Ex, Ey, S1, S2, fy, fx):
    """Project scattered field into s/p polarisation components."""
    phi = jnp.arctan2(fy, fx)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    E_s = -Ex * sin_phi + Ey * cos_phi
    E_p = Ex * cos_phi + Ey * sin_phi

    E_s_sc = S1 * E_s
    E_p_sc = S2 * E_p

    Ex_sc = -E_s_sc * sin_phi + E_p_sc * cos_phi
    Ey_sc = E_s_sc * cos_phi + E_p_sc * sin_phi
    return Ex_sc, Ey_sc


def apply_mie_pupil(field: VectorField, radius: float, m, n_medium: float, n_max: int = None):
    """Apply Mie scattering in the pupil plane.

    Args:
        field: VectorField, f_grid shape (y, x, 1, 2)
        radius: particle radius in metres
        m: complex relative refractive index (n_particle / n_medium)
        n_medium: refractive index of medium
        n_max: number of Mie terms (default: Wiscombe criterion)

    Returns:
        VectorField with scattered field
    """
    wavelength = field.spectrum.central_wavelength
    k = 2.0 * jnp.pi * n_medium / wavelength

    fy = field.f_grid[..., 0, 0]  # (y, x)
    fx = field.f_grid[..., 0, 1]  # (y, x)

    f_norm = jnp.sqrt(fy ** 2 + fx ** 2)
    sin_theta = wavelength * f_norm
    sin_theta = jnp.clip(sin_theta, 0.0, 1.0)
    theta = jnp.arcsin(sin_theta)

    x_param = k * radius
    if n_max is None:
        n_max = default_n_max(float(x_param))

    a_n, b_n = mie_coefficients(m, x_param, n_max)
    S1, S2 = s1_s2(theta, a_n, b_n)

    Ex = field.u[..., 2]  # (y, x)
    Ey = field.u[..., 1]  # (y, x)

    Ex_sc, Ey_sc = _sp_couple(Ex, Ey, S1, S2, fy, fx)

    new_u = field.u.at[..., 1].set(Ey_sc)
    new_u = new_u.at[..., 2].set(Ex_sc)
    return field.replace(u=new_u)


def apply_rayleigh_pupil(field: VectorField, radius: float, m, n_medium: float):
    """Apply isotropic Rayleigh scattering in the pupil plane."""
    wavelength = field.spectrum.central_wavelength
    k = 2.0 * jnp.pi * n_medium / wavelength
    alpha = rayleigh_polarizability(radius, m)
    S = -1j * k ** 3 * alpha / (4.0 * jnp.pi)
    return field.replace(u=field.u * S)


def shift_pupil(field: VectorField, dy: float, dx_val: float):
    """Apply lateral shift via linear phase ramp in k-space.

    Args:
        field: pupil-plane VectorField
        dy, dx_val: lateral shift in metres

    Returns:
        phase-shifted VectorField
    """
    fy = field.f_grid[..., 0, 0]  # (y, x)
    fx = field.f_grid[..., 0, 1]  # (y, x)
    phase = jnp.exp(2j * jnp.pi * (fy * dy + fx * dx_val))
    return field.replace(u=field.u * phase[..., jnp.newaxis])
