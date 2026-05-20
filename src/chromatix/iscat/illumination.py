"""Illumination and reference field models for iSCAT."""

import jax.numpy as jnp
from chromatix.core import VectorField


def oblique_phase(field: VectorField, theta_inc: float, phi_inc: float, n_medium: float):
    """Apply oblique illumination phase ramp to a field.

    Args:
        field: VectorField with grid shape (y, x, 1, 2)
        theta_inc: polar angle of illumination (radians)
        phi_inc: azimuthal angle of illumination (radians)
        n_medium: refractive index of medium

    Returns:
        phase-modulated VectorField
    """
    wavelength = field.spectrum.central_wavelength
    k = 2.0 * jnp.pi * n_medium / wavelength

    ky = k * jnp.sin(theta_inc) * jnp.sin(phi_inc)
    kx = k * jnp.sin(theta_inc) * jnp.cos(phi_inc)

    y = field.grid[..., 0, 0]  # (y, x)
    x = field.grid[..., 0, 1]  # (y, x)

    phase = jnp.exp(1j * (ky * y + kx * x))
    return field.replace(u=field.u * phase[..., jnp.newaxis])


def fresnel_reference(field: VectorField, n_glass: float, n_medium: float):
    """Fresnel reflection at glass/medium interface as iSCAT reference.

    Computes rs and rp Fresnel coefficients in pupil space and applies
    them with s/p polarisation projection.
    """
    wavelength = field.spectrum.central_wavelength

    fy = field.f_grid[..., 0, 0]
    fx = field.f_grid[..., 0, 1]

    sin2 = (wavelength ** 2) * (fy ** 2 + fx ** 2)
    sin2 = jnp.clip(sin2, 0.0, 1.0)

    cos_med = jnp.sqrt(jnp.maximum(n_medium ** 2 - n_medium ** 2 * sin2, 0.0)) / n_medium
    cos_glass = jnp.sqrt(jnp.maximum(n_glass ** 2 - n_medium ** 2 * sin2, 0.0)) / n_glass

    rs = (n_medium * cos_med - n_glass * cos_glass) / (n_medium * cos_med + n_glass * cos_glass + 1e-30)
    rp = (n_glass * cos_med - n_medium * cos_glass) / (n_glass * cos_med + n_medium * cos_glass + 1e-30)

    phi = jnp.arctan2(fy, fx)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    Ex = field.u[..., 2]
    Ey = field.u[..., 1]

    E_s = -Ex * sin_phi + Ey * cos_phi
    E_p = Ex * cos_phi + Ey * sin_phi

    E_s_r = rs * E_s
    E_p_r = rp * E_p

    Ex_r = -E_s_r * sin_phi + E_p_r * cos_phi
    Ey_r = E_s_r * cos_phi + E_p_r * sin_phi

    new_u = field.u.at[..., 1].set(Ey_r)
    new_u = new_u.at[..., 2].set(Ex_r)
    return field.replace(u=new_u)
