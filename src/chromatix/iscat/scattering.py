from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, vmap

from chromatix.field import VectorField

from .field_utils import _split_xy, _with_xy, spatial_frequency_grid
from .mie import compute_pi_tau, mie_coefficients


@jax.jit
def s1_s2_precise(
    theta_n: float, an: Array, bn: Array, n_max: int
) -> Tuple[Array, Array]:
    """Compute Mie S1/S2 scattering amplitudes at angle theta_n."""
    u = jnp.cos(theta_n)
    pi, tau = compute_pi_tau(u, n_max)
    n = jnp.arange(1, n_max + 1)
    prefactor = (2 * n + 1) / (n * (n + 1))
    s1 = jnp.sum(prefactor * (an * pi[1:] + bn * tau[1:]))
    s2 = jnp.sum(prefactor * (an * tau[1:] + bn * pi[1:]))
    return s1, s2


@jax.jit
def compute_scattering_angle(
    fx: Array, fy: Array, wavelength: float, n_medium: float
) -> Array:
    """Non-paraxial scattering angle for high-NA systems."""
    f_mag = jnp.sqrt(fx**2 + fy**2)
    sin_theta = wavelength * f_mag / n_medium
    sin_theta = jnp.clip(sin_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arcsin(sin_theta)
    return theta


@jax.jit
def apply_vectorial_mie(
    field: VectorField, s1_map: Array, s2_map: Array, fx: Array, fy: Array
) -> VectorField:
    """Apply vectorial Mie coupling for s/p polarization components."""
    phi = jnp.arctan2(fy, fx)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    ez, ey, ex = _split_xy(field)

    e_parallel = cos_phi * ex + sin_phi * ey
    e_perp = -sin_phi * ex + cos_phi * ey

    e_parallel_s = s2_map * e_parallel
    e_perp_s = s1_map * e_perp

    ex_scattered = cos_phi * e_parallel_s - sin_phi * e_perp_s
    ey_scattered = sin_phi * e_parallel_s + cos_phi * e_perp_s

    return _with_xy(field, ex_scattered, ey_scattered, ez=ez)


def _shift_field_subpixel(field: VectorField, shift_px: Tuple[float, float]) -> VectorField:
    """Shift a field by a subpixel amount using a Fourier shift."""
    shift_y, shift_x = shift_px
    fx, fy = spatial_frequency_grid(field)
    out_shift = 2 * jnp.pi * (fy * shift_y + fx * shift_x)
    phase = jnp.exp(-1j * out_shift)
    phase = phase.reshape((1,) * (field.ndim - 4) + phase.shape + (1, 1))
    u = jnp.fft.fft2(field.u, axes=field.spatial_dims)
    u = u * phase
    u = jnp.fft.ifft2(u, axes=field.spatial_dims)
    return field.replace(u=u)


@jax.jit
def mie_scatter_polarized(
    field: VectorField,
    radius: float,
    n_particle: complex,
    n_medium: float,
    wavelength: float,
    particle_pos_px: Tuple[float, float],
    fx: Array,
    fy: Array,
    n_max: int,
) -> VectorField:
    """Single-particle Mie scattering with vectorial coupling."""
    k = 2 * jnp.pi / wavelength * n_medium
    x = k * radius
    m = n_particle / n_medium

    an, bn = mie_coefficients(m, x, n_max)
    theta = compute_scattering_angle(fx, fy, wavelength, n_medium)

    s1_s2_vmap = vmap(lambda t: s1_s2_precise(t, an, bn, n_max))
    s1_map, s2_map = s1_s2_vmap(theta.ravel())
    s1_map = s1_map.reshape(field.spatial_shape)
    s2_map = s2_map.reshape(field.spatial_shape)

    scattered_field = apply_vectorial_mie(field, s1_map, s2_map, fx, fy)
    scattered_field = _shift_field_subpixel(scattered_field, particle_pos_px)
    return scattered_field


@jax.jit
def multi_particle_mie_scatter(
    field: VectorField,
    radii: Array,
    n_particles: Array,
    positions_px: Array,
    n_medium: float,
    wavelength: float,
    fx: Array,
    fy: Array,
    n_max: int,
) -> VectorField:
    """Sum scattering from multiple particles with distinct parameters."""
    single_scatter_vmap = vmap(
        lambda r, n_p, pos: mie_scatter_polarized(
            field, r, n_p, n_medium, wavelength, (pos[0], pos[1]), fx, fy, n_max
        ),
        in_axes=(0, 0, 0),
    )

    scattered_fields = single_scatter_vmap(radii, n_particles, positions_px)
    ex = jnp.sum(scattered_fields.u[..., 2], axis=0)
    ey = jnp.sum(scattered_fields.u[..., 1], axis=0)
    ez = jnp.sum(scattered_fields.u[..., 0], axis=0)
    return _with_xy(field, ex, ey, ez=ez)
