from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from chromatix.field import VectorField

from .field_utils import broadcast_spatial_mask, spatial_frequency_grid


@jax.jit
def apply_na_mask(
    field: VectorField, na: float, wavelength: float, n_medium: float
) -> VectorField:
    """Apply NA cutoff in k-space (illumination & collection cone)."""
    fx, fy = spatial_frequency_grid(field)
    f_mag = jnp.sqrt(fx**2 + fy**2)
    cutoff = na / wavelength
    mask = (f_mag <= cutoff).astype(jnp.complex128)
    mask = broadcast_spatial_mask(mask, field)

    u = jnp.fft.fft2(field.u, axes=field.spatial_dims)
    u = u * mask
    u = jnp.fft.ifft2(u, axes=field.spatial_dims)
    return field.replace(u=u)


@jax.jit
def apply_coherence_pupil(
    field: VectorField,
    coherence_length: float,
    wavelength: float,
    na: float,
) -> VectorField:
    """Apply a van Cittert–Zernike coherence pupil in k-space."""
    fx, fy = spatial_frequency_grid(field)
    f_mag = jnp.sqrt(fx**2 + fy**2)

    sigma = 1 / (2 * jnp.pi * coherence_length / wavelength)
    coherence_mask = jnp.exp(-((f_mag / (2 * sigma)) ** 2))

    na_cutoff = na / wavelength
    coherence_mask = coherence_mask * (f_mag <= na_cutoff)

    coherence_mask = broadcast_spatial_mask(coherence_mask, field)
    u = jnp.fft.fft2(field.u, axes=field.spatial_dims)
    u = u * coherence_mask
    u = jnp.fft.ifft2(u, axes=field.spatial_dims)
    return field.replace(u=u)
