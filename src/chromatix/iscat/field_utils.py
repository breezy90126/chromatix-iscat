from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chromatix.field import VectorField


def _split_xy(field: VectorField) -> tuple[Array, Array, Array]:
    """Return (ez, ey, ex) components from a VectorField."""
    ez = field.u[..., 0]
    ey = field.u[..., 1]
    ex = field.u[..., 2]
    return ez, ey, ex


def _with_xy(field: VectorField, ex: Array, ey: Array, ez: Array | None = None) -> VectorField:
    """Return a VectorField with updated x/y (and optional z) components."""
    if ez is None:
        ez = field.u[..., 0]
    u = field.u
    u = u.at[..., 0].set(ez)
    u = u.at[..., 1].set(ey)
    u = u.at[..., 2].set(ex)
    return field.replace(u=u)


def spatial_frequency_grid(field: VectorField) -> tuple[Array, Array]:
    """Return spatial frequency grids (fx, fy) using the first wavelength spacing."""
    dy, dx = field._dx[:, 0]
    ny, nx = field.spatial_shape
    fy = jnp.fft.fftfreq(ny, dy)
    fx = jnp.fft.fftfreq(nx, dx)
    ffy, ffx = jnp.meshgrid(fy, fx, indexing="ij")
    return ffx, ffy


def broadcast_spatial_mask(mask: Array, field: VectorField) -> Array:
    """Broadcast a spatial mask to match the field.u shape."""
    shape = (1,) * (field.ndim - 4) + mask.shape + (1, 1)
    return mask.reshape(shape)
