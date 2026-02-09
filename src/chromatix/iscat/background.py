from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def generate_psd_background(
    field_shape: tuple[int, int],
    dx: float,
    sigma: float = 0.002,
    alpha: float = 1.5,
    f0: float = 0.01,
    key: Array | None = None,
) -> Array:
    """Generate a PSD-based rough surface phase screen."""
    ny, nx = field_shape
    fy = jnp.fft.fftfreq(ny, dx)
    fx = jnp.fft.fftfreq(nx, dx)
    ffy, ffx = jnp.meshgrid(fy, fx, indexing="ij")

    f_mag2 = ffx**2 + ffy**2 + f0**2
    psd = 1 / f_mag2**alpha

    if key is None:
        key = jax.random.PRNGKey(0)
    key_real, key_imag = jax.random.split(key)
    noise = jax.random.normal(key_real, (ny, nx)) + 1j * jax.random.normal(
        key_imag, (ny, nx)
    )
    phase_k = noise * jnp.sqrt(psd)
    phase_real = jnp.fft.ifft2(phase_k).real * sigma

    return jnp.exp(1j * phase_real)
