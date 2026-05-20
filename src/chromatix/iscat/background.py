"""Stochastic background phase model for iSCAT."""

import jax
import jax.numpy as jnp


def psd_background_phase(key, shape, dx, sigma: float = 0.05, alpha: float = 2.0, f0: float = 1.0):
    """Generate a random background field with power-law PSD.

    The phase is drawn from a Gaussian process with PSD ~ 1/(f0^2 + f^2)^(alpha/2).

    Args:
        key: JAX PRNG key
        shape: (ny, nx) spatial shape
        dx: pixel size in metres (scalar or 2-element)
        sigma: overall phase amplitude (radians RMS)
        alpha: PSD power-law exponent
        f0: low-frequency rolloff (cycles/metre)

    Returns:
        Complex array of shape (ny, nx, 1) representing exp(i * phase)
    """
    ny, nx = shape
    dx_arr = jnp.broadcast_to(jnp.asarray(dx, dtype=jnp.float32), (2,))

    fy = jnp.fft.fftfreq(ny, d=float(dx_arr[0]))
    fx = jnp.fft.fftfreq(nx, d=float(dx_arr[1]))
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
    f2 = FY ** 2 + FX ** 2

    psd = 1.0 / (f0 ** 2 + f2) ** (alpha / 2.0)
    psd = psd.at[0, 0].set(0.0)  # remove DC

    noise = jax.random.normal(key, shape=(ny, nx)) + 1j * jax.random.normal(
        jax.random.fold_in(key, 1), shape=(ny, nx)
    )
    phase_fft = noise * jnp.sqrt(psd)
    phase = jnp.real(jnp.fft.ifft2(phase_fft))

    # Normalise to desired sigma
    phase = phase / (jnp.std(phase) + 1e-12) * sigma
    return jnp.exp(1j * phase)[..., jnp.newaxis]  # (ny, nx, 1)
