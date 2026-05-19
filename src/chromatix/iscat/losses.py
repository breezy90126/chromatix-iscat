"""Loss functions for iSCAT inverse problems."""

import jax.numpy as jnp


def mpg_nll_per_pixel(image_obs, image_sim, gain: float, read_var: float, eps: float = 1e-6):
    """Mixed Poisson-Gaussian negative log-likelihood per pixel.

    Variance model: var = gain * max(sim, 0) + read_var + eps
    NLL per pixel: 0.5 * (log(2 pi var) + (obs - sim)^2 / var)

    Args:
        image_obs: observed image
        image_sim: simulated image
        gain: effective gain (alpha for EMCCD, 1 for sCMOS Poisson)
        read_var: readout noise variance (sigma^2)
        eps: numerical floor for variance

    Returns:
        per-pixel NLL, same shape as inputs
    """
    var = gain * jnp.maximum(image_sim, 0.0) + read_var + eps
    return 0.5 * (jnp.log(2.0 * jnp.pi * var) + (image_obs - image_sim) ** 2 / var)


def mpg_nll(image_obs, image_sim, gain: float, read_var: float, eps: float = 1e-6):
    """Sum of MPG NLL over all pixels.

    Returns scalar loss.
    """
    return jnp.sum(mpg_nll_per_pixel(image_obs, image_sim, gain, read_var, eps))
