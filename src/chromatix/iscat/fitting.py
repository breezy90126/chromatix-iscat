"""Inverse problem fitting for iSCAT particle localisation."""

from typing import Sequence, Optional
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from .losses import mpg_nll


def fit_particles(
    microscope_init,
    observed_image,
    trainable_fields: Sequence[str],
    gain: float,
    read_var: float,
    n_steps: int = 200,
    lr: float = 1e-3,
    key=None,
    call_kwargs: Optional[dict] = None,
):
    """Fit microscope parameters to an observed image using Adam.

    The optimiser operates on a plain dict of trainable arrays so that
    optax never sees any Equinox sentinel values.

    Args:
        microscope_init: iSCATMicroscope instance (initial parameters)
        observed_image: target image array
        trainable_fields: list of attribute names to optimise
            (e.g. ["positions_yx", "z_particles"])
        gain: MPG gain parameter
        read_var: MPG readout variance
        n_steps: number of gradient steps
        lr: Adam learning rate
        key: optional JAX PRNG key (passed to microscope.__call__)
        call_kwargs: additional keyword arguments for microscope.__call__

    Returns:
        (fitted_microscope, losses) where losses is a 1-D array of length n_steps
    """
    call_kwargs = call_kwargs or {}

    # Extract trainable values into a plain dict — no equinox pytree magic
    trainable_vals = {f: getattr(microscope_init, f) for f in trainable_fields}

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(trainable_vals)

    def loss_fn(vals: dict):
        m = microscope_init
        for fname, v in vals.items():
            m = eqx.tree_at(lambda mc, fn=fname: getattr(mc, fn), m, v)
        sim = m(key=key, **call_kwargs)
        return mpg_nll(observed_image, sim, gain, read_var)

    @jax.jit
    def step(vals, state):
        loss, grads = jax.value_and_grad(loss_fn)(vals)
        updates, new_state = optimizer.update(grads, state, vals)
        new_vals = optax.apply_updates(vals, updates)
        return new_vals, new_state, loss

    losses = []
    for _ in range(n_steps):
        trainable_vals, opt_state, loss = step(trainable_vals, opt_state)
        losses.append(float(loss))

    # Reconstruct the fitted microscope
    fitted = microscope_init
    for fname, v in trainable_vals.items():
        fitted = eqx.tree_at(lambda mc, fn=fname: getattr(mc, fn), fitted, v)

    return fitted, jnp.asarray(losses)
