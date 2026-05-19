"""Tests for MPG noise model / NLL loss."""

import jax.numpy as jnp
import pytest
from chromatix.iscat.losses import mpg_nll, mpg_nll_per_pixel


def test_nll_finite():
    obs = jnp.ones((8, 8))
    sim = jnp.ones((8, 8))
    loss = mpg_nll(obs, sim, gain=1.0, read_var=0.1)
    assert jnp.isfinite(loss)


def test_nll_positive():
    obs = jnp.ones((8, 8))
    sim = jnp.ones((8, 8))
    loss = mpg_nll(obs, sim, gain=1.0, read_var=0.1)
    assert loss > 0


def test_nll_minimum_pure_gaussian():
    """With gain=0 (pure Gaussian), NLL is minimised when sim==obs."""
    obs = jnp.array([[2.0, 3.0], [4.0, 5.0]])
    loss_at = mpg_nll(obs, obs, gain=0.0, read_var=1.0)
    loss_off = mpg_nll(obs, obs + 0.5, gain=0.0, read_var=1.0)
    assert loss_at < loss_off, "NLL should be lower when sim == obs"


def test_per_pixel_shape():
    obs = jnp.ones((4, 6))
    sim = jnp.ones((4, 6))
    pp = mpg_nll_per_pixel(obs, sim, gain=1.0, read_var=0.1)
    assert pp.shape == (4, 6)


def test_nll_increases_with_residual():
    obs = jnp.ones((8, 8)) * 5.0
    sim_near = jnp.ones((8, 8)) * 5.1
    sim_far = jnp.ones((8, 8)) * 7.0
    loss_near = mpg_nll(obs, sim_near, gain=1.0, read_var=0.5)
    loss_far = mpg_nll(obs, sim_far, gain=1.0, read_var=0.5)
    assert loss_near < loss_far
