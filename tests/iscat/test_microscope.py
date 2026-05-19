"""Tests for iSCAT microscope forward model."""

import jax
import jax.numpy as jnp
import pytest
from chromatix.iscat.microscope import iSCATMicroscope, simulate_z_stack


SHAPE = (32, 32)
DX = 65e-9
WL = 532e-9


def make_microscope():
    return iSCATMicroscope(
        radii=jnp.array([50e-9]),
        n_particles=jnp.array([1.5 + 0.0j]),
        positions_yx=jnp.zeros((1, 2)),
        z_particles=jnp.array([0.0]),
        z_focal=jnp.array(0.0),
        na=1.3,
        n_oil=1.518,
        n_glass=1.518,
        n_medium=1.33,
        wavelength=WL,
        scattering_model="rayleigh",
    )


def test_forward_shape():
    m = make_microscope()
    img = m(shape=SHAPE, dx=DX)
    assert img.shape == SHAPE


def test_forward_positive():
    m = make_microscope()
    img = m(shape=SHAPE, dx=DX)
    assert jnp.all(img >= 0), "Intensity must be non-negative"


def test_gradient_flows_through_positions():
    m = make_microscope()

    def loss(pos):
        import equinox as eqx
        m2 = eqx.tree_at(lambda mc: mc.positions_yx, m, pos)
        img = m2(shape=SHAPE, dx=DX)
        return jnp.sum(img)

    pos = jnp.zeros((1, 2))
    grad = jax.grad(loss)(pos)
    assert jnp.isfinite(grad).all(), "Gradient must be finite"


def test_z_stack_shape():
    m = make_microscope()
    z_vals = jnp.linspace(-200e-9, 200e-9, 5)
    stack = simulate_z_stack(m, z_vals, shape=SHAPE, dx=DX)
    assert stack.shape == (5, *SHAPE)
