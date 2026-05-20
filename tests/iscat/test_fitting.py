"""Tests for iSCAT inverse problem fitting."""

import jax
import jax.numpy as jnp
import pytest
from chromatix.iscat.microscope import iSCATMicroscope
from chromatix.iscat.fitting import fit_particles


SHAPE = (32, 32)
DX = 65e-9
WL = 532e-9


def make_microscope(pos_yx=None):
    if pos_yx is None:
        pos_yx = jnp.zeros((1, 2))
    return iSCATMicroscope(
        radii=jnp.array([50e-9]),
        n_particles=jnp.array([1.5 + 0.0j]),
        positions_yx=pos_yx,
        z_particles=jnp.array([0.0]),
        z_focal=jnp.array(0.0),
        na=1.3,
        n_oil=1.518,
        n_glass=1.518,
        n_medium=1.33,
        wavelength=WL,
        scattering_model="rayleigh",
    )


def test_fitting_reduces_loss():
    """Fitting should reduce MPG NLL over 50 Adam steps."""
    # Ground truth
    true_pos = jnp.array([[100e-9, -80e-9]])
    m_true = make_microscope(pos_yx=true_pos)
    obs = m_true(shape=SHAPE, dx=DX)

    # Initial guess (wrong position)
    init_pos = jnp.zeros((1, 2))
    m_init = make_microscope(pos_yx=init_pos)
    initial_loss = jnp.sum((m_init(shape=SHAPE, dx=DX) - obs) ** 2)

    _, losses = fit_particles(
        m_init,
        obs,
        trainable_fields=["positions_yx"],
        gain=0.0,
        read_var=1.0,
        n_steps=50,
        lr=1e-9,
        call_kwargs={"shape": SHAPE, "dx": DX},
    )

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )


def test_fitting_returns_correct_shapes():
    """fit_particles should return (microscope, losses array)."""
    m = make_microscope()
    obs = m(shape=SHAPE, dx=DX)

    fitted, losses = fit_particles(
        m,
        obs,
        trainable_fields=["positions_yx"],
        gain=0.0,
        read_var=1.0,
        n_steps=5,
        lr=1e-6,
        call_kwargs={"shape": SHAPE, "dx": DX},
    )

    assert losses.shape == (5,)
    assert hasattr(fitted, "positions_yx")
    assert fitted.positions_yx.shape == m.positions_yx.shape
