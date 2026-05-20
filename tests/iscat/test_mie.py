"""Tests for Mie scattering coefficients."""

import jax.numpy as jnp
import pytest
from chromatix.iscat.mie import default_n_max, mie_coefficients, s1_s2


def test_default_n_max_positive():
    assert default_n_max(1.0) > 0
    assert default_n_max(0.01) >= 2
    assert default_n_max(100.0) > 10


def test_mie_coefficients_shape():
    m = jnp.array(1.5 + 0.0j)
    x = jnp.array(1.0)
    n_max = 10
    a_n, b_n = mie_coefficients(m, x, n_max)
    assert a_n.shape == (n_max,)
    assert b_n.shape == (n_max,)


def test_mie_m_equals_1_gives_zero():
    """For m=1 (no contrast), coefficients should vanish."""
    m = jnp.array(1.0 + 0.0j)
    x = jnp.array(1.0)
    a_n, b_n = mie_coefficients(m, x, 10)
    assert jnp.allclose(a_n, 0.0, atol=1e-5), f"a_n not zero: {a_n}"
    assert jnp.allclose(b_n, 0.0, atol=1e-5), f"b_n not zero: {b_n}"


def test_rayleigh_limit():
    """For very small x, a_1 should approach the Rayleigh approximation."""
    m = jnp.array(1.5 + 0.0j)
    x = jnp.array(0.001)
    n_max = default_n_max(float(x))
    a_n, b_n = mie_coefficients(m, x, n_max)
    # Rayleigh: a_1 ~ -i (2/3) x^3 (m^2-1)/(m^2+2)
    m2 = m ** 2
    a1_rayleigh = -1j * (2.0 / 3.0) * x ** 3 * (m2 - 1) / (m2 + 2)
    # Relative tolerance of 0.01% is generous for float64 but robust
    assert jnp.abs(a_n[0] - a1_rayleigh) < 1e-4 * jnp.abs(a1_rayleigh)


def test_s1_s2_forward():
    """At theta=0: S1 == S2, and S1 = (1/2) * sum_n (2n+1) * (a_n + b_n).

    The pi_n(theta=0) = tau_n(theta=0) = n(n+1)/2, so both amplitudes equal.
    """
    m = jnp.array(1.5 + 0.0j)
    x = jnp.array(1.0)
    n_max = 10
    a_n, b_n = mie_coefficients(m, x, n_max)
    theta = jnp.zeros((1,))
    s1, s2 = s1_s2(theta, a_n, b_n)

    # Forward scattering symmetry: S1(0) == S2(0) for any sphere
    assert jnp.allclose(s1[0], s2[0], rtol=1e-5), f"S1(0)={s1[0]} != S2(0)={s2[0]}"

    # Correct formula: pi_n(0) = n(n+1)/2, so S1(0) = 0.5 * sum_n (2n+1)*(a_n+b_n)
    ns = jnp.arange(1, n_max + 1, dtype=a_n.real.dtype)
    expected = 0.5 * jnp.sum((2 * ns + 1) * (a_n + b_n))
    assert jnp.allclose(s1[0], expected, rtol=1e-4), f"S1(0)={s1[0]} != expected={expected}"


def test_s1_s2_shape():
    m = jnp.array(1.5 + 0.0j)
    x = jnp.array(1.0)
    a_n, b_n = mie_coefficients(m, x, 10)
    theta = jnp.linspace(0, jnp.pi, 20)
    s1, s2 = s1_s2(theta, a_n, b_n)
    assert s1.shape == (20,)
    assert s2.shape == (20,)
