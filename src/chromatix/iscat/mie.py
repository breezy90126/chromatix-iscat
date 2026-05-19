"""Mie scattering coefficients and angle functions."""

import jax
import jax.numpy as jnp
from functools import partial


def default_n_max(x: float) -> int:
    """Wiscombe truncation criterion for number of Mie terms."""
    x = float(x)
    if x <= 0.02:
        return 2
    elif x <= 8.0:
        return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 1
    elif x <= 4200.0:
        return int(x + 4.05 * x ** (1.0 / 3.0) + 2.0) + 1
    else:
        return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 1


def _log_derivative_downward(m, x, n_max: int):
    """Downward recursion for log-derivative D_n(mx).

    Returns array of shape (n_max,) with D[0]=D_1, ..., D[n_max-1]=D_n_max.
    """
    mx = m * x
    n_start = n_max + 15

    def body(carry, n):
        D_next = carry
        D_curr = (n + 1) / mx - 1.0 / (D_next + (n + 1) / mx)
        return D_curr, D_curr

    D_init = jnp.zeros((), dtype=jnp.complex64) + 0j
    _, D_all = jax.lax.scan(body, D_init, jnp.arange(n_start - 1, 0, -1, dtype=jnp.float32))
    # D_all[i] corresponds to D_{n_start - i}, length n_start-1
    # We need D_1 .. D_n_max: these are at indices n_start-2 .. n_start-n_max-1 (reversed)
    # After scan from n_start-1 down to 1:
    # D_all[0] = D_{n_start-1}, D_all[1] = D_{n_start-2}, ...
    # D_n corresponds to D_all[n_start - 1 - n]
    indices = n_start - 1 - jnp.arange(1, n_max + 1)
    return D_all[indices]


def mie_coefficients(m, x, n_max: int):
    """Compute Mie coefficients a_n and b_n.

    Args:
        m: complex relative refractive index (n_particle / n_medium)
        x: size parameter (2 pi r / lambda * n_medium)
        n_max: number of terms

    Returns:
        a_n, b_n: arrays of shape (n_max,)
    """
    x = jnp.asarray(x, dtype=jnp.float32)
    m = jnp.asarray(m, dtype=jnp.complex64)

    D = _log_derivative_downward(m, x, n_max)  # shape (n_max,)

    # Upward scan for Riccati-Bessel psi_n and xi_n
    # psi_0 = sin(x), psi_1 = sin(x)/x - cos(x)
    # xi_n = psi_n - i * chi_n
    ns = jnp.arange(1, n_max + 1, dtype=jnp.float32)

    psi_prev = jnp.sin(x)
    psi_curr = jnp.sin(x) / x - jnp.cos(x)
    chi_prev = -jnp.cos(x)
    chi_curr = -jnp.cos(x) / x - jnp.sin(x)

    def body(carry, n):
        psi_nm1, psi_n, chi_nm1, chi_n = carry
        psi_np1 = (2 * n + 1) / x * psi_n - psi_nm1
        chi_np1 = (2 * n + 1) / x * chi_n - chi_nm1
        xi_n = psi_n - 1j * chi_n
        D_n = D[n - 1]
        a_n = (D_n / m + n / x) * psi_n - psi_nm1
        a_n = a_n / ((D_n / m + n / x) * xi_n - (psi_nm1 - 1j * chi_nm1))
        b_n = (m * D_n + n / x) * psi_n - psi_nm1
        b_n = b_n / ((m * D_n + n / x) * xi_n - (psi_nm1 - 1j * chi_nm1))
        return (psi_n, psi_np1, chi_n, chi_np1), (a_n, b_n)

    init = (psi_prev, psi_curr, chi_prev, chi_curr)
    _, (a_n, b_n) = jax.lax.scan(body, init, jnp.arange(1, n_max + 1, dtype=jnp.int32))
    return a_n.astype(jnp.complex64), b_n.astype(jnp.complex64)


def _pi_tau(u, n_max: int):
    """Legendre angle functions pi_n and tau_n at cos(theta) = u.

    Returns pi_n, tau_n each of shape (n_max,).
    """
    # pi_0 = 0, pi_1 = 1
    # pi_{n+1} = ((2n+1)*u*pi_n - (n+1)*pi_{n-1}) / n
    # tau_n = n * u * pi_n - (n+1) * pi_{n-1}

    def body(carry, n):
        pi_nm1, pi_n = carry
        pi_np1 = ((2 * n + 1) * u * pi_n - (n + 1) * pi_nm1) / n
        tau_n = n * u * pi_n - (n + 1) * pi_nm1
        return (pi_n, pi_np1), (pi_n, tau_n)

    init = (jnp.zeros_like(u), jnp.ones_like(u))
    _, (pi_ns, tau_ns) = jax.lax.scan(body, init, jnp.arange(1, n_max + 1, dtype=jnp.float32))
    return pi_ns, tau_ns


def s1_s2(theta, a_n, b_n):
    """Compute S1, S2 scattering amplitudes.

    Args:
        theta: scattering angles, any shape (radians)
        a_n: Mie a coefficients, shape (n_max,)
        b_n: Mie b coefficients, shape (n_max,)

    Returns:
        S1, S2: same shape as theta, complex
    """
    n_max = a_n.shape[0]
    flat = theta.ravel()
    u = jnp.cos(flat)

    def single(ui):
        pi_ns, tau_ns = _pi_tau(ui, n_max)
        ns = jnp.arange(1, n_max + 1, dtype=jnp.float32)
        w = (2 * ns + 1) / (ns * (ns + 1))
        s1 = jnp.sum(w * (a_n * pi_ns + b_n * tau_ns))
        s2 = jnp.sum(w * (a_n * tau_ns + b_n * pi_ns))
        return s1, s2

    s1_flat, s2_flat = jax.vmap(single)(u)
    return s1_flat.reshape(theta.shape), s2_flat.reshape(theta.shape)
