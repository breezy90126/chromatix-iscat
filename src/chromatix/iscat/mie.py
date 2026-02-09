from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, lax, vmap


@jax.jit
def mie_coefficients(m: complex, x: float, n_max: int) -> Tuple[Array, Array]:
    """
    Compute Mie scattering coefficients an, bn with a stable log-derivative.

    Args:
        m: Relative refractive index (n_particle / n_medium).
        x: Size parameter k * r.
        n_max: Expansion order.

    Returns:
        an, bn arrays of shape (n_max,).
    """
    x_safe = jnp.maximum(x, 1e-12)

    def dn_downward_scan(carry: Array, n: Array) -> tuple[Array, Array]:
        dn_next = n / (m * x_safe) - 1.0 / (carry + n / (m * x_safe))
        return dn_next, dn_next

    dn_init = -1j
    orders = jnp.arange(n_max, 0, -1)
    _, dn_list = lax.scan(dn_downward_scan, dn_init, orders)
    dn_list = dn_list[::-1]

    psi_0 = jnp.sin(x_safe)
    psi_1 = jnp.sin(x_safe) / x_safe - jnp.cos(x_safe)
    chi_0 = jnp.cos(x_safe)
    chi_1 = jnp.cos(x_safe) / x_safe + jnp.sin(x_safe)

    xi_0 = psi_0 - 1j * chi_0
    xi_1 = psi_1 - 1j * chi_1

    def body_fn(carry: tuple[Array, Array, Array, Array], n: Array):
        psi_nm1, psi_n, xi_nm1, xi_n = carry
        psi_np1 = (2 * n + 1) / x_safe * psi_n - psi_nm1
        xi_np1 = (2 * n + 1) / x_safe * xi_n - xi_nm1
        return (psi_n, psi_np1, xi_n, xi_np1), (psi_n, xi_n)

    init_carry = (psi_0, psi_1, xi_0, xi_1)
    _, (psi_list, xi_list) = lax.scan(body_fn, init_carry, jnp.arange(1, n_max + 1))
    psi_all = jnp.concatenate([psi_0[None], psi_1[None], psi_list])
    xi_all = jnp.concatenate([xi_0[None], xi_1[None], xi_list])

    def compute_an_bn(n: Array, dn: Array) -> tuple[Array, Array]:
        n_c = n.astype(jnp.complex128)
        an = (dn / m + n_c / x_safe) * psi_all[n] - psi_all[n - 1]
        an /= (dn / m + n_c / x_safe) * xi_all[n] - xi_all[n - 1]

        bn = (m * dn + n_c / x_safe) * psi_all[n] - psi_all[n - 1]
        bn /= (m * dn + n_c / x_safe) * xi_all[n] - xi_all[n - 1]

        return an, bn

    an_bn = vmap(compute_an_bn)(jnp.arange(1, n_max + 1), dn_list)
    an = an_bn[0]
    bn = an_bn[1]
    return an, bn


@jax.jit
def compute_pi_tau(u: float, n_max: int) -> Tuple[Array, Array]:
    """Compute pi_n and tau_n via Legendre recursion."""
    pi = jnp.zeros(n_max + 1)
    tau = jnp.zeros(n_max + 1)

    pi = pi.at[1].set(1.0)
    tau = tau.at[1].set(u)

    def body(carry: tuple[Array, Array], n: Array):
        pi_nm2, pi_nm1 = carry
        pi_n = ((2 * n - 1) / (n - 1)) * u * pi_nm1 - (n / (n - 1)) * pi_nm2
        tau_n = n * u * pi_n - (n + 1) * pi_nm1
        return (pi_nm1, pi_n), (pi_n, tau_n)

    (_, _), (pi_list, tau_list) = lax.scan(body, (0.0, 1.0), jnp.arange(2, n_max + 1))
    pi = pi.at[2:].set(pi_list)
    tau = tau.at[2:].set(tau_list)
    return pi, tau
