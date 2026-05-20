"""iSCAT microscope forward model — pupil-plane formulation.

Physics:
  E_sc(y,x) = IFFT[ S(ky,kx) · P(ky,kx) · exp(-2πi·(fy·dy+fx·dx)) · exp(i·kz·dz) ]
  E_ref      = r_Fresnel · reference_amplitude   (uniform)
  I(y,x)    = |E_ref + E_sc(y,x)|²

Working entirely in the Fourier/pupil domain and transforming to the image
plane with a single IFFT avoids the confusion of applying pupil-plane
operations to a real-space field.
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from .mie import mie_coefficients, s1_s2, default_n_max
from .background import psd_background_phase


class iSCATMicroscope(eqx.Module):
    """iSCAT forward model — pupil-plane IFFT formulation."""

    # Trainable arrays
    radii: jax.Array
    n_particles: jax.Array
    positions_yx: jax.Array
    z_particles: jax.Array
    z_focal: jax.Array

    # Static configuration
    na: float = eqx.field(static=True)
    n_oil: float = eqx.field(static=True)
    n_glass: float = eqx.field(static=True)
    n_medium: float = eqx.field(static=True)
    wavelength: float = eqx.field(static=True)
    t_oil_ideal: float = eqx.field(static=True)
    scattering_model: str = eqx.field(static=True)
    n_max: Optional[int] = eqx.field(static=True)
    reference_amplitude: float = eqx.field(static=True)
    sensor: Optional[object] = eqx.field(static=True)

    def __init__(
        self,
        radii,
        n_particles,
        positions_yx,
        z_particles,
        z_focal,
        *,
        na: float = 1.3,
        n_oil: float = 1.518,
        n_glass: float = 1.518,
        n_medium: float = 1.33,
        wavelength: float = 532e-9,
        t_oil_ideal: float = 170e-6,
        scattering_model: str = "rayleigh",
        n_max: Optional[int] = None,
        reference_amplitude: float = 1.0,
        sensor=None,
    ):
        self.radii = jnp.asarray(radii, dtype=jnp.float32)
        self.n_particles = jnp.asarray(n_particles, dtype=jnp.complex64)
        self.positions_yx = jnp.asarray(positions_yx, dtype=jnp.float32)
        self.z_particles = jnp.asarray(z_particles, dtype=jnp.float32)
        self.z_focal = jnp.asarray(z_focal, dtype=jnp.float32)
        self.na = na
        self.n_oil = n_oil
        self.n_glass = n_glass
        self.n_medium = n_medium
        self.wavelength = wavelength
        self.t_oil_ideal = t_oil_ideal
        self.scattering_model = scattering_model
        self.n_max = n_max
        self.reference_amplitude = reference_amplitude
        self.sensor = sensor

    def __call__(
        self,
        key=None,
        theta_inc: float = 0.0,
        phi_inc: float = 0.0,
        shape: tuple = (64, 64),
        dx: float = 65e-9,
        include_background: bool = False,
        pad_width: int = 0,
    ):
        """Simulate iSCAT image intensity of shape (ny, nx).

        Pupil-plane formulation:
          1. Build frequency grid (DC at centre, fftshifted convention).
          2. For each particle: compute pupil field = S(theta) * P * defocus * pos_phase.
          3. Sum pupils, IFFT → scattered field at image plane.
          4. Add uniform Fresnel reference.
          5. I = |E_ref + E_sc|².
        """
        ny, nx = shape

        # ── Pupil frequency grid (cycles/m), DC at centre ─────────────────────
        fy_1d = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=dx))
        fx_1d = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=dx))
        FY, FX = jnp.meshgrid(fy_1d, fx_1d, indexing="ij")  # (ny, nx)

        f2 = FY ** 2 + FX ** 2
        f_na = self.na / self.wavelength
        na_mask = (f2 <= f_na ** 2).astype(jnp.float32)

        k_med = 2.0 * jnp.pi * self.n_medium / self.wavelength

        # Axial k-vector (for defocus phase)
        kz_med = jnp.sqrt(jnp.maximum(k_med ** 2 - (2.0 * jnp.pi) ** 2 * f2, 0.0))

        # Scattering angles at each pupil point
        sin_theta = jnp.clip(self.wavelength * jnp.sqrt(f2), 0.0, 1.0)
        theta_grid = jnp.arcsin(sin_theta)      # (ny, nx)
        phi_grid = jnp.arctan2(FY, FX)          # (ny, nx)

        # ── Per-particle pupil contribution ───────────────────────────────────
        def pupil_one(i):
            radius = self.radii[i]
            m = self.n_particles[i]
            dy = self.positions_yx[i, 0]
            dx_p = self.positions_yx[i, 1]
            z_p = self.z_particles[i]

            x_size = k_med * radius

            if self.scattering_model == "mie":
                n_max_use = self.n_max if self.n_max is not None else 20
                a_n, b_n = mie_coefficients(m, x_size, n_max_use)
                s1, s2 = s1_s2(theta_grid, a_n, b_n)
                # x-polarised: S = S2 cos²φ + S1 sin²φ
                S = s2 * jnp.cos(phi_grid) ** 2 + s1 * jnp.sin(phi_grid) ** 2
            else:
                # Isotropic Rayleigh: S = -i k³ α / (4π)
                alpha = (4.0 * jnp.pi * radius ** 3
                         * (m ** 2 - 1.0) / (m ** 2 + 2.0))
                S = (-1j * k_med ** 3 * alpha / (4.0 * jnp.pi)
                     * jnp.ones((ny, nx), dtype=jnp.complex64))

            # Defocus phase: particle at z_p, focus at z_focal
            dz = z_p - self.z_focal
            defocus_phase = jnp.exp(1j * kz_med * dz)

            # Lateral position → phase ramp (FT shift theorem)
            pos_phase = jnp.exp(-2j * jnp.pi * (FY * dy + FX * dx_p))

            return S * na_mask * defocus_phase * pos_phase  # (ny, nx)

        n_part = self.radii.shape[0]
        all_pupils = jax.vmap(pupil_one)(jnp.arange(n_part))
        pupil_total = jnp.sum(all_pupils, axis=0)  # (ny, nx)

        # ── IFFT: pupil plane → image plane ───────────────────────────────────
        # pupil_total has DC at centre → ifftshift → standard FFT ordering → ifft2
        E_sc = jnp.fft.ifft2(jnp.fft.ifftshift(pupil_total))
        # Undo 1/(ny*nx) normalisation so amplitude scales with pupil fill
        E_sc = E_sc * (ny * nx)

        # ── Fresnel reference (uniform, normal incidence) ─────────────────────
        r_fresnel = ((self.n_medium - self.n_glass)
                     / (self.n_medium + self.n_glass))
        E_ref_scalar = r_fresnel * self.reference_amplitude

        if include_background and key is not None:
            bg = psd_background_phase(key, shape, dx)[..., 0]  # (ny, nx)
            E_ref_field = E_ref_scalar * bg
        else:
            E_ref_field = E_ref_scalar * jnp.ones((ny, nx), dtype=jnp.complex64)

        # ── Intensity ──────────────────────────────────────────────────────────
        E_total = E_ref_field + E_sc
        intensity = jnp.abs(E_total) ** 2

        if self.sensor is not None:
            intensity = self.sensor(intensity)

        return intensity


def simulate_z_stack(microscope: iSCATMicroscope, z_focal_values, **kwargs):
    """Simulate a z-stack by vmapping over focal positions.

    Returns intensity stack of shape (nz, ny, nx).
    """
    def sim_at_z(z):
        m = eqx.tree_at(lambda mc: mc.z_focal, microscope, z)
        return m(**kwargs)

    return jax.vmap(sim_at_z)(jnp.asarray(z_focal_values, dtype=jnp.float32))
