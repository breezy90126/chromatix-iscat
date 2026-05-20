"""iSCAT microscope forward model — Debye-Wolf radial integral formulation.

Physics reference: iPSF viewer (Debye-Wolf):
  - Angular quadrature (midpoint rule, NT points) over NA cone
  - Fresnel T_p, T_s at oil/glass interface
  - Rayleigh: α = 4π r³ (m²-1)/(m²+2)
  - E0 = μ √(C₂ T) exp(i·arg α),  C₂ = ks⁴/(6π)|α|²,  μ = θ_max/π
  - A0(θ) = E0·(ts + tp·cosθ),  A2(θ) = E0·(ts - tp·cosθ)
  - Defocus phase: ab = z_m·n_m·(cosθ+1) + n_oil·(t_oil−t_ideal)·(cosθ−1) − π/2
  - I0(r) = Σ_t A0·e^{ikab}·J0(k n_oil r sinθ)·sinθ√cosθ dθ
  - I2(r) similarly with A2, J2
  - Ex = −(k/2)i·(I0 + I2 cos2φ),  Ey = −(k/2)i·(I2 sin2φ)
  - I = Er² + 2 Er Re(Ex) + |Ex|² + |Ey|²,  Er = √R·ref_amp
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx

from .background import psd_background_phase


class iSCATMicroscope(eqx.Module):
    """iSCAT forward model — Debye-Wolf formulation."""

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
    n_theta: int = eqx.field(static=True)
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
        n_theta: int = 15,
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
        self.n_theta = n_theta
        self.reference_amplitude = reference_amplitude
        self.sensor = sensor

    def __call__(
        self,
        key=None,
        shape: tuple = (64, 64),
        dx: float = 65e-9,
        include_background: bool = False,
        pad_width: int = 0,
        **kwargs,
    ):
        """Simulate iSCAT image intensity of shape (ny, nx).

        Debye-Wolf formulation:
          1. Angular quadrature over NA cone (n_theta midpoint points).
          2. Fresnel coefficients T_p, T_s at oil→glass interface.
          3. Per-particle: Rayleigh α → E0 → A0/A2 amplitudes.
          4. Radial integrals I0, I2 via J0/J2 Bessel functions.
          5. Scattered field Ex, Ey at image plane.
          6. I = Er² + 2 Er Re(Ex) + |Ex|² + |Ey|².
        """
        ny, nx = shape

        # Pixel coordinate grids centred at image centre
        y_1d = (jnp.arange(ny, dtype=jnp.float32) - ny / 2.0) * dx
        x_1d = (jnp.arange(nx, dtype=jnp.float32) - nx / 2.0) * dx
        Y, X = jnp.meshgrid(y_1d, x_1d, indexing="ij")  # (ny, nx)

        # ── Optical constants ─────────────────────────────────────────────────
        k = 2.0 * jnp.pi / self.wavelength
        ks = self.n_medium * k
        R = ((self.n_glass - self.n_medium) / (self.n_glass + self.n_medium)) ** 2
        T = 1.0 - R
        Er = jnp.sqrt(R) * self.reference_amplitude

        # ── Angular quadrature (midpoint rule) ────────────────────────────────
        thmax = jnp.arcsin(jnp.clip(self.na / self.n_oil, 0.0, 1.0))
        mu = thmax / jnp.pi          # normalisation factor
        dt = thmax / self.n_theta
        t_idx = jnp.arange(self.n_theta, dtype=jnp.float32)
        th = (t_idx + 0.5) * dt      # (n_theta,) midpoint angles
        sth = jnp.sin(th)
        cth = jnp.cos(th)

        # ── Fresnel transmission coefficients (objective → coverslip → medium) ─
        sg = self.n_oil * sth / self.n_glass
        cg = jnp.sqrt(jnp.maximum(1.0 - sg ** 2, 0.0))
        tp = (2.0 * self.n_medium * cth
              / (self.n_glass * cth + self.n_medium * cg))
        ts = (2.0 * self.n_medium * cth
              / (self.n_glass * cg + self.n_medium * cth))

        # Apodization / geometrical factor
        scf = sth * jnp.sqrt(jnp.maximum(cth, 0.0))  # (n_theta,)

        # ── Per-particle scattered field ──────────────────────────────────────
        def particle_field(i):
            """Return (Ex, Ey) complex (ny, nx) for particle i."""
            radius = self.radii[i]
            m = self.n_particles[i]
            dy = self.positions_yx[i, 0]
            dx_p = self.positions_yx[i, 1]
            z_p = self.z_particles[i]

            z_m = z_p - self.z_focal   # defocus

            # Rayleigh polarisability (complex)
            alpha = (4.0 * jnp.pi * radius ** 3
                     * (m ** 2 - 1.0) / (m ** 2 + 2.0))
            C2 = (ks ** 4 / (6.0 * jnp.pi)) * jnp.abs(alpha) ** 2
            E0_amp = mu * jnp.sqrt(C2) * jnp.sqrt(T)
            E0 = E0_amp * jnp.exp(1j * jnp.angle(alpha))

            # Angular amplitude envelopes  (n_theta,)
            A0 = E0 * (ts + tp * cth)
            A2 = E0 * (ts - tp * cth)

            # Rigorous defocus phase with oil-layer aberration correction
            t_oil = z_m + self.t_oil_ideal - self.n_oil * z_m / self.n_medium
            ab = (z_m * self.n_medium * (cth + 1.0)
                  + self.n_oil * (t_oil - self.t_oil_ideal) * (cth - 1.0)
                  - 0.5 * jnp.pi)
            phase = jnp.exp(1j * k * ab)   # (n_theta,)

            # Combined per-angle weights  (n_theta,)
            w0 = A0 * phase * scf
            w2 = A2 * phase * scf

            # Pixel-relative coordinates — avoid atan2(0,0) singularity
            dy_grid = Y - dy
            dx_grid = X - dx_p
            r2 = dy_grid ** 2 + dx_grid ** 2 + 1e-20
            r_pix = jnp.sqrt(r2)            # (ny, nx)

            # Bessel arguments  (ny, nx, n_theta)
            arg = k * self.n_oil * r_pix[..., None] * sth[None, None, :]
            bj0 = jax.scipy.special.jv(0, arg)   # (ny, nx, n_theta)
            bj2 = jax.scipy.special.jv(2, arg)

            # Radial integrals  (ny, nx)
            I0 = jnp.sum(bj0 * w0[None, None, :], axis=-1) * dt
            I2 = jnp.sum(bj2 * w2[None, None, :], axis=-1) * dt

            # Azimuthal factors without atan2 (use trig identities)
            cos2phi = (dx_grid ** 2 - dy_grid ** 2) / r2
            sin2phi = 2.0 * dx_grid * dy_grid / r2

            # Scattered field components
            Ex = -(k / 2.0) * 1j * (I0 + I2 * cos2phi)
            Ey = -(k / 2.0) * 1j * (I2 * sin2phi)

            return Ex, Ey

        # Sum over particles
        n_part = self.radii.shape[0]
        all_Ex, all_Ey = jax.vmap(particle_field)(jnp.arange(n_part))
        Ex_total = jnp.sum(all_Ex, axis=0)   # (ny, nx)
        Ey_total = jnp.sum(all_Ey, axis=0)

        # ── Intensity: I = |E_ref + Ex|² + |Ey|² ────────────────────────────
        # = Er² + 2 Er Re(Ex) + |Ex|² + |Ey|²  (always ≥ 0)
        intensity = (Er ** 2
                     + 2.0 * Er * jnp.real(Ex_total)
                     + jnp.abs(Ex_total) ** 2
                     + jnp.abs(Ey_total) ** 2)

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
