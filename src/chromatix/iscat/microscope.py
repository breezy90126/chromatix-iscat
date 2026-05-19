"""iSCAT microscope forward model."""

from typing import Optional, Sequence
import jax
import jax.numpy as jnp
import equinox as eqx

from chromatix.functional.sources import plane_wave
from chromatix.functional.propagation import asm_propagate

from .scattering import apply_mie_pupil, apply_rayleigh_pupil, shift_pupil
from .illumination import oblique_phase, fresnel_reference
from .background import psd_background_phase
from .aberration import index_mismatch_phase


def _na_cutoff(field, na: float):
    """Apply hard NA mask via FFT/IFFT."""
    from chromatix.core import VectorField
    wavelength = field.spectrum.central_wavelength
    fy = field.f_grid[..., 0, 0]
    fx = field.f_grid[..., 0, 1]
    f_norm = jnp.sqrt(fy ** 2 + fx ** 2)
    mask = (f_norm <= na / wavelength).astype(jnp.float32)
    # Apply mask in frequency domain
    u_fft = jnp.fft.fft2(field.u, axes=(0, 1))
    u_fft_shifted = jnp.fft.fftshift(u_fft, axes=(0, 1))
    u_fft_masked = u_fft_shifted * mask[..., jnp.newaxis]
    u_fft_back = jnp.fft.ifftshift(u_fft_masked, axes=(0, 1))
    u_filtered = jnp.fft.ifft2(u_fft_back, axes=(0, 1))
    return field.replace(u=u_filtered)


class iSCATMicroscope(eqx.Module):
    """iSCAT forward model as an Equinox module.

    Trainable parameters: radii, n_particles, positions_yx, z_particles, z_focal.
    Static parameters: na, n_oil, n_glass, n_medium, wavelength, t_oil_ideal,
                       scattering_model, n_max, reference_amplitude, sensor.
    """

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
        scattering_model: str = "mie",
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

    def _scatter_one_particle(self, incident_field, radius, m, pos_yx, z_p):
        """Compute scattered field for a single particle.

        Particle lateral position is encoded as a phase factor exp(-2j*pi*(fy*dy+fx*dx))
        via the FT shift theorem — applied AFTER scattering so it is not cancelled.
        """
        dy, dx_p = pos_yx[0], pos_yx[1]

        # Apply aberration phase for this particle's axial position
        ab_phase = index_mismatch_phase(
            incident_field,
            self.na,
            self.n_oil,
            self.n_medium,
            self.wavelength,
            self.z_focal,
            self.t_oil_ideal,
            z_particle=z_p,
        )
        field = incident_field.replace(u=incident_field.u * jnp.exp(1j * ab_phase))

        # Apply scattering amplitude (Mie or Rayleigh)
        if self.scattering_model == "mie":
            scattered = apply_mie_pupil(field, radius, m, self.n_medium, self.n_max)
        else:
            scattered = apply_rayleigh_pupil(field, radius, m, self.n_medium)

        # Encode lateral position: FT shift theorem — exp(-2j*pi*(fy*dy + fx*dx))
        return shift_pupil(scattered, -dy, -dx_p)

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
        """Simulate iSCAT image.

        Returns intensity image of shape (y, x).
        """
        # Build incident plane wave (vectorial, x-polarised)
        incident = plane_wave(
            shape=shape,
            dx=dx,
            spectrum=self.wavelength,
            amplitude=jnp.array([0.0, 0.0, 1.0]),  # x-polarised
            scalar=False,
            power=None,
        )

        # Apply oblique illumination
        if theta_inc != 0.0:
            incident = oblique_phase(incident, theta_inc, phi_inc, self.n_medium)

        # Propagate to focal plane (z_focal=0 is a no-op but JAX-traceable)
        incident = asm_propagate(incident, self.z_focal, self.n_medium, pad_width, mode="same")

        # Sum scattered fields from all particles
        n_particles = self.radii.shape[0]

        def scatter_i(i):
            return self._scatter_one_particle(
                incident,
                self.radii[i],
                self.n_particles[i],
                self.positions_yx[i],
                self.z_particles[i],
            )

        # vmap over particles
        scattered_all = jax.vmap(scatter_i)(jnp.arange(n_particles))
        # Sum over particles: scattered_all.u shape (n_particles, y, x, 3)
        scattered_sum_u = jnp.sum(scattered_all.u, axis=0)
        scattered_field = incident.replace(u=scattered_sum_u)

        # Reference field (Fresnel reflection)
        ref_field = fresnel_reference(incident, self.n_glass, self.n_medium)
        ref_field = ref_field.replace(u=ref_field.u * self.reference_amplitude)

        # Total field = reference + scattered
        total_u = ref_field.u + scattered_field.u
        total_field = incident.replace(u=total_u)

        # Optional background
        if include_background and key is not None:
            bg = psd_background_phase(key, shape, dx)
            total_field = total_field.replace(u=total_field.u * bg)

        # Intensity
        intensity = jnp.sum(jnp.abs(total_field.u) ** 2, axis=-1)  # (y, x)

        # Apply sensor if provided
        if self.sensor is not None:
            intensity = self.sensor(intensity)

        return intensity


def simulate_z_stack(microscope: iSCATMicroscope, z_focal_values, **kwargs):
    """Simulate a z-stack by vmapping over focal positions.

    Args:
        microscope: iSCATMicroscope instance
        z_focal_values: 1D array of focal depths to simulate
        **kwargs: passed to microscope.__call__

    Returns:
        intensity stack of shape (nz, y, x)
    """
    def sim_at_z(z):
        m = eqx.tree_at(lambda mc: mc.z_focal, microscope, z)
        return m(**kwargs)

    return jax.vmap(sim_at_z)(jnp.asarray(z_focal_values, dtype=jnp.float32))
