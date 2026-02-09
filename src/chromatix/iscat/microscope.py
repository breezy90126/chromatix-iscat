from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array, vmap

from chromatix.elements import FFLens, Propagate
from chromatix.field import VectorField

from .aberrations import compute_index_mismatch_phase
from .background import generate_psd_background
from .coherence import apply_coherence_pupil, apply_na_mask
from .field_utils import spatial_frequency_grid
from .illumination import apply_oblique_illumination
from .reference import apply_fresnel_reference
from .scattering import compute_scattering_angle, mie_scatter_polarized


class ISCATMicroscope(nn.Module):
    """iSCAT microscope model with vectorial Mie scattering and coherence."""

    na: float = 1.4
    n_oil: float = 1.518
    n_glass: float = 1.52
    n_medium: float = 1.33
    wavelength: float = 532e-9
    z_focal: float = 350e-9
    t_oil_ideal: float = 100e-6
    particle_radii: Array = jnp.array([50e-9, 60e-9])
    particle_n: Array = jnp.array([1.45 + 0.1j, 0.2 + 3.5j])
    particle_positions: Array = jnp.array([[0.0, 0.0], [10.0, 20.0]])
    particle_z: Array = jnp.array([350e-9, 400e-9])
    n_max: int = 20
    coherence_length: float = 5e-6
    objective_f: float = 10e-3

    def setup(self) -> None:
        self.propagate = Propagate(z=self.z_focal, n=self.n_medium)
        self.objective = FFLens(f=self.objective_f, n=self.n_medium, NA=self.na)

    def __call__(
        self,
        field: VectorField,
        theta_inc: float = 0.0,
        phi: float = 0.0,
        background_key: Array | None = None,
    ) -> Array:
        field = apply_oblique_illumination(
            field, theta_inc, phi, self.n_oil, self.wavelength
        )
        fx, fy = spatial_frequency_grid(field)

        def single_particle_path(
            field_i: VectorField,
            r: float,
            n_p: complex,
            pos: Array,
            z: float,
        ) -> VectorField:
            aberration_phase = compute_index_mismatch_phase(
                self.na,
                self.n_oil,
                self.n_glass,
                self.n_medium,
                self.wavelength,
                self.z_focal,
                self.t_oil_ideal,
                z,
                fx,
                fy,
            )
            phase = jnp.exp(1j * aberration_phase)
            phase = phase.reshape((1,) * (field_i.ndim - 4) + phase.shape + (1, 1))
            field_i = field_i.replace(u=field_i.u * phase)

            field_i = self.propagate(field_i)
            field_i = mie_scatter_polarized(
                field_i,
                r,
                n_p,
                self.n_medium,
                self.wavelength,
                (pos[0], pos[1]),
                fx,
                fy,
                self.n_max,
            )
            field_i = apply_na_mask(field_i, self.na, self.wavelength, self.n_medium)
            return field_i

        single_path_vmap = vmap(single_particle_path, in_axes=(None, 0, 0, 0, 0))
        scattered_per_particle = single_path_vmap(
            field,
            self.particle_radii,
            self.particle_n,
            self.particle_positions,
            self.particle_z,
        )

        total_scattered = field.replace(
            u=jnp.sum(scattered_per_particle.u, axis=0, keepdims=True)
        )

        bg_phase = generate_psd_background(
            field.spatial_shape, float(field._dx[0, 0]), key=background_key
        )
        bg_phase = bg_phase.reshape((1,) * (field.ndim - 4) + bg_phase.shape + (1, 1))
        background_field = field.replace(u=field.u * bg_phase)

        theta = compute_scattering_angle(fx, fy, self.wavelength, self.n_medium)
        reference_field = apply_fresnel_reference(
            field, theta, self.n_glass, self.n_medium
        )

        total_field = reference_field + background_field + total_scattered
        total_field = apply_coherence_pupil(
            total_field, self.coherence_length, self.wavelength, self.na
        )
        total_field = self.objective(total_field)
        return total_field.intensity


def ro_iscat_average(
    model: ISCATMicroscope,
    params: dict[str, Array],
    field: VectorField,
    theta_inc: float,
    phi_list: Array,
    background_key: Array | None = None,
) -> Array:
    """Average intensity over multiple azimuthal angles (RO-iSCAT)."""

    def single_phi(phi: float) -> Array:
        return model.apply({"params": params}, field, theta_inc, phi, background_key)

    i_per_phi = vmap(single_phi)(phi_list)
    return jnp.mean(i_per_phi, axis=0)
