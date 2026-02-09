"""iSCAT-specific modules and utilities."""

from .aberrations import compute_index_mismatch_phase
from .background import generate_psd_background
from .coherence import apply_coherence_pupil, apply_na_mask
from .illumination import apply_oblique_illumination
from .mie import compute_pi_tau, mie_coefficients
from .microscope import ISCATMicroscope, ro_iscat_average
from .reference import apply_fresnel_reference
from .scattering import (
    apply_vectorial_mie,
    compute_scattering_angle,
    mie_scatter_polarized,
    multi_particle_mie_scatter,
    s1_s2_precise,
)

__all__ = [
    "apply_coherence_pupil",
    "apply_fresnel_reference",
    "apply_na_mask",
    "apply_oblique_illumination",
    "apply_vectorial_mie",
    "compute_index_mismatch_phase",
    "compute_pi_tau",
    "compute_scattering_angle",
    "generate_psd_background",
    "ISCATMicroscope",
    "mie_coefficients",
    "mie_scatter_polarized",
    "multi_particle_mie_scatter",
    "ro_iscat_average",
    "s1_s2_precise",
]
