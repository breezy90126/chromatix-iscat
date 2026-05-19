"""iSCAT simulation module for Chromatix."""

from .mie import default_n_max, mie_coefficients, s1_s2
from .scattering import apply_mie_pupil, apply_rayleigh_pupil, shift_pupil
from .aberration import index_mismatch_phase
from .illumination import oblique_phase, fresnel_reference
from .background import psd_background_phase
from .microscope import iSCATMicroscope, simulate_z_stack
from .losses import mpg_nll, mpg_nll_per_pixel
from .fitting import fit_particles

__all__ = [
    "default_n_max",
    "mie_coefficients",
    "s1_s2",
    "apply_mie_pupil",
    "apply_rayleigh_pupil",
    "shift_pupil",
    "index_mismatch_phase",
    "oblique_phase",
    "fresnel_reference",
    "psd_background_phase",
    "iSCATMicroscope",
    "simulate_z_stack",
    "mpg_nll",
    "mpg_nll_per_pixel",
    "fit_particles",
]
