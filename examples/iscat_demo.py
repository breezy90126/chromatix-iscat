"""iSCAT simulation demo: forward model + z-stack + fitting.

Run with:
    python examples/iscat_demo.py
"""

import jax
import jax.numpy as jnp

from chromatix.iscat.microscope import iSCATMicroscope, simulate_z_stack
from chromatix.iscat.fitting import fit_particles
from chromatix.iscat.losses import mpg_nll

# ── Configuration ────────────────────────────────────────────────────────────
SHAPE = (64, 64)
DX = 65e-9       # 65 nm pixel size
WL = 532e-9      # green laser

# ── 1. Single-particle forward image ─────────────────────────────────────────
print("=== 1. Forward simulation ===")

microscope = iSCATMicroscope(
    radii=jnp.array([60e-9]),
    n_particles=jnp.array([1.5 + 0.0j]),
    positions_yx=jnp.array([[0.0, 0.0]]),
    z_particles=jnp.array([0.0]),
    z_focal=jnp.array(0.0),
    na=1.3,
    n_oil=1.518,
    n_glass=1.518,
    n_medium=1.33,
    wavelength=WL,
    scattering_model="rayleigh",
    reference_amplitude=0.05,
)

image = microscope(shape=SHAPE, dx=DX)
print(f"  Image shape : {image.shape}")
print(f"  Intensity   : min={float(jnp.min(image)):.4f}  max={float(jnp.max(image)):.4f}")

# ── 2. Two-particle scene ─────────────────────────────────────────────────────
print("\n=== 2. Two-particle scene ===")

two_particle = iSCATMicroscope(
    radii=jnp.array([60e-9, 40e-9]),
    n_particles=jnp.array([1.5 + 0.0j, 1.45 + 0.0j]),
    positions_yx=jnp.array([[400e-9, -300e-9], [-500e-9, 200e-9]]),
    z_particles=jnp.array([0.0, 100e-9]),
    z_focal=jnp.array(0.0),
    na=1.3,
    n_oil=1.518,
    n_glass=1.518,
    n_medium=1.33,
    wavelength=WL,
    scattering_model="rayleigh",
    reference_amplitude=0.05,
)

image2 = two_particle(shape=SHAPE, dx=DX)
print(f"  Image shape : {image2.shape}")
print(f"  Intensity   : min={float(jnp.min(image2)):.4f}  max={float(jnp.max(image2)):.4f}")

# ── 3. Z-stack ────────────────────────────────────────────────────────────────
print("\n=== 3. Z-stack (5 planes) ===")

z_vals = jnp.linspace(-300e-9, 300e-9, 5)
stack = simulate_z_stack(microscope, z_vals, shape=SHAPE, dx=DX)
print(f"  Stack shape : {stack.shape}")
for i, z in enumerate(z_vals):
    mn = float(jnp.min(stack[i]))
    mx = float(jnp.max(stack[i]))
    print(f"  z={float(z)*1e9:+6.0f} nm  min={mn:.4f}  max={mx:.4f}")

# ── 4. Gradient check ────────────────────────────────────────────────────────
print("\n=== 4. Gradient w.r.t. positions ===")

def total_intensity(pos):
    import equinox as eqx
    m = eqx.tree_at(lambda mc: mc.positions_yx, microscope, pos)
    return jnp.sum(m(shape=SHAPE, dx=DX))

grad = jax.grad(total_intensity)(microscope.positions_yx)
print(f"  Gradient shape : {grad.shape}")
print(f"  Gradient       : {grad}")
print(f"  All finite     : {bool(jnp.all(jnp.isfinite(grad)))}")

# ── 5. Fitting demo ───────────────────────────────────────────────────────────
print("\n=== 5. Fitting (50 Adam steps) ===")

# Ground truth image
true_pos = jnp.array([[200e-9, -150e-9]])
m_true = iSCATMicroscope(
    radii=jnp.array([60e-9]),
    n_particles=jnp.array([1.5 + 0.0j]),
    positions_yx=true_pos,
    z_particles=jnp.array([0.0]),
    z_focal=jnp.array(0.0),
    na=1.3, n_oil=1.518, n_glass=1.518, n_medium=1.33,
    wavelength=WL, scattering_model="rayleigh", reference_amplitude=0.05,
)
obs = m_true(shape=SHAPE, dx=DX)

# Initial guess at origin
m_init = iSCATMicroscope(
    radii=jnp.array([60e-9]),
    n_particles=jnp.array([1.5 + 0.0j]),
    positions_yx=jnp.zeros((1, 2)),
    z_particles=jnp.array([0.0]),
    z_focal=jnp.array(0.0),
    na=1.3, n_oil=1.518, n_glass=1.518, n_medium=1.33,
    wavelength=WL, scattering_model="rayleigh", reference_amplitude=0.05,
)

fitted, losses = fit_particles(
    m_init,
    obs,
    trainable_fields=["positions_yx"],
    gain=0.0,
    read_var=1.0,
    n_steps=50,
    lr=1e-6,
    call_kwargs={"shape": SHAPE, "dx": DX},
)

print(f"  Loss step 0  : {losses[0]:.4f}")
print(f"  Loss step 50 : {losses[-1]:.4f}")
print(f"  Reduced      : {bool(losses[-1] < losses[0])}")
print(f"  True pos (nm): {true_pos[0] * 1e9}")
print(f"  Fitted pos(nm): {fitted.positions_yx[0] * 1e9}")

print("\n=== Demo complete ===")
