"""iSCAT Colab visualization demo: forward model, fitting, and GT comparison.

Run in Colab after:
    !pip install -q -e '.[dev]'
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import equinox as eqx

from chromatix.iscat.microscope import iSCATMicroscope, simulate_z_stack
from chromatix.iscat.fitting import fit_particles
from chromatix.iscat.losses import mpg_nll

# ── Global settings ───────────────────────────────────────────────────────────
SHAPE = (64, 64)
DX    = 65e-9        # 65 nm pixel
WL    = 532e-9       # green laser
NA    = 1.3
KEY   = jax.random.PRNGKey(0)

def _imshow(ax, img, title, vmin=None, vmax=None, cmap="gray"):
    im = ax.imshow(np.array(img), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, img.shape[1]*DX*1e9, 0, img.shape[0]*DX*1e9])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (nm)", fontsize=8)
    ax.set_ylabel("y (nm)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def make_scope(pos_yx, radii=None, ref_amp=0.05):
    radii = radii or [60e-9]
    n = len(radii)
    return iSCATMicroscope(
        radii=jnp.array(radii),
        n_particles=jnp.array([1.5 + 0.0j] * n),
        positions_yx=jnp.array(pos_yx, dtype=jnp.float32),
        z_particles=jnp.zeros(n),
        z_focal=jnp.array(0.0),
        na=NA, n_oil=1.518, n_glass=1.518, n_medium=1.33,
        wavelength=WL, scattering_model="rayleigh",
        reference_amplitude=ref_amp,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Single-particle PSF and iSCAT contrast
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering single-particle PSF …")

scope_center = make_scope([[0.0, 0.0]])
img_center   = scope_center(shape=SHAPE, dx=DX)

# Reference only (no particle)
scope_ref = iSCATMicroscope(
    radii=jnp.array([1e-12]),          # ~zero radius → no scatter
    n_particles=jnp.array([1.0 + 0.0j]),
    positions_yx=jnp.zeros((1, 2)),
    z_particles=jnp.zeros(1),
    z_focal=jnp.array(0.0),
    na=NA, n_oil=1.518, n_glass=1.518, n_medium=1.33,
    wavelength=WL, scattering_model="rayleigh", reference_amplitude=0.05,
)
img_ref = scope_ref(shape=SHAPE, dx=DX)

# iSCAT contrast = (I - I_ref) / I_ref
contrast = (img_center - img_ref) / (img_ref + 1e-12)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
_imshow(axes[0], img_ref,    "Reference (no particle)")
_imshow(axes[1], img_center, "With particle (center)")
_imshow(axes[2], contrast,   "iSCAT contrast", cmap="RdBu_r")
fig.suptitle("Single-particle PSF", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_psf.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → saved iscat_psf.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Z-stack (axial focus sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering z-stack …")

z_vals = jnp.linspace(-400e-9, 400e-9, 7)
stack  = simulate_z_stack(scope_center, z_vals, shape=SHAPE, dx=DX)
stack_ref = simulate_z_stack(scope_ref, z_vals, shape=SHAPE, dx=DX)
stack_contrast = (stack - stack_ref) / (stack_ref + 1e-12)

fig, axes = plt.subplots(2, 7, figsize=(18, 5))
vmin_c, vmax_c = float(stack_contrast.min()), float(stack_contrast.max())
for i, z in enumerate(z_vals):
    _imshow(axes[0, i], stack[i],          f"z={float(z)*1e9:+.0f}nm")
    _imshow(axes[1, i], stack_contrast[i], f"contrast", vmin=vmin_c, vmax=vmax_c, cmap="RdBu_r")
axes[0, 0].set_ylabel("Intensity\ny (nm)", fontsize=8)
axes[1, 0].set_ylabel("Contrast\ny (nm)", fontsize=8)
fig.suptitle("Z-stack (axial focus sweep)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_zstack.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → saved iscat_zstack.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Two-particle scene
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering two-particle scene …")

true_pos_2 = [[500e-9, -400e-9], [-600e-9, 300e-9]]
scope_2 = make_scope(true_pos_2, radii=[60e-9, 45e-9])
img_2 = scope_2(shape=SHAPE, dx=DX)
contrast_2 = (img_2 - img_ref) / (img_ref + 1e-12)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
_imshow(axes[0], img_2,       "Two-particle intensity")
_imshow(axes[1], contrast_2,  "Two-particle contrast", cmap="RdBu_r")
# Mark ground-truth positions
for ax in axes:
    for (py, px) in true_pos_2:
        ax.plot(px*1e9 + SHAPE[1]*DX*1e9/2,
                py*1e9 + SHAPE[0]*DX*1e9/2, "r+", ms=12, mew=2)
fig.suptitle("Two-particle iSCAT scene  (+ = GT position)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_two_particles.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → saved iscat_two_particles.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Inverse problem: fit particle position, compare with GT
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning inverse problem fitting …")

# Ground truth
TRUE_POS = jnp.array([[300e-9, -200e-9]])
scope_gt  = make_scope(TRUE_POS.tolist())
obs       = scope_gt(shape=SHAPE, dx=DX)

# Initial guess (origin)
INIT_POS  = jnp.zeros((1, 2))
scope_init = make_scope(INIT_POS.tolist())
img_init   = scope_init(shape=SHAPE, dx=DX)

# Fit
fitted_scope, losses = fit_particles(
    scope_init,
    obs,
    trainable_fields=["positions_yx"],
    gain=0.0,
    read_var=1.0,
    n_steps=300,
    lr=1e-9,
    call_kwargs={"shape": SHAPE, "dx": DX},
)
img_fitted    = fitted_scope(shape=SHAPE, dx=DX)
fitted_pos    = np.array(fitted_scope.positions_yx[0]) * 1e9   # nm
true_pos_nm   = np.array(TRUE_POS[0]) * 1e9
init_pos_nm   = np.array(INIT_POS[0]) * 1e9

print(f"  GT position   : y={true_pos_nm[0]:.1f} nm, x={true_pos_nm[1]:.1f} nm")
print(f"  Initial guess : y={init_pos_nm[0]:.1f} nm, x={init_pos_nm[1]:.1f} nm")
print(f"  Fitted        : y={fitted_pos[0]:.1f} nm, x={fitted_pos[1]:.1f} nm")
dist = float(jnp.linalg.norm(fitted_scope.positions_yx[0] - TRUE_POS[0])) * 1e9
print(f"  Residual error: {dist:.2f} nm")

# ── Plot: GT / Initial / Fitted / Residuals / Loss curve ─────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.4, wspace=0.4)

ax_gt      = fig.add_subplot(gs[0, 0])
ax_init    = fig.add_subplot(gs[0, 1])
ax_fitted  = fig.add_subplot(gs[0, 2])
ax_res_i   = fig.add_subplot(gs[0, 3])
ax_res_f   = fig.add_subplot(gs[0, 4])
ax_loss    = fig.add_subplot(gs[1, :3])
ax_profile = fig.add_subplot(gs[1, 3:])

vmin_i = float(min(obs.min(), img_init.min(), img_fitted.min()))
vmax_i = float(max(obs.max(), img_init.max(), img_fitted.max()))

_imshow(ax_gt,     obs,        "Ground truth (obs)", vmin=vmin_i, vmax=vmax_i)
_imshow(ax_init,   img_init,   "Initial sim",        vmin=vmin_i, vmax=vmax_i)
_imshow(ax_fitted, img_fitted, "Fitted sim",         vmin=vmin_i, vmax=vmax_i)

res_i = np.array(img_init  - obs)
res_f = np.array(img_fitted - obs)
vlim  = max(abs(res_i).max(), abs(res_f).max())
_imshow(ax_res_i, res_i, f"Residual (initial)\nRMS={np.sqrt((res_i**2).mean()):.4f}",
        vmin=-vlim, vmax=vlim, cmap="RdBu_r")
_imshow(ax_res_f, res_f, f"Residual (fitted)\nRMS={np.sqrt((res_f**2).mean()):.4f}",
        vmin=-vlim, vmax=vlim, cmap="RdBu_r")

# Mark GT position on all image panels
cx = true_pos_nm[1] + SHAPE[1]*DX*1e9/2
cy = true_pos_nm[0] + SHAPE[0]*DX*1e9/2
for ax in [ax_gt, ax_init, ax_fitted, ax_res_i, ax_res_f]:
    ax.plot(cx, cy, "r+", ms=14, mew=2.5, label="GT pos")
# Mark fitted position on fitted and residual panels
fx = fitted_pos[1] + SHAPE[1]*DX*1e9/2
fy = fitted_pos[0] + SHAPE[0]*DX*1e9/2
for ax in [ax_fitted, ax_res_f]:
    ax.plot(fx, fy, "gx", ms=14, mew=2.5, label="Fitted pos")

ax_fitted.legend(fontsize=7, loc="upper right")

# Loss curve
steps = np.arange(1, len(losses) + 1)
ax_loss.semilogy(steps, np.array(losses), "b-", lw=1.5)
ax_loss.set_xlabel("Adam step", fontsize=10)
ax_loss.set_ylabel("MPG NLL loss", fontsize=10)
ax_loss.set_title(f"Loss curve  ({losses[0]:.3f} → {losses[-1]:.3f})", fontsize=10)
ax_loss.grid(True, alpha=0.3)

# Central row profile comparison
mid = SHAPE[0] // 2
x_nm = (np.arange(SHAPE[1]) - SHAPE[1]//2) * DX * 1e9
ax_profile.plot(x_nm, np.array(obs)[mid],        "k-",  lw=1.5, label="GT (obs)")
ax_profile.plot(x_nm, np.array(img_init)[mid],   "r--", lw=1.5, label="Initial")
ax_profile.plot(x_nm, np.array(img_fitted)[mid], "g-",  lw=1.5, label="Fitted")
ax_profile.set_xlabel("x (nm)", fontsize=10)
ax_profile.set_ylabel("Intensity", fontsize=10)
ax_profile.set_title("Central row profile", fontsize=10)
ax_profile.legend(fontsize=8)
ax_profile.grid(True, alpha=0.3)

fig.suptitle(
    f"Inverse problem: particle localisation\n"
    f"GT=({true_pos_nm[0]:.0f}, {true_pos_nm[1]:.0f}) nm  |  "
    f"Fitted=({fitted_pos[0]:.1f}, {fitted_pos[1]:.1f}) nm  |  "
    f"Error={dist:.1f} nm",
    fontsize=12, fontweight="bold"
)
plt.savefig("iscat_fitting.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → saved iscat_fitting.png")

print("\n=== All plots saved ===")
