"""iSCAT Colab visualization: PSF rings, z-stack, and fitting vs GT.

Run after:  pip install -q -e '.[dev]'
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from chromatix.iscat.microscope import iSCATMicroscope, simulate_z_stack
from chromatix.iscat.fitting import fit_particles

# ── Global settings ───────────────────────────────────────────────────────────
SHAPE  = (128, 128)
DX     = 65e-9        # 65 nm pixel
WL     = 532e-9       # green laser
NA     = 1.3

# reference_amplitude: sets |E_ref| relative to scattered field.
# r_Fresnel(water/glass) ≈ -0.066.  With REF_AMP=10:
#   |E_ref| ≈ 0.66  → iSCAT contrast ≈ 5–20% (realistic)
REF_AMP = 10.0


def make_scope(pos_yx, radii=None, ref_amp=REF_AMP):
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


def contrast(img, img_ref):
    """iSCAT contrast = (I - I_ref) / I_ref."""
    return (img - img_ref) / (img_ref + 1e-30)


def imshow(ax, data, title, cmap="gray", vmin=None, vmax=None, cbar=True):
    im = ax.imshow(np.array(data), cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, data.shape[1]*DX*1e9,
                           0, data.shape[0]*DX*1e9],
                   origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (nm)", fontsize=7)
    ax.set_ylabel("y (nm)", fontsize=7)
    ax.tick_params(labelsize=7)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# ══════════════════════════════════════════════════════════════════════════════
# Reference image (no particle — virtually zero radius)
# ══════════════════════════════════════════════════════════════════════════════
print("Computing reference image …")
scope_bare = iSCATMicroscope(
    radii=jnp.array([1e-15]),
    n_particles=jnp.array([1.5 + 0.0j]),
    positions_yx=jnp.zeros((1, 2)),
    z_particles=jnp.zeros(1),
    z_focal=jnp.array(0.0),
    na=NA, n_oil=1.518, n_glass=1.518, n_medium=1.33,
    wavelength=WL, scattering_model="rayleigh",
    reference_amplitude=REF_AMP,
)
img_ref = scope_bare(shape=SHAPE, dx=DX)   # uniform: |r_Fresnel * ref_amp|²


# ══════════════════════════════════════════════════════════════════════════════
# 1. Single-particle PSF + iSCAT contrast
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering single-particle PSF …")
scope_1 = make_scope([[0.0, 0.0]])
img_1   = scope_1(shape=SHAPE, dx=DX)
c_1     = contrast(img_1, img_ref)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
imshow(axes[0], img_ref, "Reference (no particle)", cmap="gray")
imshow(axes[1], img_1,   "With particle (60 nm, r=1.5)", cmap="gray")
vlim = float(jnp.abs(c_1).max())
imshow(axes[2], c_1,     f"iSCAT contrast (±{vlim:.3f})", cmap="RdBu_r",
       vmin=-vlim, vmax=vlim)
fig.suptitle("Single-particle iSCAT PSF  (Rayleigh, NA=1.3, λ=532 nm)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_psf.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → iscat_psf.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Z-stack (axial focus sweep)
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering z-stack …")
z_vals = jnp.linspace(-600e-9, 600e-9, 7)
stack     = simulate_z_stack(scope_1,    z_vals, shape=SHAPE, dx=DX)
stack_ref = simulate_z_stack(scope_bare, z_vals, shape=SHAPE, dx=DX)
stack_c   = jax.vmap(contrast)(stack, stack_ref)

vlim_c = float(jnp.abs(stack_c).max())

fig, axes = plt.subplots(2, 7, figsize=(19, 5.5))
for i, z in enumerate(z_vals):
    vmin_i = float(stack[i].min()); vmax_i = float(stack[i].max())
    imshow(axes[0, i], stack[i],   f"z={float(z)*1e9:+.0f} nm",
           vmin=vmin_i, vmax=vmax_i)
    imshow(axes[1, i], stack_c[i], "contrast",
           cmap="RdBu_r", vmin=-vlim_c, vmax=vlim_c)
axes[0, 0].set_ylabel("Intensity\ny (nm)", fontsize=8)
axes[1, 0].set_ylabel("Contrast\ny (nm)", fontsize=8)
fig.suptitle("Z-stack — axial focus sweep  (60 nm Rayleigh particle)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_zstack.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → iscat_zstack.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multi-particle scene
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering multi-particle scene …")
px = DX * SHAPE[1] * 0.5  # half field width in metres
py = DX * SHAPE[0] * 0.5
pos_multi = [
    [ 0.25*py,  0.30*px],
    [-0.35*py, -0.20*px],
    [ 0.10*py, -0.40*px],
    [-0.15*py,  0.45*px],
]
scope_multi = make_scope(pos_multi, radii=[60e-9, 45e-9, 70e-9, 50e-9])
img_multi   = scope_multi(shape=SHAPE, dx=DX)
c_multi     = contrast(img_multi, img_ref)

vlim_m = float(jnp.abs(c_multi).max())
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
imshow(axes[0], img_multi, "Four-particle intensity", cmap="gray")
imshow(axes[1], c_multi,   f"iSCAT contrast (±{vlim_m:.3f})",
       cmap="RdBu_r", vmin=-vlim_m, vmax=vlim_m)
for ax in axes:
    for (py_, px_) in pos_multi:
        ax.plot(px_*1e9 + SHAPE[1]*DX*1e9/2,
                py_*1e9 + SHAPE[0]*DX*1e9/2,
                "w+", ms=12, mew=2)
fig.suptitle("Multi-particle iSCAT scene  (white + = GT positions)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("iscat_multi.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → iscat_multi.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Inverse problem: fit lateral position, compare with GT
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning inverse problem fitting (300 Adam steps) …")

TRUE_POS  = jnp.array([[10 * DX, -8 * DX]])   # ~650 nm offset
INIT_POS  = jnp.zeros((1, 2))

scope_gt   = make_scope(TRUE_POS.tolist())
obs        = scope_gt(shape=SHAPE, dx=DX)
obs_c      = contrast(obs, img_ref)

scope_init = make_scope(INIT_POS.tolist())
img_init   = scope_init(shape=SHAPE, dx=DX)
init_c     = contrast(img_init, img_ref)

fitted_scope, losses = fit_particles(
    scope_init,
    obs,
    trainable_fields=["positions_yx"],
    gain=0.0,
    read_var=1e-4,
    n_steps=300,
    lr=1e-9,
    call_kwargs={"shape": SHAPE, "dx": DX},
)
img_fitted = fitted_scope(shape=SHAPE, dx=DX)
fitted_c   = contrast(img_fitted, img_ref)

true_nm   = np.array(TRUE_POS[0]) * 1e9
fitted_nm = np.array(fitted_scope.positions_yx[0]) * 1e9
err_nm    = float(jnp.linalg.norm(fitted_scope.positions_yx[0] - TRUE_POS[0])) * 1e9

print(f"  GT position  : y={true_nm[0]:.1f} nm,   x={true_nm[1]:.1f} nm")
print(f"  Fitted       : y={fitted_nm[0]:.1f} nm, x={fitted_nm[1]:.1f} nm")
print(f"  Error        : {err_nm:.2f} nm")
print(f"  Loss: {float(losses[0]):.4f} → {float(losses[-1]):.4f}")

# ── Big comparison figure ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.40)

ax_gt   = fig.add_subplot(gs[0, 0])
ax_init = fig.add_subplot(gs[0, 1])
ax_fit  = fig.add_subplot(gs[0, 2])
ax_ri   = fig.add_subplot(gs[0, 3])
ax_rf   = fig.add_subplot(gs[0, 4])
ax_diff = fig.add_subplot(gs[0, 5])
ax_loss = fig.add_subplot(gs[1, :3])
ax_prof = fig.add_subplot(gs[1, 3:])

# Shared colour scale for contrast images
vlim = float(max(jnp.abs(obs_c).max(),
                 jnp.abs(init_c).max(),
                 jnp.abs(fitted_c).max()))

for ax, data, title in [
    (ax_gt,   obs_c,    "GT (observed contrast)"),
    (ax_init, init_c,   "Initial sim contrast"),
    (ax_fit,  fitted_c, "Fitted sim contrast"),
]:
    imshow(ax, data, title, cmap="RdBu_r", vmin=-vlim, vmax=vlim)

# Residuals
res_i = np.array(img_init   - obs)
res_f = np.array(img_fitted - obs)
rlim  = max(abs(res_i).max(), abs(res_f).max())
rms_i = float(np.sqrt((res_i**2).mean()))
rms_f = float(np.sqrt((res_f**2).mean()))
imshow(ax_ri, res_i, f"Residual (initial)\nRMS={rms_i:.5f}",
       cmap="RdBu_r", vmin=-rlim, vmax=rlim)
imshow(ax_rf, res_f, f"Residual (fitted)\nRMS={rms_f:.5f}",
       cmap="RdBu_r", vmin=-rlim, vmax=rlim)

# Residual improvement map
diff = np.abs(res_i) - np.abs(res_f)
imshow(ax_diff, diff, "|res_init| − |res_fit|\n(blue=improved)",
       cmap="RdBu_r", vmin=-abs(diff).max(), vmax=abs(diff).max())

# Mark positions
cx = lambda px_nm: px_nm + SHAPE[1]*DX*1e9/2
cy = lambda py_nm: py_nm + SHAPE[0]*DX*1e9/2
for ax in [ax_gt, ax_init, ax_fit, ax_ri, ax_rf, ax_diff]:
    ax.plot(cx(true_nm[1]),   cy(true_nm[0]),   "r+",  ms=14, mew=2.5, label="GT")
    ax.plot(cx(fitted_nm[1]), cy(fitted_nm[0]),  "g×",  ms=14, mew=2.5, label="Fitted")
ax_fit.legend(fontsize=7, loc="upper right")

# Loss curve
steps = np.arange(1, len(losses) + 1)
ax_loss.semilogy(steps, np.array(losses), "b-", lw=1.5)
ax_loss.set_xlabel("Adam step")
ax_loss.set_ylabel("MPG NLL loss")
ax_loss.set_title(f"Loss  {float(losses[0]):.3f} → {float(losses[-1]):.3f}  "
                  f"(reduction {100*(1-float(losses[-1])/float(losses[0])):.1f}%)")
ax_loss.grid(True, alpha=0.3)

# Central-row profile
mid   = SHAPE[0] // 2
x_nm  = (np.arange(SHAPE[1]) - SHAPE[1]//2) * DX * 1e9
ax_prof.plot(x_nm, np.array(obs_c)[mid],    "k-",  lw=1.5, label="GT")
ax_prof.plot(x_nm, np.array(init_c)[mid],   "r--", lw=1.5, label="Initial")
ax_prof.plot(x_nm, np.array(fitted_c)[mid], "g-",  lw=1.5, label="Fitted")
ax_prof.axvline(true_nm[1],   color="r", ls=":",  lw=1, label=f"GT x={true_nm[1]:.0f}nm")
ax_prof.axvline(fitted_nm[1], color="g", ls=":",  lw=1, label=f"Fit x={fitted_nm[1]:.0f}nm")
ax_prof.set_xlabel("x (nm)")
ax_prof.set_ylabel("iSCAT contrast")
ax_prof.set_title("Central row profile")
ax_prof.legend(fontsize=7)
ax_prof.grid(True, alpha=0.3)

fig.suptitle(
    f"Inverse problem: lateral localisation\n"
    f"GT=({true_nm[0]:.0f}, {true_nm[1]:.0f}) nm  |  "
    f"Fitted=({fitted_nm[0]:.1f}, {fitted_nm[1]:.1f}) nm  |  "
    f"Error={err_nm:.1f} nm",
    fontsize=12, fontweight="bold",
)
plt.savefig("iscat_fitting.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → iscat_fitting.png")
print("\n=== All plots saved ===")
