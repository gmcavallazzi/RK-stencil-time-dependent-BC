"""
Gershgorin Circle Theorem analysis of the semi-discrete spatial operator
for standard, accuracy-optimised, and stability-augmented boundary stencils.

Computes Gershgorin discs for all rows of the operator matrix, highlights
the boundary rows (1 and 2), and derives analytical CFL upper bounds.
Generates a publication-quality figure overlaying discs on the SSP-RK3
stability region.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from stencil_optimise import W1_STANDARD, W2_STANDARD

# ========== Load stencils ==========
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
data_acc = np.load(os.path.join(data_dir, 'optimised_stencils.npz'))
W1_OPT = data_acc['w1']
W2_OPT = data_acc['w2']

data_stab = np.load(os.path.join(data_dir, 'optimised_stencils_stable.npz'))
W1_STABLE = data_stab['w1']
W2_STABLE = data_stab['w2']

# ========== Colours ==========
TEAL   = '#1A8A8A'
ORANGE = '#E07020'
GREEN  = '#228B22'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ========== Inline: build_spatial_operator (from stability_analysis.py) ==========
def build_spatial_operator(N, w1, w2):
    """
    Build the N×N spatial derivative operator matrix D such that
    du/dx ≈ D @ u / dx.
    """
    D = np.zeros((N, N))
    if N > 4:
        D[1, 0:5] = w1
        D[2, 0:5] = w2
    coeff = np.array([-2, 15, -60, 20, 30, -3]) / 60.0
    for i in range(3, N-2):
        if i-3 >= 0 and i+2 < N:
            D[i, i-3] = coeff[0]
            D[i, i-2] = coeff[1]
            D[i, i-1] = coeff[2]
            D[i, i]   = coeff[3]
            D[i, i+1] = coeff[4]
            D[i, i+2] = coeff[5]
    if N > 4:
        D[-2, -4] = 1/6
        D[-2, -3] = -1
        D[-2, -2] = 1/2
        D[-2, -1] = 1/3
    D[-1, -3] = 1/2
    D[-1, -2] = -2
    D[-1, -1] = 3/2
    return D


def ssp_rk3_stability_region(n_points=500):
    """Compute boundary of SSP-RK3 stability region {z : |R(z)| <= 1}."""
    theta = np.linspace(0, 2*np.pi, n_points)
    z_boundary = []
    for th in theta:
        direction = np.exp(1j * th)
        lo, hi = 0.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            z = mid * direction
            R = 1 + z + z**2/2 + z**3/6
            if abs(R) <= 1.0 + 1e-12:
                lo = mid
            else:
                hi = mid
        z_boundary.append(lo * direction)
    return np.array(z_boundary)


# ========== Gershgorin disc computation ==========
def gershgorin_discs(M):
    """
    Compute Gershgorin discs for matrix M.
    
    Returns
    -------
    centres : array of complex, shape (N,)
    radii : array of float, shape (N,)
    """
    centres = np.diag(M).astype(complex)
    radii = np.sum(np.abs(M), axis=1) - np.abs(np.diag(M))
    return centres, radii


def compute_amplification(D, dx, dt):
    """Compute max amplification factor for SSP-RK3."""
    eigs = np.linalg.eigvals(-D / dx)
    z_vals = dt * eigs
    R_vals = 1 + z_vals + z_vals**2/2 + z_vals**3/6
    return np.max(np.abs(R_vals))


def gershgorin_cfl_bound(D, dx):
    """
    Derive an analytical CFL upper bound from Gershgorin discs.
    
    For the operator -D/dx, the Gershgorin theorem guarantees all
    eigenvalues lie within the union of discs. For SSP-RK3 stability,
    we need dt * eigenvalues ⊂ stability region.
    
    The stability region extends ~2.51 along the negative real axis.
    Conservative bound: CFL ≤ r_max_neg_real / (max spectral bound)
    where max spectral bound is the Gershgorin spectral radius bound.
    """
    neg_D = -D / dx
    centres, radii = gershgorin_discs(neg_D)
    
    # r_max for SSP-RK3 along the negative real axis ≈ 2.5127
    r_max = 2.5127
    
    # The Gershgorin spectral radius bound (maximum |lambda| possible)
    max_spectral_bound = np.max(np.abs(centres) + radii)
    
    cfl_bound = r_max / (max_spectral_bound * dx)
    return cfl_bound, centres, radii, max_spectral_bound


# ========== Analysis ==========
N = 80
dx = 3.0 / (N - 1)

stencil_configs = [
    ('Standard',                 W1_STANDARD, W2_STANDARD, TEAL),
    ('Optimised (acc.)',         W1_OPT,      W2_OPT,      ORANGE),
    ('Optimised (acc.+stab.)',   W1_STABLE,   W2_STABLE,   GREEN),
]

print("=" * 75)
print("  Gershgorin Circle Theorem — Eigenvalue Bounding Analysis")
print("=" * 75)

stab_boundary = ssp_rk3_stability_region(n_points=500)

print(f"\n{'Stencil':<30s}  {'Gersh. CFL':>12s}  {'Emp. CFL':>10s}  {'Row 1 r':>10s}  {'Row 2 r':>10s}")
print("-" * 80)

all_data = []
for name, w1, w2, color in stencil_configs:
    D = build_spatial_operator(N, w1, w2)
    cfl_bound, centres, radii, max_spec = gershgorin_cfl_bound(D, dx)
    
    # Empirical CFL via eigenvalue sweep
    crit_cfl = 0.0
    for cfl_test in np.arange(0.05, 2.0, 0.01):
        dt_test = cfl_test * dx
        amp = compute_amplification(D, dx, dt_test)
        if amp > 1.0 + 1e-8:
            crit_cfl = cfl_test - 0.01
            break
        crit_cfl = cfl_test
    
    eigs = np.linalg.eigvals(-D / dx)
    
    print(f"  {name:<28s}  {cfl_bound:>12.3f}  {crit_cfl:>10.3f}  "
          f"{radii[1]:>10.4f}  {radii[2]:>10.4f}")
    
    all_data.append({
        'name': name, 'color': color,
        'D': D, 'centres': centres, 'radii': radii,
        'cfl_bound': cfl_bound, 'crit_cfl': crit_cfl,
        'eigs': eigs,
    })

print("=" * 80)


# ========== Generate Figure ==========
print("\n--- Generating figure ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

CFL_PLOT = 0.5

for idx, (data, ax) in enumerate(zip(all_data, axes)):
    name = data['name']
    color = data['color']
    eigs = data['eigs']
    D = data['D']
    
    dt_plot = CFL_PLOT * dx
    
    # Stability boundary
    ax.plot(stab_boundary.real, stab_boundary.imag, 'k-', lw=1.5)
    ax.fill(stab_boundary.real, stab_boundary.imag, alpha=0.04, color='gray')
    
    # Compute Gershgorin discs for -D/dx, then scale by dt
    neg_D_dx = -D / dx
    g_centres, g_radii = gershgorin_discs(neg_D_dx)
    
    # Interior discs (light gray)
    for i in range(3, N):
        z_c = dt_plot * g_centres[i]
        r = dt_plot * g_radii[i]
        if r > 1e-12:
            circ = plt.Circle((z_c.real, z_c.imag), r,
                              fill=True, facecolor='gray', alpha=0.06,
                              edgecolor='gray', linewidth=0.3)
            ax.add_patch(circ)
    
    # Row 0 is boundary (identity/zero), skip
    # Row 1 disc (solid)
    z_c1 = dt_plot * g_centres[1]
    r1 = dt_plot * g_radii[1]
    circ1 = plt.Circle((z_c1.real, z_c1.imag), r1, fill=False,
                        edgecolor=color, linewidth=2.5, linestyle='-')
    ax.add_patch(circ1)
    ax.plot(z_c1.real, z_c1.imag, '+', color=color, ms=10, mew=2)
    
    # Row 2 disc (dashed)  
    z_c2 = dt_plot * g_centres[2]
    r2 = dt_plot * g_radii[2]
    circ2 = plt.Circle((z_c2.real, z_c2.imag), r2, fill=False,
                        edgecolor=color, linewidth=2.5, linestyle='--')
    ax.add_patch(circ2)
    ax.plot(z_c2.real, z_c2.imag, 'x', color=color, ms=10, mew=2)
    
    # Actual eigenvalues
    z_eigs = dt_plot * eigs
    ax.scatter(z_eigs.real, z_eigs.imag, c=color, s=12, zorder=5, alpha=0.6)
    
    # Panel label
    panel_label = chr(ord('a') + idx)
    ax.set_title(rf'({panel_label}) \textrm{{{name}}}', fontsize=11)
    ax.set_xlabel(r'$\mathrm{Re}(z)$')
    if idx == 0:
        ax.set_ylabel(r'$\mathrm{Im}(z)$')
    ax.set_aspect('equal')
    ax.grid(True, ls='--', alpha=0.2)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='k', lw=1.5, label=r'\textrm{Stability boundary}'),
        Line2D([0], [0], color=color, lw=2.5, label=rf'\textrm{{Row 1}} ($r={r1:.2f}$)'),
        Line2D([0], [0], color=color, lw=2.5, ls='--',
               label=rf'\textrm{{Row 2}} ($r={r2:.2f}$)'),
        Line2D([0], [0], marker='o', color=color, ls='', ms=4,
               label=r'\textrm{Eigenvalues}'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower left',
              framealpha=0.9)
    
    # CFL bounds text
    ax.text(0.97, 0.97,
            rf'$\mathrm{{CFL}}_{{\mathrm{{Gersh.}}}} \leq {data["cfl_bound"]:.2f}$'
            '\n'
            rf'$\mathrm{{CFL}}_{{\mathrm{{emp.}}}} = {data["crit_cfl"]:.2f}$',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Consistent limits  
all_re = np.concatenate([CFL_PLOT * dx * d['eigs'].real for d in all_data])
all_im = np.concatenate([CFL_PLOT * dx * d['eigs'].imag for d in all_data])
pad = 0.8
for ax in axes:
    ax.set_xlim(min(all_re.min(), stab_boundary.real.min()) - pad,
                max(all_re.max(), stab_boundary.real.max()) + pad)
    ax.set_ylim(min(all_im.min(), stab_boundary.imag.min()) - pad,
                max(all_im.max(), stab_boundary.imag.max()) + pad)

plt.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '..', 'images', 'fig_gershgorin.pdf')
fig.savefig(outpath, bbox_inches='tight')
print(f"  Saved {outpath}")

# ========== Detailed row analysis ==========
print("\n--- Detailed Boundary Row Analysis ---")
for data in all_data:
    print(f"\n  {data['name']}:")
    D = data['D']
    neg_D_dx = -D / dx
    
    for row_idx in [1, 2]:
        row = neg_D_dx[row_idx, :]
        centre = row[row_idx]
        off_diag_sum = np.sum(np.abs(row)) - np.abs(centre)
        print(f"    Row {row_idx}: centre = {centre:.4f}, "
              f"radius = {off_diag_sum:.4f}, "
              f"extent = [{centre - off_diag_sum:.4f}, {centre + off_diag_sum:.4f}]")

print("\nDone.")
