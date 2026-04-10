"""
Eigenvalue stability analysis of the full semi-discrete operator.

Constructs the spatial operator matrix D for standard, optimised, and
(optionally) stability-aware stencils, overlays scaled eigenvalues on
the SSP-RK3 stability region, and computes critical CFL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_HERE = os.path.dirname(__file__)
_DATA = os.path.join(_HERE, '..', 'data')
_IMGS = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMGS, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from rk_parametrise import ssp_rk3

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


# ========== Build Spatial Operator Matrix ==========

def build_spatial_operator(N, w1, w2):
    """
    Build the N×N spatial derivative operator matrix D such that
    du/dx ≈ D @ u / dx.

    Row 0 (boundary): zero (overwritten by BC).
    Row 1: w1 stencil (nodes 0..4).
    Row 2: w2 stencil (nodes 0..4).
    Rows 3..N-3: 5th-order upwind interior stencil.
    Row N-2: 3rd-order biased.
    Row N-1: 2nd-order backward.
    """
    D = np.zeros((N, N))

    # Row 0: boundary (not used in eigenvalue analysis — BC overwrite)
    # We set it to zero; the eigenvalue analysis considers the interior+closure.

    # Row 1: node-1 closure
    if N > 4:
        D[1, 0:5] = w1

    # Row 2: node-2 closure
    if N > 4:
        D[2, 0:5] = w2

    # Interior: 5th-order upwind
    coeff = np.array([-2, 15, -60, 20, 30, -3]) / 60.0
    for i in range(3, N-2):
        if i-3 >= 0 and i+2 < N:
            D[i, i-3] = coeff[0]
            D[i, i-2] = coeff[1]
            D[i, i-1] = coeff[2]
            D[i, i]   = coeff[3]
            D[i, i+1] = coeff[4]
            D[i, i+2] = coeff[5]

    # Outflow node N-2
    if N > 4:
        D[-2, -4] = 1/6
        D[-2, -3] = -1
        D[-2, -2] = 1/2
        D[-2, -1] = 1/3

    # Outflow node N-1
    D[-1, -3] = 1/2
    D[-1, -2] = -2
    D[-1, -1] = 3/2

    return D


def ssp_rk3_stability_region(n_points=500):
    """Compute boundary of SSP-RK3 stability region {z : |R(z)| <= 1}."""
    # R(z) = 1 + z + z^2/2 + z^3/6
    theta = np.linspace(0, 2*np.pi, n_points)
    # Find boundary by radial search at each angle
    z_boundary = []
    for th in theta:
        direction = np.exp(1j * th)
        # Binary search for max r such that |R(r*direction)| <= 1
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


def compute_amplification(D, dx, dt, A_rk, b_rk):
    """
    Compute max amplification factor |R(dt * lambda_i)| where lambda_i
    are eigenvalues of -D/dx (the semi-discrete advection operator).
    """
    eigs = np.linalg.eigvals(-D / dx)
    z_vals = dt * eigs

    # R(z) = 1 + z + z^2/2 + z^3/6 (for SSP-RK3)
    R_vals = 1 + z_vals + z_vals**2/2 + z_vals**3/6
    return np.max(np.abs(R_vals)), eigs, z_vals


# ========== Load Stencils ==========
from stencil_optimise import W1_STANDARD, W2_STANDARD

data = np.load(os.path.join(_DATA, 'optimised_stencils.npz'))
W1_OPT = data['w1']
W2_OPT = data['w2']

# Try loading stability-aware stencils if available
try:
    data_s = np.load(os.path.join(_DATA, 'optimised_stencils_stable.npz'))
    W1_STABLE = data_s['w1']
    W2_STABLE = data_s['w2']
    has_stable = True
except FileNotFoundError:
    has_stable = False


# ========== Analysis ==========
N = 80  # Grid size for eigenvalue analysis
dx = 3.0 / (N - 1)

D_std = build_spatial_operator(N, W1_STANDARD, W2_STANDARD)
D_opt = build_spatial_operator(N, W1_OPT, W2_OPT)

A_rk, b_rk, c_rk = ssp_rk3()
stab_boundary = ssp_rk3_stability_region()

print("=== Eigenvalue Stability Analysis ===\n")

# Compute eigenvalues
eigs_std = np.linalg.eigvals(-D_std / dx)
eigs_opt = np.linalg.eigvals(-D_opt / dx)

if has_stable:
    D_stable = build_spatial_operator(N, W1_STABLE, W2_STABLE)
    eigs_stable = np.linalg.eigvals(-D_stable / dx)

# Find critical CFL for each stencil set
print(f"{'Stencil':<20s}  {'Critical CFL':<15s}  {'Max |R| at CFL=0.5':<20s}")
print("-" * 60)

for name, D_mat in [('Standard', D_std), ('Optimised', D_opt)] + \
                   ([('Stability-aware', D_stable)] if has_stable else []):
    # Sweep CFL to find critical
    crit_cfl = 0.0
    for cfl_test in np.arange(0.05, 2.0, 0.01):
        dt_test = cfl_test * dx
        amp, _, _ = compute_amplification(D_mat, dx, dt_test, A_rk, b_rk)
        if amp > 1.0 + 1e-8:
            crit_cfl = cfl_test - 0.01
            break
        crit_cfl = cfl_test

    # Amplification at CFL=0.5
    dt05 = 0.5 * dx
    amp05, _, _ = compute_amplification(D_mat, dx, dt05, A_rk, b_rk)
    print(f"{name:<20s}  {crit_cfl:<15.3f}  {amp05:<20.6f}")


# ========== Long-time stability test ==========
print("\n=== Long-time Stability Test (T=10) ===")
from stencil_optimise import step_rk3, make_dudx_func

def run_longtime(w1, w2, CFL, T_end=10.0):
    dt = 0.005
    c_adv = 1.0
    dx_sim = dt * c_adv / CFL
    nx = int(3.0 / dx_sim) + 1
    x = np.linspace(0, 3.0, nx)
    dx_a = x[1] - x[0]
    dudx = make_dudx_func(w1, w2)
    g = lambda t: np.sin(np.pi * (-c_adv * t))
    u = np.sin(np.pi * x)
    t = 0.0
    max_norm = 0.0
    while t < T_end - 1e-12:
        u = step_rk3(u, t, min(dt, T_end - t), dx_a, c_adv, dudx, g)
        t += min(dt, T_end - t)
        norm = np.sqrt(np.mean(u**2))
        max_norm = max(max_norm, norm)
        if not np.all(np.isfinite(u)) or norm > 1e10:
            return False, max_norm
    return True, max_norm

for name, w1, w2 in [('Standard', W1_STANDARD, W2_STANDARD),
                      ('Optimised', W1_OPT, W2_OPT)]:
    for cfl in [0.3, 0.5, 0.7, 0.9, 1.0]:
        stable, max_n = run_longtime(w1, w2, cfl)
        status = "STABLE" if stable else "UNSTABLE"
        print(f"  {name:<12s}  CFL={cfl:.1f}  {status}  max_norm={max_n:.4f}")


# ========== Generate Figure ==========
print("\n=== Generating Figures ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

cfl_base = 0.5
dt_base = cfl_base * dx
z_std = dt_base * eigs_std
z_opt = dt_base * eigs_opt
if has_stable:
    z_stab = dt_base * eigs_stable

for j, ax in enumerate(axes):
    # Stability boundary
    ax.plot(stab_boundary.real, stab_boundary.imag, 'k-', lw=2,
            label=r'\textrm{SSP-RK3 boundary}')
    ax.fill(stab_boundary.real, stab_boundary.imag, alpha=0.05, color='gray')
    
    # Standard spectrum (appears on both)
    ax.scatter(z_std.real, z_std.imag, c=TEAL, s=30, zorder=5, alpha=0.8,
               label=rf'\textrm{{Standard}} (CFL = {cfl_base})')
               
    ax.axis('equal')
    ax.set_xlabel(r'$\mathrm{Re}(z)$')
    ax.set_ylabel(r'$\mathrm{Im}(z)$')
    ax.grid(True, ls='--', alpha=0.3)

# --- Left Panel: Standard vs Opt(acc) ---
ax = axes[0]
ax.set_title(r'(a) Standard vs. Optimised (acc.)')
ax.scatter(z_opt.real, z_opt.imag, c=ORANGE, s=30, zorder=5, alpha=0.8,
           label=rf'\textrm{{Optimised, acc.}} (CFL = {cfl_base})')

for cfl_hi, marker, alpha in [(0.7, 'x', 0.5), (0.9, '+', 0.35), (1.0, '.', 0.25)]:
    dt_hi = cfl_hi * dx
    z_hi = dt_hi * eigs_opt
    ax.scatter(z_hi.real, z_hi.imag, c=ORANGE, s=15, marker=marker, alpha=alpha,
               label=rf'\textrm{{Opt. acc., CFL = {cfl_hi}}}')
ax.legend(fontsize=8, loc='lower left', framealpha=0.9)

# --- Right Panel: Standard vs Opt(acc+stab) ---
ax = axes[1]
ax.set_title(r'(b) Standard vs. Optimised (acc.+stab.)')
if has_stable:
    ax.scatter(z_stab.real, z_stab.imag, c=GREEN, s=30, zorder=5, alpha=0.8,
               label=rf'\textrm{{Optimised, acc.+stab.}} (CFL = {cfl_base})')
    
    for cfl_hi, marker, alpha in [(0.7, 'x', 0.5), (0.9, '+', 0.35), (1.0, '.', 0.25)]:
        dt_hi = cfl_hi * dx
        z_hi = dt_hi * eigs_stable
        ax.scatter(z_hi.real, z_hi.imag, c=GREEN, s=15, marker=marker, alpha=alpha,
                   label=rf'\textrm{{Opt. acc.+stab., CFL = {cfl_hi}}}')
                   
ax.legend(fontsize=8, loc='lower left', framealpha=0.9)

plt.tight_layout()
fig.savefig(os.path.join(_IMGS, 'fig_stability.pdf'), bbox_inches='tight')
print("  Saved fig_stability.pdf")

print("\nDone.")

