"""
Generate publication-quality figures for the RK4 extension.

Same rcParams, colour scheme and style as scripts/generate_figures.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from stencil_optimise import (make_dudx_func, get_weights,
                              W1_STANDARD, W2_STANDARD)
from stencil_optimise_stable import build_spatial_operator
from rk4_optimise import step_rk4, measure_order_rk4
from rk4_optimise_stable import max_amplification_rk4
from rk4_evaluate import (measure_order_rk4_burgers, measure_order_rk4_2d)

# ========== Load stencils ==========
datadir = os.path.join(os.path.dirname(__file__), 'data')
imgdir = os.path.join(os.path.dirname(__file__), '..', 'images')

data_acc = np.load(os.path.join(datadir, 'rk4_stencils.npz'))
W1_ACC, W2_ACC = data_acc['w1'], data_acc['w2']

data_stab = np.load(os.path.join(datadir, 'rk4_stencils_stable.npz'))
W1_STAB, W2_STAB = data_stab['w1'], data_stab['w2']

dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
dt_2d = np.array([0.01, 0.005, 0.0025, 0.00125])

# ========== Colour scheme ==========
TEAL = '#1a8a8a'
ORANGE = '#e07020'
GREEN = '#228B22'
BLUE   = '#3366CC'
RED    = '#CC3333'
PURPLE = '#8844AA'

# ========== rcParams ==========
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ==================================================================
# FIGURE 1: Convergence (1x3) — advection, Burgers, 2D
# ==================================================================
print("Figure 1: RK4 convergence (3-panel)...")

o_std, e_std = measure_order_rk4(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
o_acc, e_acc = measure_order_rk4(W1_ACC, W2_ACC, dt_vals=dt_full)
o_stab, e_stab = measure_order_rk4(W1_STAB, W2_STAB, dt_vals=dt_full)

o_std_b, e_std_b = measure_order_rk4_burgers(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
o_acc_b, e_acc_b = measure_order_rk4_burgers(W1_ACC, W2_ACC, dt_vals=dt_full)
o_stab_b, e_stab_b = measure_order_rk4_burgers(W1_STAB, W2_STAB, dt_vals=dt_full)

o_std_2d, e_std_2d = measure_order_rk4_2d(W1_STANDARD, W2_STANDARD, dt_vals=dt_2d)
o_acc_2d, e_acc_2d = measure_order_rk4_2d(W1_ACC, W2_ACC, dt_vals=dt_2d)
o_stab_2d, e_stab_2d = measure_order_rk4_2d(W1_STAB, W2_STAB, dt_vals=dt_2d)

xlim_conv = (4e-4, 1.5e-2)

fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4.2))

# (a) 1D Advection
ax = axes1[0]
ax.loglog(dt_full, e_std, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{o_std:.2f}}})$)')
ax.loglog(dt_full, e_acc, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_acc:.2f}}})$)')
ax.loglog(dt_full, e_stab, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_stab:.2f}}})$)')
ref2 = e_std[0] * (dt_full / dt_full[0])**2
ref4 = e_acc[0] * (dt_full / dt_full[0])**4
ax.loglog(dt_full, ref2, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_full, ref4, 'k:', alpha=0.35, lw=1)
ax.text(dt_full[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_full[-1]*0.9, ref4[-1], r'$\mathcal{O}(\Delta t^4)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel(r'$L_2$ error')
ax.set_title(r'(a) Linear advection: $u_t + u_x = 0$')
ax.legend(fontsize=8)
ax.set_xlim(xlim_conv)

# (b) Burgers MMS
ax = axes1[1]
ax.loglog(dt_full, e_std_b, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{o_std_b:.2f}}})$)')
ax.loglog(dt_full, e_acc_b, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_acc_b:.2f}}})$)')
ax.loglog(dt_full, e_stab_b, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_stab_b:.2f}}})$)')
ref2b = e_std_b[0] * (dt_full / dt_full[0])**2
ref4b = e_acc_b[0] * (dt_full / dt_full[0])**4
ax.loglog(dt_full, ref2b, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_full, ref4b, 'k:', alpha=0.35, lw=1)
ax.text(dt_full[-1]*0.9, ref2b[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_full[-1]*0.9, ref4b[-1], r'$\mathcal{O}(\Delta t^4)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(b) Burgers (MMS): $u_t + u u_x = S$')
ax.legend(fontsize=8)
ax.set_xlim(xlim_conv)

# (c) 2D Advection
ax = axes1[2]
ax.loglog(dt_2d, e_std_2d, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{o_std_2d:.2f}}})$)')
ax.loglog(dt_2d, e_acc_2d, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_acc_2d:.2f}}})$)')
ax.loglog(dt_2d, e_stab_2d, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_stab_2d:.2f}}})$)')
ref2c = e_std_2d[0] * (dt_2d / dt_2d[0])**2
ref4c = e_acc_2d[0] * (dt_2d / dt_2d[0])**4
ax.loglog(dt_2d, ref2c, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_2d, ref4c, 'k:', alpha=0.35, lw=1)
ax.text(dt_2d[-1]*0.9, ref2c[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_2d[-1]*0.9, ref4c[-1], r'$\mathcal{O}(\Delta t^4)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(c) 2D advection: $u_t + c_x u_x + c_y u_y = 0$')
ax.legend(fontsize=8)
ax.set_xlim(xlim_conv)

plt.tight_layout()
fig1.savefig(os.path.join(imgdir, 'fig_rk4_convergence.pdf'), bbox_inches='tight')
print("  Saved fig_rk4_convergence.pdf")


# ==================================================================
# FIGURE 2: RK4 Stability region + eigenvalue spectrum
# ==================================================================
print("Figure 2: Stability region...")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

# RK4 stability boundary
r_vals = np.linspace(0, 4.0, 500)
theta_vals = np.linspace(0, 2*np.pi, 500)
RR, TT = np.meshgrid(r_vals, theta_vals)
ZZ = RR * np.exp(1j * TT)
RR4 = np.abs(1 + ZZ + ZZ**2/2 + ZZ**3/6 + ZZ**4/24)

N = 80
dx = 3.0 / (N - 1)
D_std = build_spatial_operator(N, W1_STANDARD, W2_STANDARD)
D_opt = build_spatial_operator(N, W1_ACC, W2_ACC)
D_stable = build_spatial_operator(N, W1_STAB, W2_STAB)

eigs_std = np.linalg.eigvals(-D_std / dx)
eigs_opt = np.linalg.eigvals(-D_opt / dx)
eigs_stable = np.linalg.eigvals(-D_stable / dx)

cfl_base = 0.5
dt_base = cfl_base * dx
z_std = dt_base * eigs_std
z_opt = dt_base * eigs_opt
z_stab = dt_base * eigs_stable

for j, ax in enumerate(axes2):
    ax.contour(RR * np.cos(TT), RR * np.sin(TT), RR4, levels=[1.0], colors='k', linewidths=2)
    ax.contourf(RR * np.cos(TT), RR * np.sin(TT), RR4, levels=[0, 1.0], colors=['#e0e0e0'], alpha=0.3)
    
    # Standard spectrum
    ax.scatter(z_std.real, z_std.imag, c=TEAL, s=30, zorder=5, alpha=0.8,
               label=rf'\textrm{{Standard}} (CFL = {cfl_base})')
               
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{Re}(z)$')
    ax.set_ylabel(r'$\mathrm{Im}(z)$')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlim([-5, 2.5])
    ax.set_ylim([-4.5, 4.5])

# --- Left Panel: Standard vs Opt(acc) ---
ax = axes2[0]
ax.set_title(r'(a) Standard vs. Optimised (acc.)')
ax.scatter(z_opt.real, z_opt.imag, c=ORANGE, s=30, zorder=5, alpha=0.8,
           label=rf'\textrm{{Optimised, acc.}} (CFL = {cfl_base})')

for cfl_hi, marker, alpha in [(0.7, 'x', 0.5), (0.9, '+', 0.35), (1.0, '.', 0.25)]:
    z_hi = (cfl_hi * dx) * eigs_opt
    ax.scatter(z_hi.real, z_hi.imag, c=ORANGE, s=15, marker=marker, alpha=alpha,
               label=rf'\textrm{{Opt. acc., CFL = {cfl_hi}}}')
ax.legend(fontsize=8, loc='lower left', framealpha=0.9)

# --- Right Panel: Standard vs Opt(acc+stab) ---
ax = axes2[1]
ax.set_title(r'(b) Standard vs. Optimised (acc.+stab.)')
ax.scatter(z_stab.real, z_stab.imag, c=GREEN, s=30, zorder=5, alpha=0.8,
           label=rf'\textrm{{Optimised, acc.+stab.}} (CFL = {cfl_base})')

for cfl_hi, marker, alpha in [(0.7, 'x', 0.5), (0.9, '+', 0.35), (1.0, '.', 0.25)]:
    z_hi = (cfl_hi * dx) * eigs_stable
    ax.scatter(z_hi.real, z_hi.imag, c=GREEN, s=15, marker=marker, alpha=alpha,
               label=rf'\textrm{{Opt. acc.+stab., CFL = {cfl_hi}}}')
ax.legend(fontsize=8, loc='lower left', framealpha=0.9)

plt.tight_layout()
fig2.savefig(os.path.join(imgdir, 'fig_rk4_stability.pdf'), bbox_inches='tight')
print("  Saved fig_rk4_stability.pdf")


# ==================================================================
# FIGURE 3: CFL sweep — measured order vs CFL
# ==================================================================
print("Figure 3: CFL sweep...")

cfls = np.arange(0.1, 1.5, 0.05)
orders_std, orders_acc, orders_stab = [], [], []

for cfl in cfls:
    try:
        o, e = measure_order_rk4(W1_STANDARD, W2_STANDARD, CFL=cfl, dt_vals=dt_full)
        orders_std.append(o if (np.all(np.isfinite(e)) and 0 < o < 5) else np.nan)
    except:
        orders_std.append(np.nan)

    try:
        o, e = measure_order_rk4(W1_ACC, W2_ACC, CFL=cfl, dt_vals=dt_full)
        orders_acc.append(o if (np.all(np.isfinite(e)) and 0 < o < 5) else np.nan)
    except:
        orders_acc.append(np.nan)

    try:
        o, e = measure_order_rk4(W1_STAB, W2_STAB, CFL=cfl, dt_vals=dt_full)
        orders_stab.append(o if (np.all(np.isfinite(e)) and 0 < o < 5) else np.nan)
    except:
        orders_stab.append(np.nan)

orders_std = np.array(orders_std)
orders_acc = np.array(orders_acc)
orders_stab = np.array(orders_stab)

fig3, ax3 = plt.subplots(1, 1, figsize=(8.5, 4.2))

mask_s = np.isfinite(orders_std)
mask_a = np.isfinite(orders_acc)
mask_st = np.isfinite(orders_stab)

ax3.plot(cfls[mask_s], orders_std[mask_s], 'o-', color=TEAL, lw=2, ms=5,
         label=r'\textrm{Standard}')
ax3.plot(cfls[mask_a], orders_acc[mask_a], 's-', color=ORANGE, lw=2, ms=5,
         label=r'\textrm{Optimised (acc.)}')
ax3.plot(cfls[mask_st], orders_stab[mask_st], '^-', color=GREEN, lw=2, ms=5,
         label=r'\textrm{Optimised (acc.+stab.)}')

ax3.axhline(y=4, color='k', ls=':', alpha=0.4, label=r'\textrm{Target order} 4')
ax3.axhline(y=2, color='gray', ls='--', alpha=0.2)

# Find last plotted points
last_acc_cfl = cfls[mask_a][-1] if np.any(mask_a) else 0.7
last_std_cfl = cfls[mask_s][-1] if np.any(mask_s) else 1.1

# Highlight Unstable Areas (from rk4 results)
ax3.axvspan(last_acc_cfl, 1.4, alpha=0.08, color='orange')
ax3.annotate(r'\textrm{Opt. (acc.) Unstable}', xy=(last_acc_cfl + 0.02, 3.25), fontsize=10, color='orange', alpha=0.9)

ax3.axvspan(last_std_cfl, 1.4, alpha=0.1, color='red')
ax3.annotate(r'\textrm{Std Unstable}', xy=(last_std_cfl + 0.02, 3.25), fontsize=10, color='red', alpha=0.9)

ax3.set_xlabel(r'\textrm{CFL number}')
ax3.set_ylabel(r'\textrm{Measured convergence order}')
ax3.set_title(r'RK4: CFL generalisability')
ax3.set_ylim([1.0, 4.5])
ax3.set_xlim([0.05, 1.4])
ax3.legend(loc='upper right', fontsize=7)

plt.tight_layout()
fig3.savefig(os.path.join(imgdir, 'fig_rk4_cfl_sweep.pdf'), bbox_inches='tight')
print("  Saved fig_rk4_cfl_sweep.pdf")

# ==================================================================
# FIGURE 4: Spectral analysis for RK4
# ==================================================================
print("Figure 4: Spectral analysis for RK4...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from plot_spectral import offsets_1, offsets_2, modified_wavenumber

kdx = np.linspace(0, np.pi, 200)
kmod_1_std, disp_1_std = modified_wavenumber(W1_STANDARD, offsets_1, kdx)
kmod_2_std, disp_2_std = modified_wavenumber(W2_STANDARD, offsets_2, kdx)
kmod_1_opt, disp_1_opt = modified_wavenumber(W1_ACC, offsets_1, kdx)
kmod_2_opt, disp_2_opt = modified_wavenumber(W2_ACC, offsets_2, kdx)
kmod_1_stab, disp_1_stab = modified_wavenumber(W1_STAB, offsets_1, kdx)
kmod_2_stab, disp_2_stab = modified_wavenumber(W2_STAB, offsets_2, kdx)

fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))

axes4[0,0].plot(kdx, kdx, 'k:', alpha=0.5, label=r'\textrm{Exact}')
axes4[0,0].plot(kdx, kmod_1_std, '-', color=TEAL, lw=2, label=r'\textrm{Standard}')
axes4[0,0].plot(kdx, kmod_1_opt, '-', color=ORANGE, lw=2, label=r'\textrm{Optimised (acc.)}')
axes4[0,0].plot(kdx, kmod_1_stab, '-', color=GREEN, lw=2, label=r'\textrm{Optimised (acc.+stab.)}')
axes4[0,0].set_xlabel(r'$\kappa \Delta x$'); axes4[0,0].set_ylabel(r'$\kappa_{\mathrm{mod}} \Delta x$')
axes4[0,0].set_title(r'(a) Node~1: Dispersion'); axes4[0,0].legend()

axes4[1,0].plot(kdx, np.zeros_like(kdx), 'k:', alpha=0.5, label=r'\textrm{Exact}~(0)')
axes4[1,0].plot(kdx, disp_1_std, '-', color=TEAL, lw=2, label=r'\textrm{Standard}')
axes4[1,0].plot(kdx, disp_1_opt, '-', color=ORANGE, lw=2, label=r'\textrm{Optimised (acc.)}')
axes4[1,0].plot(kdx, disp_1_stab, '-', color=GREEN, lw=2, label=r'\textrm{Optimised (acc.+stab.)}')
axes4[1,0].set_xlabel(r'$\kappa \Delta x$'); axes4[1,0].set_ylabel(r'$\operatorname{Re}(\hat{D})$')
axes4[1,0].set_title(r'(b) Node~1: Dissipation'); axes4[1,0].legend()

axes4[0,1].plot(kdx, kdx, 'k:', alpha=0.5, label=r'\textrm{Exact}')
axes4[0,1].plot(kdx, kmod_2_std, '-', color=TEAL, lw=2, label=r'\textrm{Standard}')
axes4[0,1].plot(kdx, kmod_2_opt, '-', color=ORANGE, lw=2, label=r'\textrm{Optimised (acc.)}')
axes4[0,1].plot(kdx, kmod_2_stab, '-', color=GREEN, lw=2, label=r'\textrm{Optimised (acc.+stab.)}')
axes4[0,1].set_xlabel(r'$\kappa \Delta x$'); axes4[0,1].set_ylabel(r'$\kappa_{\mathrm{mod}} \Delta x$')
axes4[0,1].set_title(r'(c) Node~2: Dispersion'); axes4[0,1].legend()

axes4[1,1].plot(kdx, np.zeros_like(kdx), 'k:', alpha=0.5, label=r'\textrm{Exact}~(0)')
axes4[1,1].plot(kdx, disp_2_std, '-', color=TEAL, lw=2, label=r'\textrm{Standard}')
axes4[1,1].plot(kdx, disp_2_opt, '-', color=ORANGE, lw=2, label=r'\textrm{Optimised (acc.)}')
axes4[1,1].plot(kdx, disp_2_stab, '-', color=GREEN, lw=2, label=r'\textrm{Optimised (acc.+stab.)}')
axes4[1,1].set_xlabel(r'$\kappa \Delta x$'); axes4[1,1].set_ylabel(r'$\operatorname{Re}(\hat{D})$')
axes4[1,1].set_title(r'(d) Node~2: Dissipation'); axes4[1,1].legend()

plt.tight_layout()
fig4.savefig(os.path.join(imgdir, 'fig_rk4_spectral.pdf'), bbox_inches='tight')
print("  Saved fig_rk4_spectral.pdf")

print("\nAll RK4 figures generated.")
