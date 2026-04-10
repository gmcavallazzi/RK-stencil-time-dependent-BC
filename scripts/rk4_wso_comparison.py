"""
RK4 vs Biswas (6,4,3) WSO comparison.

Generates fig_rk4_wso_comparison.pdf: Classical RK4 with standard/acc/acc+stab
boundary stencils vs the order-4 WSO method Biswas(6,4,3) on three test problems:
  (a) 1D linear advection
  (b) 1D Burgers (MMS)
  (c) 2D linear advection
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rk4_extension'))

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from rk4_optimise import measure_order_rk4
from rk4_evaluate import measure_order_rk4_burgers, measure_order_rk4_2d
from rk_parametrise import biswas_643
from rk_evaluate import measure_order as measure_order_rk
from rk_evaluate import measure_order_2d as measure_order_rk_2d
from stencil_optimise import W1_STANDARD, W2_STANDARD

# ========== Load RK4 stencils ==========
datadir = os.path.join(os.path.dirname(__file__), '..', 'rk4_extension', 'data')
imgdir = os.path.join(os.path.dirname(__file__), '..', 'images')

data_acc = np.load(os.path.join(datadir, 'rk4_stencils.npz'))
W1_ACC, W2_ACC = data_acc['w1'], data_acc['w2']

data_stab = np.load(os.path.join(datadir, 'rk4_stencils_stable.npz'))
W1_STAB, W2_STAB = data_stab['w1'], data_stab['w2']

dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
dt_2d = np.array([0.01, 0.005, 0.0025, 0.00125])

# ========== Colour scheme ==========
TEAL   = '#1a8a8a'
ORANGE = '#e07020'
GREEN  = '#228B22'
PURPLE = '#8844AA'

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

# ========== RK4 convergence ==========
print("=== RK4 vs Biswas (6,4,3) Comparison ===")

# 1D Advection
o_std, e_std = measure_order_rk4(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
o_acc, e_acc = measure_order_rk4(W1_ACC, W2_ACC, dt_vals=dt_full)
o_stab, e_stab = measure_order_rk4(W1_STAB, W2_STAB, dt_vals=dt_full)
print(f"  1D Advection: std={o_std:.2f}, acc={o_acc:.2f}, stab={o_stab:.2f}")

# Biswas (6,4,3) on advection
A_643, b_643, c_643 = biswas_643()
o_wso, e_wso = measure_order_rk(A_643, b_643, c_643,
                                  pde='advection', bc_type='time_dependent',
                                  dt_vals=dt_full)
print(f"  Biswas (6,4,3) advection: order {o_wso:.2f}")

# 1D Burgers
o_std_b, e_std_b = measure_order_rk4_burgers(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
o_acc_b, e_acc_b = measure_order_rk4_burgers(W1_ACC, W2_ACC, dt_vals=dt_full)
o_stab_b, e_stab_b = measure_order_rk4_burgers(W1_STAB, W2_STAB, dt_vals=dt_full)
print(f"  Burgers: std={o_std_b:.2f}, acc={o_acc_b:.2f}, stab={o_stab_b:.2f}")

o_wso_b, e_wso_b = measure_order_rk(A_643, b_643, c_643,
                                      pde='burgers_mms', bc_type='time_dependent',
                                      dt_vals=dt_full)
print(f"  Biswas (6,4,3) Burgers: order {o_wso_b:.2f}")

# 2D Advection
o_std_2d, e_std_2d = measure_order_rk4_2d(W1_STANDARD, W2_STANDARD, dt_vals=dt_2d)
o_acc_2d, e_acc_2d = measure_order_rk4_2d(W1_ACC, W2_ACC, dt_vals=dt_2d)
o_stab_2d, e_stab_2d = measure_order_rk4_2d(W1_STAB, W2_STAB, dt_vals=dt_2d)
print(f"  2D Advection: std={o_std_2d:.2f}, acc={o_acc_2d:.2f}, stab={o_stab_2d:.2f}")

o_wso_2d, e_wso_2d = measure_order_rk_2d(A_643, b_643, c_643, dt_vals=dt_2d)
print(f"  Biswas (6,4,3) 2D: order {o_wso_2d:.2f}")

# ========== Generate Figure ==========
print("\nGenerating figure...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
xlim_conv = (4e-4, 1.5e-2)

# Helper to plot one panel
def plot_panel(ax, dt, e_s, e_a, e_st, e_w, o_s, o_a, o_st, o_w, title):
    ax.loglog(dt, e_s, 'o-', color=TEAL, lw=2, ms=7,
              label=rf'\textrm{{RK4 + Standard}} ($\mathcal{{O}}(\Delta t^{{{o_s:.2f}}})$)')
    ax.loglog(dt, e_a, 's-', color=ORANGE, lw=2, ms=7,
              label=rf'\textrm{{RK4 + Opt. (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_a:.2f}}})$)')
    ax.loglog(dt, e_st, '^-', color=GREEN, lw=2, ms=7,
              label=rf'\textrm{{RK4 + Opt. (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st:.2f}}})$)')
    ax.loglog(dt, e_w, 'v', color=PURPLE, ms=7,
              markerfacecolor='none', markeredgewidth=1.2, linestyle='none', zorder=6,
              label=rf'\textrm{{Biswas (6,4,3) WSO~3}} ($\mathcal{{O}}(\Delta t^{{{o_w:.2f}}})$)')

    ref2 = e_s[0] * (dt / dt[0])**2
    ref4 = e_a[0] * (dt / dt[0])**4
    ax.loglog(dt, ref2, 'k--', alpha=0.35, lw=1)
    ax.loglog(dt, ref4, 'k:', alpha=0.35, lw=1)
    ax.text(dt[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    ax.text(dt[-1]*0.9, ref4[-1], r'$\mathcal{O}(\Delta t^4)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    ax.set_xlabel(r'$\Delta t$')
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.set_xlim(xlim_conv)

plot_panel(axes[0], dt_full, e_std, e_acc, e_stab, e_wso,
           o_std, o_acc, o_stab, o_wso,
           r'(a) Linear advection: $u_t + u_x = 0$')
axes[0].set_ylabel(r'$L_2$ error')

plot_panel(axes[1], dt_full, e_std_b, e_acc_b, e_stab_b, e_wso_b,
           o_std_b, o_acc_b, o_stab_b, o_wso_b,
           r'(b) Burgers (MMS): $u_t + u u_x = S$')

plot_panel(axes[2], dt_2d, e_std_2d, e_acc_2d, e_stab_2d, e_wso_2d,
           o_std_2d, o_acc_2d, o_stab_2d, o_wso_2d,
           r'(c) 2D advection: $u_t + c_x u_x + c_y u_y = 0$')

plt.tight_layout()
outpath = os.path.join(imgdir, 'fig_rk4_wso_comparison.pdf')
fig.savefig(outpath, bbox_inches='tight')
print(f"  Saved {outpath}")
