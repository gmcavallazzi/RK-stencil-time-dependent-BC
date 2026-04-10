"""
Head-to-head comparison: SSP-RK3 (standard/optimised stencils) vs WSO methods.

Evaluates convergence order on 1D linear advection and Burgers (MMS)
with time-dependent BCs for five method configurations.
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

from rk_parametrise import ssp_rk3, erk312, erk313, biswas_533
from rk_evaluate import measure_order as measure_order_rk
from rk_evaluate import measure_order_2d as measure_order_rk_2d
from stencil_optimise import (measure_order as measure_order_stencil,
                              W1_STANDARD, W2_STANDARD)
from validate_stencils import run_eval
from validate_stencils_2d import run_eval_2d

# Load optimised stencils
data = np.load(os.path.join(_DATA, 'optimised_stencils.npz'))
W1_OPT = data['w1']
W2_OPT = data['w2']

data_stab = np.load(os.path.join(_DATA, 'optimised_stencils_stable.npz'))
W1_STABLE = data_stab['w1']
W2_STABLE = data_stab['w2']

# ========== Configuration ==========
DT_FULL = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

TEAL   = '#1A8A8A'
ORANGE = '#E07020'
GREEN  = '#228B22'
BLUE   = '#3366CC'
RED    = '#CC3333'
PURPLE = '#8844AA'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ========== 1. WSO methods on advection (time-dep BC) ==========
print("=== WSO Comparison: 1D Advection (time-dep BC) ===")
results = {}

# SSP-RK3 + standard stencils (uses rk_evaluate with standard spatial op)
A_ssp, b_ssp, c_ssp = ssp_rk3()
o_ssp_std, e_ssp_std = measure_order_rk(A_ssp, b_ssp, c_ssp,
                                         pde='advection', bc_type='time_dependent',
                                         dt_vals=DT_FULL)
results['SSP-RK3 + std'] = (o_ssp_std, e_ssp_std)
print(f"  SSP-RK3 + std stencils:  order {o_ssp_std:.2f}")

# SSP-RK3 + optimised stencils
o_ssp_opt, e_ssp_opt = measure_order_stencil(W1_OPT, W2_OPT, dt_vals=DT_FULL)
results['SSP-RK3 + opt'] = (o_ssp_opt, e_ssp_opt)
print(f"  SSP-RK3 + opt (acc):     order {o_ssp_opt:.2f}")

# SSP-RK3 + stabilised stencils
o_ssp_stab, e_ssp_stab = measure_order_stencil(W1_STABLE, W2_STABLE, dt_vals=DT_FULL)
results['SSP-RK3 + stab'] = (o_ssp_stab, e_ssp_stab)
print(f"  SSP-RK3 + opt (acc+stab):order {o_ssp_stab:.2f}")

# ERK312, ERK313, (5,3,3) — all with standard stencils (rk_evaluate)
for name, fn in [('ERK312 (WSO 2)', erk312),
                 ('ERK313 (WSO 3)', erk313),
                 ('Biswas (5,3,3)', biswas_533)]:
    A, b, c = fn()
    o, e = measure_order_rk(A, b, c, pde='advection', bc_type='time_dependent',
                            dt_vals=DT_FULL)
    results[name] = (o, e)
    print(f"  {name:25s}  order {o:.2f}")


# ========== 2. Burgers MMS comparison ==========
print("\n=== WSO Comparison: Inviscid Burgers (MMS) ===")
results_bg = {}
_, e_bg_std, o_bg_std = run_eval(W1_STANDARD, W2_STANDARD, 'burgers_mms')
results_bg['SSP-RK3 + std'] = (o_bg_std, e_bg_std)
print(f"  SSP-RK3 + std:           order {o_bg_std:.2f}")

_, e_bg_opt, o_bg_opt = run_eval(W1_OPT, W2_OPT, 'burgers_mms')
results_bg['SSP-RK3 + opt'] = (o_bg_opt, e_bg_opt)
print(f"  SSP-RK3 + opt (acc):     order {o_bg_opt:.2f}")

_, e_bg_stab, o_bg_stab = run_eval(W1_STABLE, W2_STABLE, 'burgers_mms')
results_bg['SSP-RK3 + stab'] = (o_bg_stab, e_bg_stab)
print(f"  SSP-RK3 + opt (acc+stab):order {o_bg_stab:.2f}")

for name, fn in [('ERK312 (WSO 2)', erk312),
                 ('ERK313 (WSO 3)', erk313),
                 ('Biswas (5,3,3)', biswas_533)]:
    A, b, c = fn()
    o, e = measure_order_rk(A, b, c, pde='burgers_mms', bc_type='time_dependent', dt_vals=DT_FULL)
    results_bg[name] = (o, e)
    print(f"  {name:25s}  order {o:.2f}")

# ========== 3. 2D Advection comparison ==========
print("\n=== WSO Comparison: 2D Advection ===")
DT_2D = np.array([0.01, 0.005, 0.0025, 0.00125])
results_2d = {}

_, e_2d_std, o_2d_std = run_eval_2d(W1_STANDARD, W2_STANDARD, dt_vals=DT_2D)
results_2d['SSP-RK3 + std'] = (o_2d_std, e_2d_std)
print(f"  SSP-RK3 + std:           order {o_2d_std:.2f}")

_, e_2d_opt, o_2d_opt = run_eval_2d(W1_OPT, W2_OPT, dt_vals=DT_2D)
results_2d['SSP-RK3 + opt'] = (o_2d_opt, e_2d_opt)
print(f"  SSP-RK3 + opt (acc):     order {o_2d_opt:.2f}")

_, e_2d_stab, o_2d_stab = run_eval_2d(W1_STABLE, W2_STABLE, dt_vals=DT_2D)
results_2d['SSP-RK3 + stab'] = (o_2d_stab, e_2d_stab)
print(f"  SSP-RK3 + opt (acc+stab):order {o_2d_stab:.2f}")

for name, fn in [('ERK312 (WSO 2)', erk312),
                 ('ERK313 (WSO 3)', erk313),
                 ('Biswas (5,3,3)', biswas_533)]:
    A, b, c = fn()
    o, e = measure_order_rk_2d(A, b, c, dt_vals=DT_2D)
    results_2d[name] = (o, e)
    print(f"  {name:25s}  order {o:.2f}")


# [CFL SWEEP REMOVED TO CENTRALIZE ANALYSIS IN FIG 3]


# ========== Generate Figures ==========
print("\n=== Generating Figures ===")

fig = plt.figure(figsize=(11, 8.5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

# --- Panel 1: Advection ---
ax = ax1
o_s, e_s = results['SSP-RK3 + std']
o_o, e_o = results['SSP-RK3 + opt']
o_st, e_st = results['SSP-RK3 + stab']
ax.loglog(DT_FULL, e_s, 'o-', color=TEAL, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + std}} ($\mathcal{{O}}(\Delta t^{{{o_s:.2f}}})$)')
ax.loglog(DT_FULL, e_o, 's-', color=ORANGE, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_o:.2f}}})$)')
ax.loglog(DT_FULL, e_st, '^-', color=GREEN, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st:.2f}}})$)')

wso_styles = [
    ('ERK312 (WSO 2)', '^', BLUE,   1.2, 'ERK312 (WSO~2)'),
    ('ERK313 (WSO 3)', 'D', RED,    1.4, 'ERK313 (WSO~3)'),
    ('Biswas (5,3,3)', 'v', PURPLE, 1.6, 'Biswas (5,3,3) WSO~3'),
]
for key, marker, color, offset, label in wso_styles:
    o, e = results[key]
    ax.loglog(DT_FULL, e * offset, marker, color=color, ms=7,
              markerfacecolor='none', markeredgewidth=1.2, linestyle='none', zorder=6,
              label=rf'\textrm{{{label}}} ($\mathcal{{O}}(\Delta t^{{{o:.2f}}})$)')

ref2 = e_s[0] * (DT_FULL / DT_FULL[0])**2
ref3 = e_o[0] * (DT_FULL / DT_FULL[0])**3
ax.loglog(DT_FULL, ref2, 'k--', alpha=0.35, lw=1)
ax.loglog(DT_FULL, ref3, 'k:',  alpha=0.35, lw=1)
ax.text(DT_FULL[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(DT_FULL[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_ylabel(r'$L_2$ error')
ax.set_title(r'(a) Linear advection: $u_t + u_x = 0$')
ax.legend(fontsize=7)
xlim_conv = (4e-4, 1.5e-2)
ax.set_xlim(xlim_conv)

# --- Panel 2: Burgers ---
ax = ax2
o_s, e_s = results_bg['SSP-RK3 + std']
o_o, e_o = results_bg['SSP-RK3 + opt']
o_st, e_st = results_bg['SSP-RK3 + stab']
ax.loglog(DT_FULL, e_s, 'o-', color=TEAL, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + std}} ($\mathcal{{O}}(\Delta t^{{{o_s:.2f}}})$)')
ax.loglog(DT_FULL, e_o, 's-', color=ORANGE, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_o:.2f}}})$)')
ax.loglog(DT_FULL, e_st, '^-', color=GREEN, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st:.2f}}})$)')

for key, marker, color, offset, label in wso_styles:
    o, e = results_bg[key]
    ax.loglog(DT_FULL, e * offset, marker, color=color, ms=7,
              markerfacecolor='none', markeredgewidth=1.2, linestyle='none', zorder=6,
              label=rf'\textrm{{{label}}} ($\mathcal{{O}}(\Delta t^{{{o:.2f}}})$)')

ref2 = e_s[0] * (DT_FULL / DT_FULL[0])**2
ref3 = e_o[0] * (DT_FULL / DT_FULL[0])**3
ax.loglog(DT_FULL, ref2, 'k--', alpha=0.35, lw=1)
ax.loglog(DT_FULL, ref3, 'k:',  alpha=0.35, lw=1)
ax.text(DT_FULL[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(DT_FULL[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(b) Burgers (MMS): $u_t + u u_x = S$')
ax.legend(fontsize=7)
ax.set_xlim(xlim_conv)

# --- Panel 3: 2D Advection ---
ax = ax3
o_s, e_s = results_2d['SSP-RK3 + std']
o_o, e_o = results_2d['SSP-RK3 + opt']
o_st, e_st = results_2d['SSP-RK3 + stab']
ax.loglog(DT_2D, e_s, 'o-', color=TEAL, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + std}} ($\mathcal{{O}}(\Delta t^{{{o_s:.2f}}})$)')
ax.loglog(DT_2D, e_o, 's-', color=ORANGE, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_o:.2f}}})$)')
ax.loglog(DT_2D, e_st, '^-', color=GREEN, lw=2, ms=7, label=rf'\textrm{{SSP-RK3 + opt (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st:.2f}}})$)')

for key, marker, color, offset, label in wso_styles:
    o, e = results_2d[key]
    ax.loglog(DT_2D, e * offset, marker, color=color, ms=7,
              markerfacecolor='none', markeredgewidth=1.2, linestyle='none', zorder=6,
              label=rf'\textrm{{{label}}} ($\mathcal{{O}}(\Delta t^{{{o:.2f}}})$)')

ref2 = e_s[0] * (DT_2D / DT_2D[0])**2
ref3 = e_o[0] * (DT_2D / DT_2D[0])**3
ax.loglog(DT_2D, ref2, 'k--', alpha=0.35, lw=1)
ax.loglog(DT_2D, ref3, 'k:',  alpha=0.35, lw=1)
ax.text(DT_2D[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(DT_2D[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(c) 2D advection: $u_t + c_x u_x + c_y u_y = 0$')
ax.legend(fontsize=7)
ax.set_xlim(4e-4, 1.5e-2)

plt.tight_layout()
fig.savefig(os.path.join(_IMGS, 'fig_wso_comparison.pdf'), bbox_inches='tight')
print("  Saved fig_wso_comparison.pdf")


# ========== Summary Table ==========
print("\n" + "="*85)
print(f"{'Method':<30s}  {'Advection':<12s} {'Burgers':<12s} {'2D Adv':<12s}")
print("-" * 85)
for label in results:
    o1, _ = results[label]
    o2, _ = results_bg[label]
    o3, _ = results_2d[label]
    print(f"{label:<30s}  {o1:<12.2f} {o2:<12.2f} {o3:<12.2f}")
print("="*85)
