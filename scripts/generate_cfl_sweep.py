"""
Generate combined CFL sweep figure (1x2): SSP-RK3 left, Classical RK4 right.

No WSO lines — WSO comparison is in its own dedicated section.
Outputs fig_cfl_sweep.pdf.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rk4_extension'))

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from stencil_optimise import measure_order, W1_STANDARD, W2_STANDARD
from rk4_optimise import measure_order_rk4

# ========== Load stencils ==========
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
imgdir = os.path.join(os.path.dirname(__file__), '..', 'images')

# RK3 stencils
data_rk3_acc = np.load(os.path.join(data_dir, 'optimised_stencils.npz'))
W1_RK3_ACC, W2_RK3_ACC = data_rk3_acc['w1'], data_rk3_acc['w2']

data_rk3_stab = np.load(os.path.join(data_dir, 'optimised_stencils_stable.npz'))
W1_RK3_STAB, W2_RK3_STAB = data_rk3_stab['w1'], data_rk3_stab['w2']

# RK4 stencils
rk4_datadir = os.path.join(os.path.dirname(__file__), '..', 'rk4_extension', 'data')
data_rk4_acc = np.load(os.path.join(rk4_datadir, 'rk4_stencils.npz'))
W1_RK4_ACC, W2_RK4_ACC = data_rk4_acc['w1'], data_rk4_acc['w2']

data_rk4_stab = np.load(os.path.join(rk4_datadir, 'rk4_stencils_stable.npz'))
W1_RK4_STAB, W2_RK4_STAB = data_rk4_stab['w1'], data_rk4_stab['w2']

dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

# ========== Colour scheme ==========
TEAL   = '#1a8a8a'
ORANGE = '#e07020'
GREEN  = '#228B22'

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


def sweep_rk3(cfls, w1, w2):
    orders = []
    for cfl in cfls:
        try:
            o, e = measure_order(w1, w2, CFL=cfl, dt_vals=dt_full)
            orders.append(o if (np.all(np.isfinite(e)) and 0 < o < 4) else np.nan)
        except:
            orders.append(np.nan)
    return np.array(orders)


def sweep_rk4(cfls, w1, w2):
    orders = []
    for cfl in cfls:
        try:
            o, e = measure_order_rk4(w1, w2, CFL=cfl, dt_vals=dt_full)
            orders.append(o if (np.all(np.isfinite(e)) and 0 < o < 5) else np.nan)
        except:
            orders.append(np.nan)
    return np.array(orders)


# ========== CFL ranges ==========
cfls_rk3 = np.arange(0.1, 1.30, 0.05)
cfls_rk4 = np.arange(0.1, 1.55, 0.05)

print("SSP-RK3 CFL sweep...")
o_rk3_std  = sweep_rk3(cfls_rk3, W1_STANDARD, W2_STANDARD)
o_rk3_acc  = sweep_rk3(cfls_rk3, W1_RK3_ACC, W2_RK3_ACC)
o_rk3_stab = sweep_rk3(cfls_rk3, W1_RK3_STAB, W2_RK3_STAB)

print("Classical RK4 CFL sweep...")
o_rk4_std  = sweep_rk4(cfls_rk4, W1_STANDARD, W2_STANDARD)
o_rk4_acc  = sweep_rk4(cfls_rk4, W1_RK4_ACC, W2_RK4_ACC)
o_rk4_stab = sweep_rk4(cfls_rk4, W1_RK4_STAB, W2_RK4_STAB)

# ========== Plot ==========
print("Generating figure...")
fig, (ax_rk3, ax_rk4) = plt.subplots(1, 2, figsize=(13, 4.5))

# --- Left: SSP-RK3 ---
mask = np.isfinite(o_rk3_std)
ax_rk3.plot(cfls_rk3[mask], o_rk3_std[mask], 'o-', color=TEAL, lw=2, ms=5,
            label=r'\textrm{Standard}')
mask = np.isfinite(o_rk3_acc)
ax_rk3.plot(cfls_rk3[mask], o_rk3_acc[mask], 's-', color=ORANGE, lw=2, ms=5,
            label=r'\textrm{Optimised (acc.)}')
mask = np.isfinite(o_rk3_stab)
ax_rk3.plot(cfls_rk3[mask], o_rk3_stab[mask], '^-', color=GREEN, lw=2, ms=5,
            label=r'\textrm{Optimised (acc.+stab.)}')

ax_rk3.axhline(y=3, color='k', ls=':', alpha=0.4, label=r'\textrm{Target order} 3')
ax_rk3.axhline(y=2, color='gray', ls='--', alpha=0.2)

# Unstable regions
acc_last = cfls_rk3[np.isfinite(o_rk3_acc)][-1] if np.any(np.isfinite(o_rk3_acc)) else 0.7
std_last = cfls_rk3[np.isfinite(o_rk3_std)][-1] if np.any(np.isfinite(o_rk3_std)) else 1.1
ax_rk3.axvspan(acc_last, std_last, alpha=0.08, color='orange')
ax_rk3.annotate(r'\textrm{Opt. (acc.) unstable}', xy=(acc_last + 0.01, 1.2),
                fontsize=9, color='orange', alpha=0.9)
ax_rk3.axvspan(std_last, 1.4, alpha=0.1, color='red')
ax_rk3.annotate(r'\textrm{Std unstable}', xy=(std_last + 0.01, 1.05),
                fontsize=9, color='red', alpha=0.9)

ax_rk3.set_xlabel(r'\textrm{CFL number}')
ax_rk3.set_ylabel(r'\textrm{Measured convergence order}')
ax_rk3.set_title(r'(a) SSP-RK3')
ax_rk3.set_ylim([0.8, 3.6])
ax_rk3.set_xlim([0.05, 1.35])
ax_rk3.legend(loc='upper right', fontsize=8)

# --- Right: Classical RK4 ---
mask = np.isfinite(o_rk4_std)
ax_rk4.plot(cfls_rk4[mask], o_rk4_std[mask], 'o-', color=TEAL, lw=2, ms=5,
            label=r'\textrm{Standard}')
mask = np.isfinite(o_rk4_acc)
ax_rk4.plot(cfls_rk4[mask], o_rk4_acc[mask], 's-', color=ORANGE, lw=2, ms=5,
            label=r'\textrm{Optimised (acc.)}')
mask = np.isfinite(o_rk4_stab)
ax_rk4.plot(cfls_rk4[mask], o_rk4_stab[mask], '^-', color=GREEN, lw=2, ms=5,
            label=r'\textrm{Optimised (acc.+stab.)}')

ax_rk4.axhline(y=4, color='k', ls=':', alpha=0.4, label=r'\textrm{Target order} 4')
ax_rk4.axhline(y=2, color='gray', ls='--', alpha=0.2)

acc_last_4 = cfls_rk4[np.isfinite(o_rk4_acc)][-1] if np.any(np.isfinite(o_rk4_acc)) else 1.25
std_last_4 = cfls_rk4[np.isfinite(o_rk4_std)][-1] if np.any(np.isfinite(o_rk4_std)) else 1.3
ax_rk4.axvspan(acc_last_4, std_last_4, alpha=0.08, color='orange')
ax_rk4.annotate(r'\textrm{Opt. (acc.) unstable}', xy=(acc_last_4 + 0.01, 3.5),
                fontsize=9, color='orange', alpha=0.9)
ax_rk4.axvspan(std_last_4, 1.6, alpha=0.1, color='red')
ax_rk4.annotate(r'\textrm{Std unstable}', xy=(std_last_4 + 0.01, 3.5),
                fontsize=9, color='red', alpha=0.9)

ax_rk4.set_xlabel(r'\textrm{CFL number}')
ax_rk4.set_ylabel(r'\textrm{Measured convergence order}')
ax_rk4.set_title(r'(b) Classical RK4')
ax_rk4.set_ylim([1.0, 4.5])
ax_rk4.set_xlim([0.05, 1.55])
ax_rk4.legend(loc='upper right', fontsize=8)

plt.tight_layout()
outpath = os.path.join(imgdir, 'fig_cfl_sweep.pdf')
fig.savefig(outpath, bbox_inches='tight')
print(f"  Saved {outpath}")
