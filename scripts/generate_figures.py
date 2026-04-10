"""
Generate all publication-quality figures for the report.
Color scheme: teal for standard, orange for optimised.
All text rendered via LaTeX (text.usetex=True).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from stencil_optimise import make_dudx_func, step_rk3, measure_order, W1_STANDARD, W2_STANDARD
from validate_stencils import run_eval
from validate_stencils_2d import run_eval_2d
from rk_parametrise import ssp_rk3, erk312, erk313, biswas_533
from rk_evaluate import measure_order as measure_order_rk

_HERE = os.path.dirname(__file__)
_DATA = os.path.join(_HERE, '..', 'data')
_IMGS = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMGS, exist_ok=True)

data_stable = np.load(os.path.join(_DATA, 'optimised_stencils_stable.npz'))
W1_STABLE = data_stable['w1']
W2_STABLE = data_stable['w2']

data = np.load(os.path.join(_DATA, 'optimised_stencils.npz'))
W1_OPT = data['w1']
W2_OPT = data['w2']

dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

TEAL = '#1a8a8a'
ORANGE = '#e07020'
GREEN = '#228B22'
BLUE   = '#3366CC'
RED    = '#CC3333'
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

# ==================================================================
# FIGURE 1: Solution snapshots (2, panels, no stencil diagram)
# ==================================================================
print("Figure 1: Solution snapshots...")

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))

c = 1.0
x = np.linspace(0, 3.0, 300)
exact = lambda x, t: np.sin(np.pi * (x - c*t))

ax = axes1[0]
for t_val, ls, alpha in [(0, '-', 1.0), (0.25, '--', 0.7), (0.5, '-.', 0.5)]:
    ax.plot(x, exact(x, t_val), ls, color='k', alpha=alpha,
            label=f'$t = {t_val}$')
ax.axvline(x=0, color=ORANGE, lw=2, alpha=0.7, label=r'$u(0,t) = g(t)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$u(x,t)$')
ax.set_title(r'(a) Exact solution: $u_t + u_x = 0$')
ax.legend(fontsize=9, framealpha=0.8)

ax = axes1[1]
t_arr = np.linspace(0, 0.5, 200)
g_sin = np.sin(np.pi * (-c * t_arr))
g_poly = (-t_arr)**3 - 2*(-t_arr)
g_multi = np.sin(2*np.pi*(-t_arr)) + 0.5*np.sin(6*np.pi*(-t_arr))
g_exp = np.exp(-0.5*(-t_arr)) * np.sin(4*np.pi*(-t_arr))

ax.plot(t_arr, g_sin, '-', color=TEAL, lw=2, label=r'$\sin(\pi t)$')
ax.plot(t_arr, g_poly, '--', color=ORANGE, lw=2, label=r'$t^3 - 2t$')
ax.plot(t_arr, g_multi, ':', color='#8855aa', lw=2, label=r'$\sin(2\pi t) + \frac{1}{2}\sin(6\pi t)$')
ax.plot(t_arr, g_exp, '-.', color='#666666', lw=2, label=r'$e^{-t/2}\sin(4\pi t)$')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$g(t)$')
ax.set_title(r'(b) Boundary conditions $u(0,t) = g(t)$')
ax.legend(fontsize=8)

plt.tight_layout()
fig1.savefig(os.path.join(_IMGS, 'fig_solution.pdf'), bbox_inches='tight')
print("  Saved fig_solution.pdf")

# ==================================================================
# FIGURE 2: Convergence validation (1D advection + Burgers + 2D)
# ==================================================================
print("Figure 2: Convergence validation...")

o_std, e_std = measure_order(W1_STANDARD, W2_STANDARD, CFL=0.5, dt_vals=dt_full)
o_opt, e_opt = measure_order(W1_OPT, W2_OPT, CFL=0.5, dt_vals=dt_full)
o_stab, e_stab = measure_order(W1_STABLE, W2_STABLE, CFL=0.5, dt_vals=dt_full)

_, err_std_bg, ord_std_bg = run_eval(W1_STANDARD, W2_STANDARD, 'burgers_mms')
_, err_opt_bg, ord_opt_bg = run_eval(W1_OPT, W2_OPT, 'burgers_mms')
_, err_stab_bg, ord_stab_bg = run_eval(W1_STABLE, W2_STABLE, 'burgers_mms')

dt_2d_std, err_2d_std, ord_2d_std = run_eval_2d(W1_STANDARD, W2_STANDARD)
dt_2d_opt, err_2d_opt, ord_2d_opt = run_eval_2d(W1_OPT, W2_OPT)
dt_2d_stab, err_2d_stab, ord_2d_stab = run_eval_2d(W1_STABLE, W2_STABLE)

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.2))

ax = axes2[0]
ax.loglog(dt_full, e_std, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{o_std:.2f}}})$)')
ax.loglog(dt_full, e_opt, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_opt:.2f}}})$)')
ax.loglog(dt_full, e_stab, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_stab:.2f}}})$)')
# Reference lines anchored to data (No labels)
ref2 = e_std[0] * (dt_full / dt_full[0])**2
ref3 = e_opt[0] * (dt_full / dt_full[0])**3
ax.loglog(dt_full, ref2, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_full, ref3, 'k:', alpha=0.35, lw=1)
ax.text(dt_full[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_full[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$'); ax.set_ylabel(r'$L_2$ error')
ax.set_title(r'(a) Linear advection: $u_t + u_x = 0$')
ax.legend(fontsize=8)

ax = axes2[1]
ax.loglog(dt_full, err_std_bg, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{ord_std_bg:.2f}}})$)')
ax.loglog(dt_full, err_opt_bg, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{ord_opt_bg:.2f}}})$)')
ax.loglog(dt_full, err_stab_bg, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{ord_stab_bg:.2f}}})$)')
ref2b = err_std_bg[0] * (dt_full / dt_full[0])**2
ref3b = err_opt_bg[0] * (dt_full / dt_full[0])**3
ax.loglog(dt_full, ref2b, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_full, ref3b, 'k:', alpha=0.35, lw=1)
ax.text(dt_full[-1]*0.9, ref2b[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_full[-1]*0.9, ref3b[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(b) Burgers (MMS): $u_t + u u_x = S$')
ax.legend(fontsize=8)

ax = axes2[2]
ax.loglog(dt_2d_std, err_2d_std, 'o-', color=TEAL, lw=2, ms=6,
          label=rf'\textrm{{Standard}} ($\mathcal{{O}}(\Delta t^{{{ord_2d_std:.2f}}})$)')
ax.loglog(dt_2d_opt, err_2d_opt, 's-', color=ORANGE, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.)}} ($\mathcal{{O}}(\Delta t^{{{ord_2d_opt:.2f}}})$)')
ax.loglog(dt_2d_stab, err_2d_stab, '^-', color=GREEN, lw=2, ms=6,
          label=rf'\textrm{{Optimised (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{ord_2d_stab:.2f}}})$)')
ref2c = err_2d_std[0] * (dt_2d_std / dt_2d_std[0])**2
ref3c = err_2d_opt[0] * (dt_2d_std / dt_2d_std[0])**3
ax.loglog(dt_2d_std, ref2c, 'k--', alpha=0.35, lw=1)
ax.loglog(dt_2d_std, ref3c, 'k:', alpha=0.35, lw=1)
ax.text(dt_2d_std[-1]*0.9, ref2c[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.text(dt_2d_std[-1]*0.9, ref3c[-1], r'$\mathcal{O}(\Delta t^3)$', fontsize=9, color='k', alpha=0.7, ha='right', va='center')
ax.set_xlabel(r'$\Delta t$')
ax.set_title(r'(c) 2D advection: $u_t + c_x u_x + c_y u_y = 0$')
ax.legend(fontsize=8)

xlim_conv = (4e-4, 1.5e-2)
for ax in axes2:
    ax.set_xlim(xlim_conv)

plt.tight_layout()
fig2.savefig(os.path.join(_IMGS, 'fig_convergence.pdf'), bbox_inches='tight')
print("  Saved fig_convergence.pdf")

# ==================================================================
# FIGURE 3: CFL generalisability + BC-shape independence
# ==================================================================
print("Figure 3: Generalisability...")

cfls = np.arange(0.1, 0.75, 0.05)
cfls_stab = np.arange(0.1, 1.25, 0.05)
orders_std, orders_opt, orders_stab = [], [], []

for cfl in cfls_stab:
    try:
        os, es = measure_order(W1_STANDARD, W2_STANDARD, CFL=cfl, dt_vals=dt_full)
        if np.all(np.isfinite(es)) and 0.0 < os < 4.0:
            orders_std.append(os)
        else:
            orders_std.append(np.nan)
    except:
        orders_std.append(np.nan)
    
    try:
        oo, eo = measure_order(W1_OPT, W2_OPT, CFL=cfl, dt_vals=dt_full)
        if cfl <= 0.75:
            orders_opt.append(oo if np.all(np.isfinite(eo)) else np.nan)
    except:
        if cfl <= 0.75:
            orders_opt.append(np.nan)
            
    try:
        ostab, estab = measure_order(W1_STABLE, W2_STABLE, CFL=cfl, dt_vals=dt_full)
        orders_stab.append(ostab if np.all(np.isfinite(estab)) else np.nan)
    except:
        orders_stab.append(np.nan)

orders_wso = {'ERK312': [], 'ERK313': [], '(5,3,3)': []}
cfl_vals_wso = np.arange(0.1, 1.6, 0.1)
for name, fn in [('ERK312', erk312), ('ERK313', erk313), ('(5,3,3)', biswas_533)]:
    A, b, c = fn()
    for cfl in cfl_vals_wso:
        try:
            o, e = measure_order_rk(A, b, c, pde='advection', bc_type='time_dependent', CFL=cfl, dt_vals=dt_full)
            orders_wso[name].append(o if (np.all(np.isfinite(e)) and o > 0) else np.nan)
        except:
            orders_wso[name].append(np.nan)
    orders_wso[name] = np.array(orders_wso[name])

def run_conv_bc(w1, w2, exact_func, g_func, CFL=0.5, T_end=0.5, c=1.0):
    dudx = make_dudx_func(w1, w2)
    errors = []
    for dt in dt_full:
        dx = dt * c / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        u = exact_func(x, 0.0)
        t = 0.0
        while t < T_end - 1e-12:
            u = step_rk3(u, t, min(dt, T_end-t), dx_a, c, dudx, g_func)
            t += min(dt, T_end-t)
        err = np.sqrt(np.mean((u - exact_func(x, T_end))**2))
        errors.append(err)
    errors = np.array(errors)
    mask = (errors > 1e-15) & np.isfinite(errors)
    if mask.sum() < 2:
        return 0.0, errors
    return np.polyfit(np.log(dt_full[mask]), np.log(errors[mask]), 1)[0], errors

bc_tests = [
    (r'$\sin(\pi \xi)$',
     lambda x, t: np.sin(np.pi*(x-t)), lambda t: np.sin(np.pi*(-t))),
    (r'$\xi^3 - 2\xi$',
     lambda x, t: (x-t)**3 - 2*(x-t), lambda t: (-t)**3 - 2*(-t)),
    (r'multi-freq.',
     lambda x, t: np.sin(2*np.pi*(x-t)) + 0.5*np.sin(6*np.pi*(x-t)),
     lambda t: np.sin(2*np.pi*(-t)) + 0.5*np.sin(6*np.pi*(-t))),
    (r'exp.-sine',
     lambda x, t: np.exp(-0.5*(x-t))*np.sin(4*np.pi*(x-t)),
     lambda t: np.exp(0.5*t)*np.sin(-4*np.pi*t)),
]

fig3, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
ax.plot(cfls_stab, orders_std, 'o-', color=TEAL, lw=2, ms=5, label=r'\textrm{Standard}')
ax.plot(cfls, orders_opt, 's-', color=ORANGE, lw=2, ms=5, label=r'\textrm{Optimised (acc.)}')
ax.plot(cfls_stab, orders_stab, '^-', color=GREEN, lw=2, ms=5, label=r'\textrm{Optimised (acc.+stab.)}')

wso_styles = [
    ('ERK312', '^-', BLUE, 'ERK312 (WSO~2)'),
    ('ERK313', 'D-', RED, 'ERK313 (WSO~3)'),
    ('(5,3,3)', 'v-', PURPLE, '(5,3,3) WSO~3'),
]
for name, fmt, color, label in wso_styles:
    ydata = orders_wso[name]
    mask = np.isfinite(ydata)
    ax.plot(cfl_vals_wso[mask], ydata[mask], fmt, color=color, lw=2, ms=5, label=rf'\textrm{{{label}}}')

ax.axhline(y=3, color='k', ls=':', alpha=0.4, label=r'\textrm{Target order} 3')
ax.axhline(y=2, color='gray', ls='--', alpha=0.2)

# Highlight Unstable Areas
ax.axvspan(0.72, 1.11, alpha=0.08, color='orange')
ax.annotate(r'\textrm{Opt. (acc.) Unstable}', xy=(0.73, 1.2), fontsize=10, color='orange', alpha=0.9)

ax.axvspan(1.11, 1.4, alpha=0.1, color='red')
ax.annotate(r'\textrm{Std Unstable}', xy=(1.13, 1.1), fontsize=10, color='red', alpha=0.9)

ax.set_xlabel(r'\textrm{CFL number}')
ax.set_ylabel(r'\textrm{Measured convergence order}')
ax.set_title(r'CFL generalisability')
ax.set_ylim([0.8, 3.4]); ax.set_xlim([0.05, 1.4])
ax.legend(loc='upper right', fontsize=7)

# Print BC evaluation results to console for LaTeX table migration
print("\n=== BC-Shape Independence Evaluation ===")
print(f"{'Boundary Condition':<25s} | {'Std':<8s} | {'Opt(acc)':<10s} | {'Opt(stab)':<10s}")
print("-" * 65)
bc_labels = []
for label, exact_f, g_f in bc_tests:
    o_s, _ = run_conv_bc(W1_STANDARD, W2_STANDARD, exact_f, g_f)
    o_o, _ = run_conv_bc(W1_OPT, W2_OPT, exact_f, g_f)
    o_stab, _ = run_conv_bc(W1_STABLE, W2_STABLE, exact_f, g_f)
    print(f"{label:<25s} | {o_s:<8.2f} | {o_o:<10.2f} | {o_stab:<10.2f}")
print("=" * 65)

plt.tight_layout()
fig3.savefig(os.path.join(_IMGS, 'fig_generalisability.pdf'), bbox_inches='tight')
print("  Saved fig_generalisability.pdf")

# ==================================================================
# FIGURE 4: Spectral analysis
# ==================================================================
print("Figure 4: Spectral analysis...")
from plot_spectral import offsets_1, offsets_2, modified_wavenumber

kdx = np.linspace(0, np.pi, 200)
kmod_1_std, disp_1_std = modified_wavenumber(W1_STANDARD, offsets_1, kdx)
kmod_2_std, disp_2_std = modified_wavenumber(W2_STANDARD, offsets_2, kdx)
kmod_1_opt, disp_1_opt = modified_wavenumber(W1_OPT, offsets_1, kdx)
kmod_2_opt, disp_2_opt = modified_wavenumber(W2_OPT, offsets_2, kdx)
kmod_1_stab, disp_1_stab = modified_wavenumber(W1_STABLE, offsets_1, kdx)
kmod_2_stab, disp_2_stab = modified_wavenumber(W2_STABLE, offsets_2, kdx)

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
fig4.savefig(os.path.join(_IMGS, 'fig_spectral.pdf'), bbox_inches='tight')
print("  Saved fig_spectral.pdf")

print("\nAll figures generated.")
