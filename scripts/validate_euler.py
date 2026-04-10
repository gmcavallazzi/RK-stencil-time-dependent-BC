"""
Validate optimised boundary stencils on a coupled 1D hyperbolic system:
the Linearised Euler Equations with supersonic mean flow.

System:  q_t + A q_x = 0   where q = (rho', u', p')^T

With mean state rho0=1, u0=2.5, p0=1, gamma=1.4:
  A = [[u0,   rho0, 0        ],
       [0,    u0,   1/rho0   ],
       [0,    gamma*p0, u0   ]]

Eigenvalues of A: {u0-c0, u0, u0+c0} where c0=sqrt(gamma*p0/rho0)~1.18
With u0=2.5, all eigenvalues are positive (~1.32, 2.5, 3.68), making
the right-biased upwind stencil appropriate for all characteristics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from stencil_optimise import make_dudx_func, W1_STANDARD, W2_STANDARD
from rk_parametrise import ssp_rk3, classical_rk4, erk312, erk313, biswas_533, biswas_643

# ========== Load optimised stencils ==========
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
data_acc = np.load(os.path.join(data_dir, 'optimised_stencils.npz'))
W1_OPT = data_acc['w1']
W2_OPT = data_acc['w2']

data_stab = np.load(os.path.join(data_dir, 'optimised_stencils_stable.npz'))
W1_STABLE = data_stab['w1']
W2_STABLE = data_stab['w2']

# RK4 stencils
rk4_data_dir = os.path.join(os.path.dirname(__file__), '..', 'rk4_extension', 'data')
data_rk4_acc = np.load(os.path.join(rk4_data_dir, 'rk4_stencils.npz'))
W1_RK4_OPT = data_rk4_acc['w1']
W2_RK4_OPT = data_rk4_acc['w2']

data_rk4_stab = np.load(os.path.join(rk4_data_dir, 'rk4_stencils_stable.npz'))
W1_RK4_STABLE = data_rk4_stab['w1']
W2_RK4_STABLE = data_rk4_stab['w2']

# ========== Colour palette ==========
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

# ========== Physical parameters ==========
GAMMA = 1.4
RHO0 = 1.0
U0 = 2.5          # Supersonic mean flow (u0 > c0)
P0 = 1.0
C0 = np.sqrt(GAMMA * P0 / RHO0)  # sound speed ~ 1.183

# Flux Jacobian A for linearised Euler
A_FLUX = np.array([
    [U0,   RHO0,       0.0      ],
    [0.0,  U0,         1.0/RHO0 ],
    [0.0,  GAMMA*P0,   U0       ],
])

# Eigenvalues: u0-c0 > 0, u0, u0+c0  (all positive for supersonic flow)
EIGENVALUES = np.linalg.eigvals(A_FLUX)
EIGENVALUES = np.sort(EIGENVALUES.real)
MAX_WAVE_SPEED = np.max(np.abs(EIGENVALUES))


# ========== Exact solution ==========
# Use eigendecomposition: A = R @ diag(lambda) @ L
# Pick a smooth mode along the fastest characteristic (u0+c0) for the
# clearest test, but the exact solution propagates all 3 variables.
EVALS, R = np.linalg.eig(A_FLUX)
# Sort by eigenvalue
idx = np.argsort(EVALS.real)
EVALS = EVALS[idx].real
R = R[:, idx].real
L = np.linalg.inv(R)

def exact_solution(x, t, k=2*np.pi):
    """
    Smooth exact solution: superposition of all three characteristic waves.
    
    q(x,t) = sum_i  alpha_i * sin(k*(x - lambda_i * t)) * r_i
    
    We use equal amplitude for all three modes for a non-trivial coupled test.
    Returns shape (3, len(x)).
    """
    q = np.zeros((3, len(x)))
    for i in range(3):
        phase = np.sin(k * (x - EVALS[i] * t))
        q += np.outer(R[:, i], phase)
    return q


def exact_bc(t, k=2*np.pi):
    """Boundary values at x=0 for the exact solution."""
    return exact_solution(np.array([0.0]), t, k).flatten()


# ========== System RHS with parameterised boundary stencils ==========
def make_system_rhs(w1, w2):
    """Build RHS function for the linearised Euler system: dq/dt = -A dq/dx."""
    dudx_func = make_dudx_func(w1, w2)
    
    def rhs(q, dx, t=0.0):
        """q has shape (3, nx). Returns dqdt of same shape."""
        nvar, nx = q.shape
        dqdx = np.zeros_like(q)
        for v in range(nvar):
            dqdx[v, :] = dudx_func(q[v, :], dx)
        # dq/dt = -A @ dq/dx  (vectorised over grid points)
        dqdt = -A_FLUX @ dqdx  # (3,3) @ (3, nx) -> (3, nx)
        return dqdt
    
    return rhs


# ========== SSP-RK3 stepper for system ==========
def step_rk3_system(q, t, dt, dx, rhs_func, bc_func):
    """One SSP-RK3 step for the system q. q shape = (nvar, nx)."""
    q[:, 0] = bc_func(t)
    k1 = rhs_func(q, dx, t)
    
    q1 = q + dt * k1
    q1[:, 0] = bc_func(t + dt)
    
    k2 = rhs_func(q1, dx, t + dt)
    q2 = 0.75 * q + 0.25 * q1 + 0.25 * dt * k2
    q2[:, 0] = bc_func(t + 0.5 * dt)
    
    k3 = rhs_func(q2, dx, t + 0.5 * dt)
    q_new = (1.0/3.0) * q + (2.0/3.0) * q2 + (2.0/3.0) * dt * k3
    q_new[:, 0] = bc_func(t + dt)
    return q_new


# ========== Classical RK4 stepper for system ==========
def step_rk4_system(q, t, dt, dx, rhs_func, bc_func):
    """One classical RK4 step for the system q. q shape = (nvar, nx)."""
    q[:, 0] = bc_func(t)
    k1 = rhs_func(q, dx, t)
    
    q2 = q + 0.5 * dt * k1
    q2[:, 0] = bc_func(t + 0.5 * dt)
    k2 = rhs_func(q2, dx, t + 0.5 * dt)
    
    q3 = q + 0.5 * dt * k2
    q3[:, 0] = bc_func(t + 0.5 * dt)
    k3 = rhs_func(q3, dx, t + 0.5 * dt)
    
    q4 = q + dt * k3
    q4[:, 0] = bc_func(t + dt)
    k4 = rhs_func(q4, dx, t + dt)
    
    q_new = q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    q_new[:, 0] = bc_func(t + dt)
    return q_new


# ========== Generic RK stepper for system ==========
def step_generic_rk_system(q, t, dt, dx, rhs_func, bc_func, A_rk, b_rk, c_rk):
    """One step of a generic explicit RK method for the system."""
    s = len(b_rk)
    nvar, nx = q.shape
    
    K = np.zeros((s, nvar, nx))
    
    for i in range(s):
        q_stage = q.copy()
        # Vectorised accumulation of previous stages
        if i > 0:
            coeffs = A_rk[i, :i]
            nonzero = np.nonzero(coeffs)[0]
            for j in nonzero:
                q_stage += coeffs[j] * dt * K[j]
        q_stage[:, 0] = bc_func(t + c_rk[i] * dt)
        K[i] = rhs_func(q_stage, dx, t + c_rk[i] * dt)
    
    q_new = q.copy()
    for i in range(s):
        if b_rk[i] != 0:
            q_new += b_rk[i] * dt * K[i]
    q_new[:, 0] = bc_func(t + dt)
    return q_new


# ========== Convergence measurement ==========
def measure_euler_order(w1, w2, CFL=0.5, T_end=0.5, dt_vals=None,
                        rk_method='ssp_rk3'):
    """
    Run convergence study on linearised Euler with optimised stencils.
    Returns (order, dt_vals, errors).
    """
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    
    rhs_func = make_system_rhs(w1, w2)
    errors = []
    
    for dt in dt_vals:
        dx = dt * MAX_WAVE_SPEED / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        
        q = exact_solution(x, 0.0)
        bc = lambda t_val: exact_bc(t_val)
        
        t = 0.0
        blown = False
        while t < T_end - 1e-12:
            step_dt = min(dt, T_end - t)
            if rk_method == 'ssp_rk3':
                q = step_rk3_system(q, t, step_dt, dx_a, rhs_func, bc)
            elif rk_method == 'rk4':
                q = step_rk4_system(q, t, step_dt, dx_a, rhs_func, bc)
            else:
                A_rk, b_rk, c_rk = rk_method
                q = step_generic_rk_system(q, t, step_dt, dx_a, rhs_func, bc,
                                           A_rk, b_rk, c_rk)
            t += step_dt
            if not np.all(np.isfinite(q)):
                blown = True
                break
        
        if blown:
            errors.append(1e10)
        else:
            q_exact = exact_solution(x, T_end)
            err = np.sqrt(np.mean((q - q_exact)**2))
            errors.append(err)
    
    errors = np.array(errors)
    mask = (errors > 1e-15) & (errors < 1e9)
    if mask.sum() < 2:
        return 0.0, dt_vals, errors
    order = np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0]
    return order, dt_vals, errors


# ========== Main ==========
if __name__ == '__main__':
    DT_FULL = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    
    print("=" * 70)
    print("  Linearised Euler Equations — Convergence Study")
    print("=" * 70)
    print(f"  gamma={GAMMA}, rho0={RHO0}, u0={U0}, p0={P0}")
    print(f"  Sound speed c0={C0:.4f}")
    print(f"  Max wave speed = {MAX_WAVE_SPEED:.4f}")
    print(f"  Eigenvalues = {EIGENVALUES}")
    print()
    
    # ============ SSP-RK3 ============
    results_rk3 = {}
    
    print("--- SSP-RK3 + Standard stencils ---")
    o, dt_v, e = measure_euler_order(W1_STANDARD, W2_STANDARD, dt_vals=DT_FULL)
    results_rk3['SSP-RK3 + std'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    print("--- SSP-RK3 + Optimised (acc.) stencils ---")
    o, _, e = measure_euler_order(W1_OPT, W2_OPT, dt_vals=DT_FULL)
    results_rk3['SSP-RK3 + opt'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    print("--- SSP-RK3 + Optimised (acc.+stab.) stencils ---")
    o, _, e = measure_euler_order(W1_STABLE, W2_STABLE, dt_vals=DT_FULL)
    results_rk3['SSP-RK3 + stab'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    # WSO methods with standard stencils
    print("\n--- WSO Methods + Standard stencils ---")
    for name, fn in [('ERK312 (WSO 2)', erk312),
                     ('ERK313 (WSO 3)', erk313),
                     ('Biswas (5,3,3)', biswas_533)]:
        A_rk, b_rk, c_rk = fn()
        o, _, e = measure_euler_order(W1_STANDARD, W2_STANDARD, dt_vals=DT_FULL,
                                      rk_method=(A_rk, b_rk, c_rk), T_end=0.2)
        results_rk3[name] = (o, e)
        print(f"  {name:25s}  order {o:.2f}")
    
    # ============ Classical RK4 ============
    results_rk4 = {}
    
    print("\n--- RK4 + Standard stencils ---")
    o, _, e = measure_euler_order(W1_STANDARD, W2_STANDARD, dt_vals=DT_FULL,
                                  rk_method='rk4')
    results_rk4['RK4 + std'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    print("--- RK4 + Optimised (acc.) stencils ---")
    o, _, e = measure_euler_order(W1_RK4_OPT, W2_RK4_OPT, dt_vals=DT_FULL,
                                  rk_method='rk4')
    results_rk4['RK4 + opt'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    print("--- RK4 + Optimised (acc.+stab.) stencils ---")
    o, _, e = measure_euler_order(W1_RK4_STABLE, W2_RK4_STABLE, dt_vals=DT_FULL,
                                  rk_method='rk4')
    results_rk4['RK4 + stab'] = (o, e)
    print(f"  Order: {o:.2f}")
    
    # Order-4 WSO method for fair RK4 comparison
    print("\n--- Order-4 WSO Method + Standard stencils ---")
    A_643, b_643, c_643 = biswas_643()
    o, _, e = measure_euler_order(W1_STANDARD, W2_STANDARD, dt_vals=DT_FULL,
                                  rk_method=(A_643, b_643, c_643), T_end=0.2)
    results_rk4['(6,4,3) WSO 3'] = (o, e)
    print(f"  (6,4,3) WSO 3               order {o:.2f}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<30s}  {'Order':>10s}")
    print("-" * 45)
    for label, (o, _) in results_rk3.items():
        print(f"{label:<30s}  {o:>10.2f}")
    print()
    for label, (o, _) in results_rk4.items():
        print(f"{label:<30s}  {o:>10.2f}")
    print("=" * 70)
    
    # ========== Generate Figure (2 panels) ==========
    print("\n--- Generating figure ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # --- Panel (a): SSP-RK3 ---
    ax = ax1
    o_s, e_s = results_rk3['SSP-RK3 + std']
    o_o, e_o = results_rk3['SSP-RK3 + opt']
    o_st, e_st = results_rk3['SSP-RK3 + stab']
    
    ax.loglog(DT_FULL, e_s, 'o-', color=TEAL, lw=2, ms=7,
              label=rf'\textrm{{SSP-RK3 + std}} ($\mathcal{{O}}(\Delta t^{{{o_s:.2f}}})$)')
    ax.loglog(DT_FULL, e_o, 's-', color=ORANGE, lw=2, ms=7,
              label=rf'\textrm{{SSP-RK3 + opt (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_o:.2f}}})$)')
    ax.loglog(DT_FULL, e_st, '^-', color=GREEN, lw=2, ms=7,
              label=rf'\textrm{{SSP-RK3 + opt (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st:.2f}}})$)')
    
    # WSO methods
    markers_wso = {'ERK312 (WSO 2)': ('D', BLUE),
                   'ERK313 (WSO 3)': ('v', RED),
                   'Biswas (5,3,3)': ('P', PURPLE)}
    for name, (marker, color) in markers_wso.items():
        o_w, e_w = results_rk3[name]
        ax.loglog(DT_FULL, e_w, marker=marker, ls='--', color=color, lw=1.5, ms=6,
                  alpha=0.8,
                  label=rf'\textrm{{{name}}} ($\mathcal{{O}}(\Delta t^{{{o_w:.2f}}})$)')
    
    ref2 = e_s[0] * (DT_FULL / DT_FULL[0])**2
    ref3 = e_o[0] * (DT_FULL / DT_FULL[0])**3
    ax.loglog(DT_FULL, ref2, 'k--', alpha=0.35, lw=1)
    ax.loglog(DT_FULL, ref3, 'k:', alpha=0.35, lw=1)
    ax.text(DT_FULL[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    ax.text(DT_FULL[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$L_2$ error (all variables)')
    ax.set_title(r'(a) SSP-RK3: $\mathbf{q}_t + A\,\mathbf{q}_x = 0$')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlim(4e-4, 1.5e-2)
    
    # --- Panel (b): RK4 ---
    ax = ax2
    o_s4, e_s4 = results_rk4['RK4 + std']
    o_o4, e_o4 = results_rk4['RK4 + opt']
    o_st4, e_st4 = results_rk4['RK4 + stab']
    
    ax.loglog(DT_FULL, e_s4, 'o-', color=TEAL, lw=2, ms=7,
              label=rf'\textrm{{RK4 + std}} ($\mathcal{{O}}(\Delta t^{{{o_s4:.2f}}})$)')
    ax.loglog(DT_FULL, e_o4, 's-', color=ORANGE, lw=2, ms=7,
              label=rf'\textrm{{RK4 + opt (acc.)}} ($\mathcal{{O}}(\Delta t^{{{o_o4:.2f}}})$)')
    ax.loglog(DT_FULL, e_st4, '^-', color=GREEN, lw=2, ms=7,
              label=rf'\textrm{{RK4 + opt (acc.+stab.)}} ($\mathcal{{O}}(\Delta t^{{{o_st4:.2f}}})$)')
    
    # (6,4,3) WSO method — order 4, WSO 3
    o_643, e_643 = results_rk4['(6,4,3) WSO 3']
    ax.loglog(DT_FULL, e_643, marker='D', ls='--', color=BLUE, lw=1.5, ms=6,
              alpha=0.8,
              label=rf'\textrm{{(6,4,3) WSO 3}} ($\mathcal{{O}}(\Delta t^{{{o_643:.2f}}})$)')
    
    ref2 = e_s4[0] * (DT_FULL / DT_FULL[0])**2
    ref3 = e_o4[0] * (DT_FULL / DT_FULL[0])**3
    ax.loglog(DT_FULL, ref2, 'k--', alpha=0.35, lw=1)
    ax.loglog(DT_FULL, ref3, 'k:', alpha=0.35, lw=1)
    ax.text(DT_FULL[-1]*0.9, ref2[-1], r'$\mathcal{O}(\Delta t^2)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    ax.text(DT_FULL[-1]*0.9, ref3[-1], r'$\mathcal{O}(\Delta t^3)$',
            fontsize=9, color='k', alpha=0.7, ha='right', va='center')
    
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$L_2$ error (all variables)')
    ax.set_title(r'(b) Classical RK4: $\mathbf{q}_t + A\,\mathbf{q}_x = 0$')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlim(4e-4, 1.5e-2)
    
    plt.tight_layout()
    
    outpath = os.path.join(os.path.dirname(__file__), '..', 'images', 'fig_euler_convergence.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    print(f"  Saved {outpath}")
    
    print("\nDone.")
