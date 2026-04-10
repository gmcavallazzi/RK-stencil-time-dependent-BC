"""
Validate the explicitly optimised boundary stencils on:
1. Linear Advection (re-confirming results)
2. Nonlinear advection (inviscid Burgers via Method of Manufactured Solutions)
"""

import numpy as np
import matplotlib.pyplot as plt
from stencil_optimise import step_rk3, make_dudx_func, W1_STANDARD, W2_STANDARD

# Load optimised stencils
data = np.load('optimised_stencils.npz')
W1_OPT = data['w1']
W2_OPT = data['w2']

dudx_std = make_dudx_func(W1_STANDARD, W2_STANDARD)
dudx_opt = make_dudx_func(W1_OPT, W2_OPT)

def run_convergence(pde_type, T_end=0.5):
    dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    errors_std = []
    errors_opt = []
    
    c = 1.0
    CFL = 0.5
    
    # Exact solution for both PDEs
    def exact_u(x, t):
        return np.sin(np.pi * (x - c * t))
        
def step_rk3_custom(u, t, dt, dx, rhs_func, g_func):
    u[0] = g_func(t)
    k1 = rhs_func(u, dx, t)
    u1 = u + dt * k1
    u1[0] = g_func(t + dt)

    k2 = rhs_func(u1, dx, t + dt)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * k2
    u2[0] = g_func(t + 0.5 * dt)

    k3 = rhs_func(u2, dx, t + 0.5 * dt)
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * k3
    u_new[0] = g_func(t + dt)
    return u_new


def run_eval(w1, w2, pde_type='advection', T_end=0.5):
    dudx_func = make_dudx_func(w1, w2)
    dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    errors = []
    
    CFL = 0.5
    c_adv = 1.0
    
    for dt in dt_vals:
        if pde_type == 'burgers_mms':
            # Local wave speed for burgers_mms is u = 2.0 + sin(...) which maxes at 3.0.
            # To maintain stability (effective CFL ~ 0.5), we need smaller dt per dx.
            # We will use c_eff = 3.0 to determine dx so dx is safe.
            dx = dt * 3.0 / CFL
        else:
            dx = dt * c_adv / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        
        if pde_type == 'advection':
            exact_u = lambda x, t: np.sin(np.pi * (x - c_adv * t))
            g = lambda t: exact_u(0.0, t)
            rhs = lambda u, _dx=dx_a, _t=0.0: -c_adv * dudx_func(u, _dx)
        elif pde_type == 'burgers_mms':
            exact_u = lambda x, t: 2.0 + np.sin(np.pi * (x - c_adv * t))
            g = lambda t: exact_u(0.0, t)
            # S = u_t + u u_x
            def rhs(u, _dx=dx_a, _t=0.0):
                phase = np.pi * (x - c_adv * _t)
                S = np.pi * np.cos(phase) * (2.0 - c_adv + np.sin(phase))
                # upwind works because u > 0 everywhere
                return -u * dudx_func(u, _dx) + S
                
        u = exact_u(x, 0.0)
        t = 0.0
        while t < T_end - 1e-12:
            step_dt = min(dt, T_end - t)
            u = step_rk3_custom(u, t, step_dt, dx_a, rhs, g)
            t += step_dt
            
        err = np.sqrt(np.mean((u - exact_u(x, T_end))**2))
        errors.append(err)
        
    errors = np.array(errors)
    order = np.polyfit(np.log(dt_vals), np.log(errors), 1)[0]
    return dt_vals, errors, order


if __name__ == '__main__':
    print("=== Linear Advection ===")
    _, err_std_adv, ord_std_adv = run_eval(W1_STANDARD, W2_STANDARD, 'advection')
    _, err_opt_adv, ord_opt_adv = run_eval(W1_OPT, W2_OPT, 'advection')
    
    print(f"Standard  Order: {ord_std_adv:.2f}")
    print(f"Optimised Order: {ord_opt_adv:.2f}\n")
    
    print("=== Nonlinear Inviscid Burgers (MMS) ===")
    dt_vals, err_std_bg, ord_std_bg = run_eval(W1_STANDARD, W2_STANDARD, 'burgers_mms')
    _, err_opt_bg, ord_opt_bg = run_eval(W1_OPT, W2_OPT, 'burgers_mms')
    
    print(f"Standard  Order: {ord_std_bg:.2f}")
    print(f"Optimised Order: {ord_opt_bg:.2f}\n")
    
    # Plotting with LaTeX formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Advection
    ax1.loglog(dt_vals, err_std_adv, 'o-', color='#1f77b4', label=f'Standard ($\\mathcal{{O}}(\\Delta t^{{{ord_std_adv:.2f}}})$)')
    ax1.loglog(dt_vals, err_opt_adv, 's-', color='#ff7f0e', label=f'Optimised ($\\mathcal{{O}}(\\Delta t^{{{ord_opt_adv:.2f}}})$)')
    ax1.loglog(dt_vals, 1e-1 * dt_vals**2, 'k--', alpha=0.5, label='$\\mathcal{O}(\\Delta t^2)$')
    ax1.loglog(dt_vals, 1e-1 * dt_vals**3, 'k:', alpha=0.5, label='$\\mathcal{O}(\\Delta t^3)$')
    ax1.set_xlabel('Time Step $\\Delta t$')
    ax1.set_ylabel('$L_2$ Error')
    ax1.set_title('Linear Advection: $u_t + c u_x = 0$')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    
    # Burgers
    ax2.loglog(dt_vals, err_std_bg, 'o-', color='#1f77b4', label=f'Standard ($\\mathcal{{O}}(\\Delta t^{{{ord_std_bg:.2f}}})$)')
    ax2.loglog(dt_vals, err_opt_bg, 's-', color='#ff7f0e', label=f'Optimised ($\\mathcal{{O}}(\\Delta t^{{{ord_opt_bg:.2f}}})$)')
    ax2.loglog(dt_vals, 2e-0 * dt_vals**2, 'k--', alpha=0.5, label='$\\mathcal{O}(\\Delta t^2)$')
    ax2.loglog(dt_vals, 2e-0 * dt_vals**3, 'k:', alpha=0.5, label='$\\mathcal{O}(\\Delta t^3)$')
    ax2.set_xlabel('Time Step $\\Delta t$')
    ax2.set_title('Inviscid Burgers (MMS): $u_t + u u_x = S(x,t)$')
    ax2.legend()
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('stencil_validation.pdf', bbox_inches='tight')
    plt.savefig('stencil_validation.pgf', bbox_inches='tight')
    print("Saved plots to stencil_validation.pdf and stencil_validation.pgf")
