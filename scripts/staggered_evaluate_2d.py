import numpy as np
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

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

def load_weights():
    # Load optimal
    path = os.path.join(os.path.dirname(__file__), '../data/staggered_weights_2d.npz')
    if os.path.exists(path):
        data = np.load(path)
        w_opt = data['w']
    else:
        # Fallback to the printed ones if file missing
        w_opt = np.array([-7.20186003, 28.12740539, -22.80923415, -10.99991677,  17.60900167, -1.40718326, -2.31821287])
        
    w_classic = np.array([4.0, -6.0, 4.0, -1.0, 0.0, 0.0, 0.0])
    return w_classic, w_opt

def ssp_rk3_step_2d(rho, u, v, t, dt, dx, dy, exact_rho, exact_u, exact_v, w_ghost):
    
    def compute_rhs(rho_val, u_val, v_val, t_val):
        drho = np.zeros_like(rho_val)
        du = np.zeros_like(u_val)
        dv = np.zeros_like(v_val)

        y_u = np.linspace(0.5*dy, 1.0 - 0.5*dy, rho_val.shape[1])
        u_bound_L = exact_u(0.0, y_u, t_val)
        u_bound_R = exact_u(1.0, y_u, t_val)

        # Extrapolate u ghost cells based on parametrised weights
        u_ghost_L = (w_ghost[0] * u_bound_L + w_ghost[1] * u_val[1, :] + 
                     w_ghost[2] * u_val[2, :] + w_ghost[3] * u_val[3, :] + 
                     w_ghost[4] * u_val[4, :] + w_ghost[5] * u_val[5, :] + w_ghost[6] * u_val[6, :])
                     
        u_ghost_R = (w_ghost[0] * u_bound_R + w_ghost[1] * u_val[-2, :] + 
                     w_ghost[2] * u_val[-3, :] + w_ghost[3] * u_val[-4, :] + 
                     w_ghost[4] * u_val[-5, :] + w_ghost[5] * u_val[-6, :] + w_ghost[6] * u_val[-7, :])

        u_pad = np.vstack([u_ghost_L, u_val, u_ghost_R])
        u_x = (-u_pad[3:] + 27.0 * u_pad[2:-1] - 27.0 * u_pad[1:-2] + u_pad[:-3]) / (24.0 * dx)

        v_yp2 = np.roll(v_val, -2, axis=1)
        v_yp1 = np.roll(v_val, -1, axis=1)
        v_ym1 = np.roll(v_val, 1, axis=1)
        v_y = (-v_yp2 + 27.0 * v_yp1 - 27.0 * v_val + v_ym1) / (24.0 * dy)

        drho = -(u_x + v_y)

        # Extrapolate rho ghost cells 
        rho_ghost_L = 4.0 * rho_val[0, :] - 6.0 * rho_val[1, :] + 4.0 * rho_val[2, :] - rho_val[3, :]
        rho_ghost_R = 4.0 * rho_val[-1, :] - 6.0 * rho_val[-2, :] + 4.0 * rho_val[-3, :] - rho_val[-4, :]

        rho_pad = np.vstack([rho_ghost_L, rho_val, rho_ghost_R])
        rho_x = (-rho_pad[3:] + 27.0 * rho_pad[2:-1] - 27.0 * rho_pad[1:-2] + rho_pad[:-3]) / (24.0 * dx)
        du[1:-1, :] = -rho_x

        rho_yp1 = np.roll(rho_val, -1, axis=1)
        rho_ym1 = np.roll(rho_val, 1, axis=1)
        rho_ym2 = np.roll(rho_val, 2, axis=1)
        rho_y = (-rho_yp1 + 27.0 * rho_val - 27.0 * rho_ym1 + rho_ym2) / (24.0 * dy)
        dv = -rho_y

        return drho, du, dv

    dr1, du1, dv1 = compute_rhs(rho, u, v, t)
    rho1 = rho + dt * dr1
    u1 = u + dt * du1
    v1 = v + dt * dv1
    y_u = np.linspace(0.5*dy, 1.0 - 0.5*dy, rho.shape[1])
    u1[0, :] = exact_u(0.0, y_u, t + dt)
    u1[-1, :] = exact_u(1.0, y_u, t + dt)

    dr2, du2, dv2 = compute_rhs(rho1, u1, v1, t + dt)
    rho2 = 0.75 * rho + 0.25 * rho1 + 0.25 * dt * dr2
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * du2
    v2 = 0.75 * v + 0.25 * v1 + 0.25 * dt * dv2
    u2[0, :] = exact_u(0.0, y_u, t + 0.5 * dt)
    u2[-1, :] = exact_u(1.0, y_u, t + 0.5 * dt)

    dr3, du3, dv3 = compute_rhs(rho2, u2, v2, t + 0.5 * dt)
    rho_new = (1.0/3.0) * rho + (2.0/3.0) * rho2 + (2.0/3.0) * dt * dr3
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * du3
    v_new = (1.0/3.0) * v + (2.0/3.0) * v2 + (2.0/3.0) * dt * dv3
    u_new[0, :] = exact_u(0.0, y_u, t + dt)
    u_new[-1, :] = exact_u(1.0, y_u, t + dt)

    return rho_new, u_new, v_new

def run_simulation(N, CFL, T_end, w_ghost):
    L = 1.0
    dx = L / N
    dy = L / N
    dt = CFL * dx
    
    x_1d = np.linspace(0.5*dx, L - 0.5*dx, N)
    y_1d = np.linspace(0.5*dy, L - 0.5*dy, N)
    x_rho, y_rho = np.meshgrid(x_1d, y_1d, indexing='ij')

    x_u_1d = np.linspace(0, L, N+1)
    x_u, y_u = np.meshgrid(x_u_1d, y_1d, indexing='ij')

    x_v_1d = np.linspace(0.5*dx, L - 0.5*dx, N)
    y_v_1d = np.linspace(0, L, N, endpoint=False)
    x_v, y_v = np.meshgrid(x_v_1d, y_v_1d, indexing='ij')

    def exact_rho(x, y, t): return np.sin(2 * np.pi * (x - t))
    def exact_u(x, y, t): return np.sin(2 * np.pi * (x - t))
    def exact_v(x, y, t): return np.zeros_like(x)

    rho = exact_rho(x_rho, y_rho, 0)
    u = exact_u(x_u, y_u, 0)
    v = exact_v(x_v, y_v, 0)
    
    t = 0.0
    while t < T_end - 1e-12:
        dt_step = min(dt, T_end - t)
        rho, u, v = ssp_rk3_step_2d(rho, u, v, t, dt_step, dx, dy, exact_rho, exact_u, exact_v, w_ghost)
        t += dt_step
        
    err_rho = np.max(np.abs(rho - exact_rho(x_rho, y_rho, T_end)))
    err_u = np.max(np.abs(u - exact_u(x_u, y_u, T_end)))
    err_v = np.max(np.abs(v - exact_v(x_v, y_v, T_end)))
    return max(err_rho, err_u, err_v)

if __name__ == "__main__":
    w_classic, w_opt = load_weights()
    resolutions = [20, 30, 40, 50, 60, 80, 100, 120, 160, 200]
    CFL = 0.4
    T_end = 0.2
    
    dt_vals = CFL / np.array(resolutions, dtype=float)

    results = []
    
    for w, name, color, fmt in [(w_classic, r"\textrm{Classic 4th-Order}", '#1a8a8a', 'o-'), 
                                (w_opt, r"\textrm{DE-Optimised Staggered}", '#e07020', 's-')]:
        print(f"\n=== {name} ===")
        errors = []
        for N in resolutions:
            err = run_simulation(N, CFL, T_end, w)
            errors.append(err)
            print(f"N={N:4d} | dt={CFL/N:.4f} | L2_err = {err:.3e}")
        
        # Proper log-log polyfit for convergence order
        log_dt = np.log(dt_vals)
        log_err = np.log(np.array(errors))
        slope, _ = np.polyfit(log_dt, log_err, 1)
        print(f"Order (polyfit): {slope:.2f}")
        results.append((name, dt_vals, errors, color, fmt, slope))
        
    plt.figure(figsize=(7, 5))
    for name, dts, errs, color, fmt, order in results:
        plt.loglog(dts, errs, fmt, color=color, label=rf'{name} ($\mathcal{{O}}(\Delta t^{{{order:.2f}}})$)', lw=2, ms=6)
        
    # Reference lines anchored at the FIRST (largest dt) data point of the classic curve
    anchor_dt = dt_vals[0]
    anchor_err = results[0][2][0]
    ref2 = anchor_err * (dt_vals / anchor_dt)**2
    ref4 = anchor_err * (dt_vals / anchor_dt)**4
    plt.loglog(dt_vals, ref2, 'k--', alpha=0.35, lw=1)
    plt.loglog(dt_vals, ref4, 'k:', alpha=0.35, lw=1)
    plt.text(dt_vals[-1]*1.05, ref2[-1], r'$\mathcal{O}(\Delta t^2)$', fontsize=10, color='k', alpha=0.7, ha='left', va='center')
    plt.text(dt_vals[-1]*1.05, ref4[-1], r'$\mathcal{O}(\Delta t^4)$', fontsize=10, color='k', alpha=0.7, ha='left', va='center')
    
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$L_\infty$ error')
    plt.title(r'2D Arakawa C-grid: SSP-RK3 convergence')
    plt.legend()
    os.makedirs(os.path.join(os.path.dirname(__file__), '../images'), exist_ok=True)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../images/fig_staggered_convergence.pdf'), bbox_inches='tight')
    print("\nSaved figure to images/fig_staggered_convergence.pdf")
