"""
Differential Evolution optimiser for 4th-order staggered ghost-cell weights.
Trains DIRECTLY on the 2D Acoustic Wave Equation (Arakawa C-grid).
"""
import sys, os
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import null_space
import time

# --- Constraint matrix for O(dx^4) ghost cell ---
# x_pts = 0 (boundary), 1, 2, 3, 4, 5, 6 (interior u nodes, normalised)
# Extrapolate to x = -1
x_pts = np.array([0, 1, 2, 3, 4, 5, 6])
A = np.vstack([x_pts**0, x_pts**1, x_pts**2, x_pts**3])
b = np.array([1, -1, 1, -1])

w_p = np.linalg.pinv(A) @ b
Z = null_space(A)

def get_weights(c):
    return w_p + Z @ c

# --- 2D Arakawa C-grid SSP-RK3 solver ---
def ssp_rk3_step_2d(rho, u, v, t, dt, dx, dy, exact_rho, exact_u, exact_v, w_ghost):
    def compute_rhs(rho_val, u_val, v_val, t_val):
        drho = np.zeros_like(rho_val)
        du = np.zeros_like(u_val)
        dv = np.zeros_like(v_val)

        y_u = np.linspace(0.5*dy, 1.0 - 0.5*dy, rho_val.shape[1])
        u_bound_L = exact_u(0.0, y_u, t_val)
        u_bound_R = exact_u(1.0, y_u, t_val)

        u_ghost_L = (w_ghost[0] * u_bound_L + w_ghost[1] * u_val[1, :] +
                     w_ghost[2] * u_val[2, :] + w_ghost[3] * u_val[3, :] +
                     w_ghost[4] * u_val[4, :] + w_ghost[5] * u_val[5, :] + w_ghost[6] * u_val[6, :])
        u_ghost_R = (w_ghost[0] * u_bound_R + w_ghost[1] * u_val[-2, :] +
                     w_ghost[2] * u_val[-3, :] + w_ghost[3] * u_val[-4, :] +
                     w_ghost[4] * u_val[-5, :] + w_ghost[5] * u_val[-6, :] + w_ghost[6] * u_val[-7, :])

        u_pad = np.vstack([u_ghost_L, u_val, u_ghost_R])
        u_x = (-u_pad[3:] + 27.0*u_pad[2:-1] - 27.0*u_pad[1:-2] + u_pad[:-3]) / (24.0*dx)

        v_yp2 = np.roll(v_val, -2, axis=1)
        v_yp1 = np.roll(v_val, -1, axis=1)
        v_ym1 = np.roll(v_val, 1, axis=1)
        v_y = (-v_yp2 + 27.0*v_yp1 - 27.0*v_val + v_ym1) / (24.0*dy)

        drho = -(u_x + v_y)

        rho_ghost_L = 4.0*rho_val[0,:] - 6.0*rho_val[1,:] + 4.0*rho_val[2,:] - rho_val[3,:]
        rho_ghost_R = 4.0*rho_val[-1,:] - 6.0*rho_val[-2,:] + 4.0*rho_val[-3,:] - rho_val[-4,:]
        rho_pad = np.vstack([rho_ghost_L, rho_val, rho_ghost_R])
        rho_x = (-rho_pad[3:] + 27.0*rho_pad[2:-1] - 27.0*rho_pad[1:-2] + rho_pad[:-3]) / (24.0*dx)
        du[1:-1,:] = -rho_x

        rho_yp1 = np.roll(rho_val, -1, axis=1)
        rho_ym1 = np.roll(rho_val, 1, axis=1)
        rho_ym2 = np.roll(rho_val, 2, axis=1)
        rho_y = (-rho_yp1 + 27.0*rho_val - 27.0*rho_ym1 + rho_ym2) / (24.0*dy)
        dv = -rho_y

        return drho, du, dv

    dr1, du1, dv1 = compute_rhs(rho, u, v, t)
    rho1 = rho + dt*dr1; u1 = u + dt*du1; v1 = v + dt*dv1
    y_u = np.linspace(0.5*dy, 1.0-0.5*dy, rho.shape[1])
    u1[0,:] = exact_u(0.0, y_u, t+dt); u1[-1,:] = exact_u(1.0, y_u, t+dt)

    dr2, du2, dv2 = compute_rhs(rho1, u1, v1, t+dt)
    rho2 = 0.75*rho + 0.25*rho1 + 0.25*dt*dr2
    u2 = 0.75*u + 0.25*u1 + 0.25*dt*du2
    v2 = 0.75*v + 0.25*v1 + 0.25*dt*dv2
    u2[0,:] = exact_u(0.0, y_u, t+0.5*dt); u2[-1,:] = exact_u(1.0, y_u, t+0.5*dt)

    dr3, du3, dv3 = compute_rhs(rho2, u2, v2, t+0.5*dt)
    rho_new = (1./3.)*rho + (2./3.)*rho2 + (2./3.)*dt*dr3
    u_new = (1./3.)*u + (2./3.)*u2 + (2./3.)*dt*du3
    v_new = (1./3.)*v + (2./3.)*v2 + (2./3.)*dt*dv3
    u_new[0,:] = exact_u(0.0, y_u, t+dt); u_new[-1,:] = exact_u(1.0, y_u, t+dt)

    return rho_new, u_new, v_new

def run_sim_2d(N, CFL, T_end, w_ghost):
    L = 1.0; dx = L/N; dy = L/N; dt = CFL*dx
    x_1d = np.linspace(0.5*dx, L-0.5*dx, N)
    y_1d = np.linspace(0.5*dy, L-0.5*dy, N)
    x_rho, y_rho = np.meshgrid(x_1d, y_1d, indexing='ij')
    x_u_1d = np.linspace(0, L, N+1)
    x_u, y_u = np.meshgrid(x_u_1d, y_1d, indexing='ij')
    x_v_1d = np.linspace(0.5*dx, L-0.5*dx, N)
    y_v_1d = np.linspace(0, L, N, endpoint=False)
    x_v, y_v = np.meshgrid(x_v_1d, y_v_1d, indexing='ij')

    def exact_rho(x, y, t): return np.sin(2*np.pi*(x - t))
    def exact_u(x, y, t): return np.sin(2*np.pi*(x - t))
    def exact_v(x, y, t): return np.zeros_like(x)

    rho = exact_rho(x_rho, y_rho, 0)
    u = exact_u(x_u, y_u, 0)
    v = exact_v(x_v, y_v, 0)

    t = 0.0
    while t < T_end - 1e-12:
        dt_step = min(dt, T_end - t)
        rho, u, v = ssp_rk3_step_2d(rho, u, v, t, dt_step, dx, dy, exact_rho, exact_u, exact_v, w_ghost)
        t += dt_step
        if not np.all(np.isfinite(rho)) or np.max(np.abs(rho)) > 1e3:
            return 1e10

    err_rho = np.max(np.abs(rho - exact_rho(x_rho, y_rho, T_end)))
    err_u = np.max(np.abs(u - exact_u(x_u, y_u, T_end)))
    return max(err_rho, err_u)

# --- Cost function: directly uses the 2D solver ---
# Train on a wide geometric span so the optimizer must produce a
# consistent slope across the full evaluation range, not just one window.
TRAIN_RESOLUTIONS = [20, 40, 80, 160]
TRAIN_CFL = 0.4
TRAIN_T_END = 0.3

def evaluate_order_2d(c):
    w = get_weights(c)
    errors = []
    for N in TRAIN_RESOLUTIONS:
        err = run_sim_2d(N, TRAIN_CFL, TRAIN_T_END, w)
        if err > 1e9: return -1.0, [], 0.0
        errors.append(err)
    dt_vals = TRAIN_CFL / np.array(TRAIN_RESOLUTIONS, dtype=float)
    log_dt = np.log(dt_vals)
    log_err = np.log(np.array(errors))
    coeffs = np.polyfit(log_dt, log_err, 1)
    order = coeffs[0]
    fitted = np.polyval(coeffs, log_dt)
    ss_res = np.sum((log_err - fitted)**2)
    ss_tot = np.sum((log_err - np.mean(log_err))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return order, errors, r_squared

def cost_function(c):
    order, errors, r2 = evaluate_order_2d(c)
    if order < 0 or len(errors) == 0:
        return 1e6

    w = get_weights(c)

    # 1. PRIMARY: straight line on log-log plot (R² → 1)
    cost = 300.0 * (1.0 - r2)

    # 2. Soft lower bound on slope: SSP-RK3 with fixed CFL is at most 3rd order
    if order < 3.0:
        cost += 20.0 * (3.0 - order)**2

    # 3. Secondary: minimise error at the finest training grid
    cost += 1e5 * errors[-1]

    # 4. Light regularisation on weight magnitude
    cost += 0.005 * np.sum(w**2)

    return cost

if __name__ == "__main__":
    print(f"Null space DOF: {Z.shape[1]}")
    print(f"Training resolutions: {TRAIN_RESOLUTIONS}  CFL={TRAIN_CFL}  T_end={TRAIN_T_END}")

    w_classic = np.array([4.0, -6.0, 4.0, -1.0, 0.0, 0.0, 0.0])
    o_c, e_c, r2_c = evaluate_order_2d(np.zeros(Z.shape[1]))
    print(f"Particular solution:   order={o_c:.3f}, R²={r2_c:.6f}, errors={e_c}")

    c_classic = np.linalg.lstsq(Z, w_classic - w_p, rcond=None)[0]
    o_cl, e_cl, r2_cl = evaluate_order_2d(c_classic)
    print(f"Classic [4,-6,4,-1,0,0,0]: order={o_cl:.3f}, R²={r2_cl:.6f}, errors={e_cl}")

    bounds = [(-30, 30)] * Z.shape[1]

    print("\nStarting 2D Differential Evolution (wide-range, R²-primary) ...")
    start = time.time()
    res = differential_evolution(
        cost_function, bounds,
        strategy='best1bin', maxiter=150, popsize=30,
        tol=1e-7, mutation=(0.5, 1.5), recombination=0.9,
        workers=-1, disp=True, seed=42
    )

    w_opt = get_weights(res.x)
    o_opt, e_opt, r2_opt = evaluate_order_2d(res.x)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Success: {res.success}")
    print(f"2D Order: {o_opt:.3f}, R²: {r2_opt:.6f}")
    print(f"2D Errors: {e_opt}")
    print(f"Weights: {w_opt}")

    out_path = os.path.join(os.path.dirname(__file__), '../data/staggered_weights_2d.npz')
    np.savez(out_path, w=w_opt)
    print(f"Saved to {out_path}")
