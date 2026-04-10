import sys, os
import numpy as np
from scipy.optimize import differential_evolution
import time

def parse_weights(x):
    w2, w3 = x
    w1 = -1.0 - 3.0*w2 - 5.0*w3
    w0 = 2.0 + 2.0*w2 + 4.0*w3
    return np.array([w0, w1, w2, w3])

def ssp_rk3_step_cc(u, t, dt, dx, g_func, w):
    N = len(u)
    
    def compute_rhs(u_val, t_val):
        du = np.zeros_like(u_val)
        
        # Ghost cell evaluated via parametrised extended stencil
        # u_{-1/2} = w0*g(t) + w1*u_{1/2} + w2*u_{3/2} + w3*u_{5/2}
        u_ghost_L = w[0] * g_func(t_val) + w[1] * u_val[0] + w[2] * u_val[1] + w[3] * u_val[2]
            
        # 2nd order central
        du[0] = -(u_val[1] - u_ghost_L) / (2 * dx)
        du[1:-1] = -(u_val[2:] - u_val[:-2]) / (2 * dx)
        
        # Exact right ghost cell
        x_ghost_R = 1.0 + 0.5 * dx
        u_ghost_R = np.sin(2 * np.pi * (x_ghost_R - t_val))
        du[-1] = -(u_ghost_R - u_val[-2]) / (2 * dx)
        
        return du
        
    du1 = compute_rhs(u, t)
    u1 = u + dt * du1
    
    du2 = compute_rhs(u1, t + dt)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * du2
    
    du3 = compute_rhs(u2, t + 0.5 * dt)
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * du3
    
    return u_new

def run_simulation(N, CFL, T_end, w):
    L = 1.0
    dx = L / N
    dt = CFL * dx
    x = np.linspace(0.5 * dx, L - 0.5 * dx, N)
    
    def exact_sol(x, t): return np.sin(2 * np.pi * (x - t))
    def g_func(t): return exact_sol(0, t)

    u = exact_sol(x, 0)
    t = 0.0
    while t < T_end - 1e-12:
        dt_step = min(dt, T_end - t)
        u = ssp_rk3_step_cc(u, t, dt_step, dx, g_func, w)
        t += dt_step
        
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e3:
            return 1e10
            
    err = np.max(np.abs(u - exact_sol(x, T_end)))
    return err

def evaluate_order(x):
    w = parse_weights(x)
    resolutions = [40, 80, 160]
    CFL = 0.4
    T_end = 0.5
    
    errors = []
    for N in resolutions:
        err = run_simulation(N, CFL, T_end, w)
        if err > 1e9:
            return -1.0
        errors.append(err)
        
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    return np.mean(orders)

def cost_function(x):
    order = evaluate_order(x)
    if order < 0:
        return 1e6  # heavily penalize unstable stencils
    
    target_order = 3.0
    diff = target_order - order
    # heavily penalise anything below target, light penalty for exceeding
    if diff > 0:
        return diff**2
    else:
        return 0.1 * diff**2

if __name__ == "__main__":
    print("=== Testing baseline staggered ghost cell ===")
    order_base = evaluate_order([0.0, 0.0])
    print(f"Classic 2-point ghost cell (w2=0, w3=0): Order = {order_base:.3f}")
    
    print("\\n=== Starting Differential Evolution for Staggered Boundary ===")
    bounds = [(-5, 5), (-5, 5)]
    
    start_time = time.time()
    result = differential_evolution(
        cost_function, bounds, 
        strategy='best1bin', maxiter=30, popsize=15, 
        tol=1e-4, mutation=(0.5, 1.0), recombination=0.7, disp=True
    )
    
    w_opt = parse_weights(result.x)
    opt_order = evaluate_order(result.x)
    print(f"\\nOptimization finished in {time.time() - start_time:.1f}s")
    print(f"Success: {result.success}")
    print(f"Found Order: {opt_order:.3f}")
    print(f"Optimal weights [w2, w3] = {result.x}")
    print(f"Full expansion weights [w0, w1, w2, w3] = {w_opt}")
    
    out_path = os.path.join(os.path.dirname(__file__), '../data/staggered_weights.npz')
    np.savez(out_path, w=w_opt)
    print(f"Saved optimal weights to {out_path}")
