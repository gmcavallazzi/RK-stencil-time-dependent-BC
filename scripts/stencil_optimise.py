"""
Optimise near-boundary FD closure stencils for 1D advection with SSP-RK3.

Interior: 5th-order upwind stencil.
Boundary nodes 1, 2: parameterised stencils (5 weights each, 10 total).
Outflow nodes n-2, n-1: fixed (not the focus).

Consistency constraint: stencil weights w must satisfy
  sum(w_j * j^k) = k! * delta_{k,1}   for k = 0, 1
This ensures the stencil approximates du/dx (not a constant or higher derivative).
"""

import numpy as np
import warnings
from scipy.optimize import differential_evolution
import time


# ==================== SSP-RK3 stepper ====================

def step_rk3(u, t, dt, dx, c, dudx_func, g_func):
    u[0] = g_func(t)
    k1 = -c * dudx_func(u, dx)
    u1 = u + dt * k1
    u1[0] = g_func(t + dt)

    k2 = -c * dudx_func(u1, dx)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * k2
    u2[0] = g_func(t + 0.5 * dt)

    k3 = -c * dudx_func(u2, dx)
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * k3
    u_new[0] = g_func(t + dt)
    return u_new


# ==================== Spatial derivative ====================

def make_dudx_func(w1, w2):
    """
    Build a du/dx function with parameterised boundary stencils.

    w1: array of 5 weights for node 1 (uses nodes 0..4)
    w2: array of 5 weights for node 2 (uses nodes 0..4)
    Interior (nodes 3..n-3): 5th-order upwind.
    """
    def dudx(u, dx):
        n = len(u)
        du = np.zeros(n)

        # Node 1: parameterised
        du[1] = (w1[0]*u[0] + w1[1]*u[1] + w1[2]*u[2]
                 + w1[3]*u[3] + w1[4]*u[4]) / dx

        # Node 2: parameterised
        du[2] = (w2[0]*u[0] + w2[1]*u[1] + w2[2]*u[2]
                 + w2[3]*u[3] + w2[4]*u[4]) / dx

        # Interior: 5th-order upwind
        du[3:n-2] = (-2*u[0:n-5] + 15*u[1:n-4] - 60*u[2:n-3]
                     + 20*u[3:n-2] + 30*u[4:n-1] - 3*u[5:n]) / (60*dx)

        # Outflow (fixed, not focus of optimisation)
        if n > 4:
            du[-2] = (u[-4] - 6*u[-3] + 3*u[-2] + 2*u[-1]) / (6*dx)
        du[-1] = (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx)

        return du
    return dudx


# Standard stencils (for comparison)
W1_STANDARD = np.array([-3.0, 4.0, -1.0, 0.0, 0.0]) / 2.0  # 2nd-order forward
W2_STANDARD = np.array([1.0, -6.0, 3.0, 2.0, 0.0]) / 6.0   # 3rd-order upwind


# ==================== Convergence evaluator ====================

def measure_order(w1, w2, CFL=0.5, T_end=0.5, c=1.0,
                  dt_vals=None):
    """Run convergence study on smooth advection with time-dependent BC."""
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

    dudx = make_dudx_func(w1, w2)

    def exact_u(x, t):
        return np.sin(np.pi * (x - c * t))

    errors = []
    for dt in dt_vals:
        dx = dt * c / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]

        g = lambda t, _c=c: np.sin(np.pi * (0 - _c * t))
        u = exact_u(x, 0.0)

        t = 0.0
        blown = False
        while t < T_end - 1e-12:
            u = step_rk3(u, t, min(dt, T_end - t), dx_a, c, dudx, g)
            t += min(dt, T_end - t)
            if not np.all(np.isfinite(u)):
                blown = True
                break

        if blown:
            errors.append(1e10)
        else:
            err = np.sqrt(np.mean((u - exact_u(x, T_end))**2))
            errors.append(err)

    errors = np.array(errors)
    mask = (errors > 1e-15) & (errors < 1e9)
    if mask.sum() < 2:
        return 0.0, errors
    order = np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0]
    return order, errors


# ==================== Objective function ====================

def get_weights(params):
    """
    Given 6 free parameters, reconstruct w1 and w2 such that
    zeroth moment = 0 and first moment = 1 are strictly enforced.

    params[0..2] -> w1[2], w1[3], w1[4]
    params[3..5] -> w2[2], w2[3], w2[4]
    """
    w1 = np.zeros(5)
    w2 = np.zeros(5)

    w1[2], w1[3], w1[4] = params[0], params[1], params[2]
    w1[1] = 1.0 - 2*w1[2] - 3*w1[3] - 4*w1[4]
    w1[0] = -w1[1] - w1[2] - w1[3] - w1[4]

    w2[2], w2[3], w2[4] = params[3], params[4], params[5]
    w2[1] = 1.0 - 2*w2[2] - 3*w2[3] - 4*w2[4]
    w2[0] = -w2[1] - w2[2] - w2[3] - w2[4]

    return w1, w2


def objective(params):
    """6 free weights -> convergence order."""
    w1, w2 = get_weights(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all='ignore'):
            try:
                dt_fast = np.array([0.01, 0.004, 0.0016])
                order, errs = measure_order(w1, w2, dt_vals=dt_fast, T_end=0.3)
            except Exception:
                return 80.0

    if not np.isfinite(order):
        return 80.0

    # Want order as close to 3 as possible
    loss = (3.0 - order)**2
    return loss


# ==================== Main ====================

def run():
    # Initial guess: standard stencils
    # W1: [-0.5, 0.0, 0.0], W2: [0.5, 1/3, 0.0]
    x0 = np.array([-0.5, 0.0, 0.0, 0.5, 1.0/3.0, 0.0])

    # Evaluate baseline
    print("=== Baseline (standard stencils) ===", flush=True)
    o_std, _ = measure_order(W1_STANDARD, W2_STANDARD)
    print(f"  Order: {o_std:.2f}\n", flush=True)

    # Bounds for the free parameters
    bounds = [(-3, 3)] * 6

    print("Running optimisation...", flush=True)
    t0 = time.time()

    best_val = [100.0]
    best_x = [None]

    def callback(xk, convergence):
        val = objective(xk)
        if val < best_val[0]:
            best_val[0] = val
            best_x[0] = xk.copy()
            w1, w2 = get_weights(xk)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all='ignore'):
                    o, _ = measure_order(w1, w2, dt_vals=np.array([0.01, 0.004, 0.0016]), T_end=0.3)
            print(f"  BEST  loss={val:.4f}  order={o:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    result = differential_evolution(
        objective, bounds=bounds,
        maxiter=100, popsize=20, tol=1e-4, seed=42,
        callback=callback, workers=1, polish=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    x_best = best_x[0] if best_x[0] is not None else result.x
    w1_opt, w2_opt = get_weights(x_best)

    print("--- Discovered Stencils ---")
    print(f"Node 1: {w1_opt.round(6)}")
    print(f"Node 2: {w2_opt.round(6)}")

    # Full evaluation
    dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    o_opt, e_opt = measure_order(w1_opt, w2_opt, dt_vals=dt_full)
    o_std, e_std = measure_order(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)

    print(f"\n{'Stencil':<20s} {'Order':>6s}")
    print("-" * 28)
    print(f"{'Standard':<20s} {o_std:>6.2f}")
    print(f"{'Optimised':<20s} {o_opt:>6.2f}")

    np.savez('optimised_stencils.npz', w1=w1_opt, w2=w2_opt,
             order_std=o_std, order_opt=o_opt)
    print(f"\nSaved to optimised_stencils.npz")


if __name__ == '__main__':
    run()
