"""
Accuracy-only DE optimisation of near-boundary FD closure stencils for
classical RK4 on 1D linear advection.

Reuses the stencil parametrisation from scripts/stencil_optimise.py.
Target: restore O(dt^4) convergence from the O(dt^2) order-reduced baseline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import warnings
from scipy.optimize import differential_evolution
import time

from stencil_optimise import (make_dudx_func, get_weights,
                              W1_STANDARD, W2_STANDARD)


# ==================== Classical RK4 stepper ====================

def step_rk4(u, t, dt, dx, c, dudx_func, g_func):
    """Classical 4-stage RK4 with boundary injection at each stage."""
    # Stage 1
    u[0] = g_func(t)
    k1 = -c * dudx_func(u, dx)

    # Stage 2
    u2 = u + 0.5 * dt * k1
    u2[0] = g_func(t + 0.5 * dt)
    k2 = -c * dudx_func(u2, dx)

    # Stage 3
    u3 = u + 0.5 * dt * k2
    u3[0] = g_func(t + 0.5 * dt)
    k3 = -c * dudx_func(u3, dx)

    # Stage 4
    u4 = u + dt * k3
    u4[0] = g_func(t + dt)
    k4 = -c * dudx_func(u4, dx)

    # Update
    u_new = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    u_new[0] = g_func(t + dt)
    return u_new


# ==================== Convergence evaluator ====================

def measure_order_rk4(w1, w2, CFL=0.5, T_end=0.5, c=1.0, dt_vals=None):
    """Run convergence study on smooth advection with RK4."""
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

        t_curr = 0.0
        blown = False
        while t_curr < T_end - 1e-12:
            u = step_rk4(u, t_curr, min(dt, T_end - t_curr), dx_a, c, dudx, g)
            t_curr += min(dt, T_end - t_curr)
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

def objective(params):
    """
    6 free weights -> convergence order targeting 4.
    Uses 5 dt points including fine grids, T_end=0.5, and a monotonicity guard.
    The order is measured on the finest 4 points to capture asymptotic behavior.
    """
    w1, w2 = get_weights(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all='ignore'):
            try:
                dt_fast = np.array([0.008, 0.004, 0.002, 0.001, 0.0005])
                order, errs = measure_order_rk4(w1, w2, dt_vals=dt_fast, T_end=0.2)
            except Exception:
                return 80.0

    if not np.isfinite(order):
        return 80.0

    # Reject if errors are not monotonically decreasing (sign of instability)
    good = (errs > 1e-15) & (errs < 1e9)
    if good.sum() < 4:
        return 80.0
    for i in range(1, len(errs)):
        if good[i] and good[i-1] and errs[i] > errs[i-1] * 1.1:
            return 80.0

    # Also measure order on the finest 3 points for asymptotic behavior
    fine_dt = dt_fast[-3:]
    fine_err = errs[-3:]
    fine_mask = (fine_err > 1e-15) & (fine_err < 1e9)
    if fine_mask.sum() >= 2:
        fine_order = np.polyfit(np.log(fine_dt[fine_mask]),
                                np.log(fine_err[fine_mask]), 1)[0]
    else:
        fine_order = order

    # Blend: emphasize asymptotic order
    loss = 0.3 * (4.0 - order)**2 + 0.7 * (4.0 - fine_order)**2
    return loss


# ==================== Main ====================

def run():
    print("=== RK4 Accuracy-Only Stencil Optimisation ===\n")

    # Evaluate baseline
    print("Baseline (standard stencils with RK4):")
    o_std, e_std = measure_order_rk4(W1_STANDARD, W2_STANDARD)
    print(f"  Order: {o_std:.2f}")
    for dt, err in zip([0.01, 0.005, 0.0025, 0.00125, 0.000625], e_std):
        print(f"    dt={dt:.5f}  err={err:.3e}")
    print()

    bounds = [(-5, 5)] * 6

    print("Running DE optimisation (popsize=50, maxiter=200)...", flush=True)
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
                    o, _ = measure_order_rk4(w1, w2,
                                             dt_vals=np.array([0.008, 0.004, 0.002, 0.001, 0.0005]),
                                             T_end=0.2)
            print(f"  BEST  loss={val:.4f}  order={o:.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    result = differential_evolution(
        objective, bounds=bounds,
        maxiter=200, popsize=50, tol=1e-8, seed=42,
        strategy='best1bin',
        callback=callback, workers=1, polish=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    x_best = best_x[0] if best_val[0] < result.fun else result.x
    w1_opt, w2_opt = get_weights(x_best)

    print("--- Discovered RK4-Optimised Stencils ---")
    print(f"Node 1: {w1_opt.round(6)}")
    print(f"Node 2: {w2_opt.round(6)}")

    # Full evaluation
    dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    o_opt, e_opt = measure_order_rk4(w1_opt, w2_opt, dt_vals=dt_full)
    o_std, e_std = measure_order_rk4(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)

    print(f"\n{'Stencil':<20s} {'Order':>6s}")
    print("-" * 28)
    print(f"{'Standard':<20s} {o_std:>6.2f}")
    print(f"{'Optimised (acc.)':<20s} {o_opt:>6.2f}")

    print("\nDetailed errors (optimised):")
    for dt, err in zip(dt_full, e_opt):
        print(f"    dt={dt:.5f}  err={err:.3e}")

    outdir = os.path.join(os.path.dirname(__file__), 'data')
    outpath = os.path.join(outdir, 'rk4_stencils.npz')
    np.savez(outpath, w1=w1_opt, w2=w2_opt,
             order_std=o_std, order_opt=o_opt,
             errors_std=e_std, errors_opt=e_opt, dt_vals=dt_full)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    run()
