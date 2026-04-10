"""
Stability-augmented re-optimisation of boundary closure stencils.

Augments the accuracy-only cost function with a spectral stability penalty:
  J_aug = (3 - p_emp)^2 + lambda * max(0, rho(dt*D) - 1)^2

Uses warm-start from the existing optimised stencils.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
from scipy.optimize import differential_evolution
import time

from stencil_optimise import (step_rk3, make_dudx_func, measure_order,
                              get_weights, W1_STANDARD, W2_STANDARD)


# ========== Stability Computation ==========

def build_spatial_operator(N, w1, w2):
    """Build N×N spatial derivative operator matrix D."""
    D = np.zeros((N, N))
    if N > 4:
        D[1, 0:5] = w1
        D[2, 0:5] = w2

    coeff = np.array([-2, 15, -60, 20, 30, -3]) / 60.0
    for i in range(3, N-2):
        if i-3 >= 0 and i+2 < N:
            D[i, i-3:i+3] = coeff

    if N > 4:
        D[-2, -4] = 1/6; D[-2, -3] = -1; D[-2, -2] = 1/2; D[-2, -1] = 1/3
    D[-1, -3] = 1/2; D[-1, -2] = -2; D[-1, -1] = 3/2
    return D


def max_amplification(w1, w2, cfl_target=1.0, N=50):
    """
    Compute max |R(dt * lambda_i)| for the spatial operator with given stencils
    at the target CFL.
    """
    dx = 3.0 / (N - 1)
    dt = cfl_target * dx
    D = build_spatial_operator(N, w1, w2)
    eigs = np.linalg.eigvals(-D / dx)
    z = dt * eigs
    # SSP-RK3: R(z) = 1 + z + z^2/2 + z^3/6
    R = 1 + z + z**2/2 + z**3/6
    return np.max(np.abs(R))


# ========== Augmented Objective ==========

def objective_stable(params, lam=5.0, cfl_target=0.95):
    """Augmented cost: accuracy + stability penalty."""
    w1, w2 = get_weights(params)

    # Accuracy term
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

    loss_accuracy = (3.0 - order)**2

    # Stability penalty
    try:
        amp = max_amplification(w1, w2, cfl_target=cfl_target, N=50)
        if not np.isfinite(amp):
            return 80.0
        stab_penalty = max(0, amp - 1.0)**2
    except Exception:
        return 80.0

    return loss_accuracy + lam * stab_penalty


# ========== Main ==========

def run():
    # Load existing optimised stencils for warm-start
    data = np.load('optimised_stencils.npz')
    w1_opt = data['w1']
    w2_opt = data['w2']

    # Convert to free params
    x_warm = np.array([w1_opt[2], w1_opt[3], w1_opt[4],
                       w2_opt[2], w2_opt[3], w2_opt[4]])

    print("=== Stability-Augmented Stencil Re-Optimisation ===\n")

    # Evaluate baseline
    print("Baseline (existing optimised stencils):")
    o_opt, _ = measure_order(w1_opt, w2_opt)
    amp_07 = max_amplification(w1_opt, w2_opt, cfl_target=0.7)
    amp_10 = max_amplification(w1_opt, w2_opt, cfl_target=1.0)
    print(f"  Order: {o_opt:.2f}")
    print(f"  |R| at CFL=0.7: {amp_07:.6f}")
    print(f"  |R| at CFL=1.0: {amp_10:.6f}\n")

    # Search parameters
    LAMBDA = 5.0
    CFL_TARGET = 0.95

    print(f"Optimising with lambda={LAMBDA}, CFL_target={CFL_TARGET}...")
    bounds = [(-3, 3)] * 6

    t0 = time.time()
    best_val = [100.0]
    best_x = [None]

    def callback(xk, convergence):
        val = objective_stable(xk, lam=LAMBDA, cfl_target=CFL_TARGET)
        if val < best_val[0]:
            best_val[0] = val
            best_x[0] = xk.copy()
            w1, w2 = get_weights(xk)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all='ignore'):
                    o, _ = measure_order(w1, w2,
                                         dt_vals=np.array([0.01, 0.004, 0.0016]),
                                         T_end=0.3)
            amp = max_amplification(w1, w2, cfl_target=CFL_TARGET)
            print(f"  BEST  loss={val:.4f}  order={o:.2f}  "
                  f"|R|(CFL={CFL_TARGET})={amp:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Use warm-start: seed the initial population with the known solution
    result = differential_evolution(
        lambda p: objective_stable(p, lam=LAMBDA, cfl_target=CFL_TARGET),
        bounds=bounds,
        x0=x_warm,           # warm-start from existing solution
        maxiter=80,
        popsize=20,
        tol=1e-4,
        seed=42,
        callback=callback,
        workers=1,
        polish=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    x_best = best_x[0] if best_x[0] is not None else result.x
    w1_new, w2_new = get_weights(x_best)

    print("--- Stability-Aware Stencils ---")
    print(f"Node 1: {w1_new.round(6)}")
    print(f"Node 2: {w2_new.round(6)}")

    # Full evaluation
    dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    o_new, e_new = measure_order(w1_new, w2_new, dt_vals=dt_full)
    o_std, e_std = measure_order(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
    o_opt_full, e_opt = measure_order(w1_opt, w2_opt, dt_vals=dt_full)

    print(f"\n{'Stencil':<25s} {'Order':<8s} {'|R| CFL=0.7':<14s} {'|R| CFL=1.0':<14s}")
    print("-" * 65)
    for name, w1, w2, o in [('Standard', W1_STANDARD, W2_STANDARD, o_std),
                             ('Optimised (acc-only)', w1_opt, w2_opt, o_opt_full),
                             ('Optimised (acc+stab)', w1_new, w2_new, o_new)]:
        a07 = max_amplification(w1, w2, cfl_target=0.7)
        a10 = max_amplification(w1, w2, cfl_target=1.0)
        print(f"{name:<25s} {o:<8.2f} {a07:<14.6f} {a10:<14.6f}")

    # Find critical CFL for new stencils
    crit = 0.0
    for cfl_test in np.arange(0.05, 2.0, 0.01):
        amp = max_amplification(w1_new, w2_new, cfl_target=cfl_test)
        if amp > 1.0 + 1e-8:
            crit = cfl_test - 0.01
            break
        crit = cfl_test
    print(f"\nCritical CFL (stability-aware): {crit:.3f}")

    np.savez('optimised_stencils_stable.npz', w1=w1_new, w2=w2_new,
             order=o_new, critical_cfl=crit)
    print(f"Saved to optimised_stencils_stable.npz")


if __name__ == '__main__':
    run()
