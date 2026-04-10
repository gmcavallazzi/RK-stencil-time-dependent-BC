"""
Stability-augmented DE optimisation of near-boundary FD closure stencils
for classical RK4.

Uses a simulation-based stability penalty: runs the actual RK4 simulation at
an elevated CFL and penalizes blow-up or error growth.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import warnings
from scipy.optimize import differential_evolution
import time

from stencil_optimise import get_weights, make_dudx_func, W1_STANDARD, W2_STANDARD
from stencil_optimise_stable import build_spatial_operator
from rk4_optimise import step_rk4, measure_order_rk4


# ========== RK4 Stability Computation ==========

def max_amplification_rk4(w1, w2, cfl_target=1.0, N=50):
    """
    Compute max |R4(dt * lambda_i)| for the spatial operator at target CFL.
    R4(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    """
    dx = 3.0 / (N - 1)
    dt = cfl_target * dx
    D = build_spatial_operator(N, w1, w2)
    eigs = np.linalg.eigvals(-D / dx)
    z = dt * eigs
    R = 1 + z + z**2/2 + z**3/6 + z**4/24
    return np.max(np.abs(R))


def check_simulation_stability(w1, w2, CFL=0.8, T_end=0.3):
    """
    Run a short simulation at the given CFL and return whether it's stable
    and the error magnitude. This directly tests practical stability.
    """
    dudx = make_dudx_func(w1, w2)
    c = 1.0
    dt = 0.005
    dx = dt * c / CFL
    nx = int(3.0 / dx) + 1
    x = np.linspace(0, 3.0, nx)
    dx_a = x[1] - x[0]

    def exact_u(x, t):
        return np.sin(np.pi * (x - c * t))

    g = lambda t, _c=c: np.sin(np.pi * (0 - _c * t))
    u = exact_u(x, 0.0)

    t_curr = 0.0
    while t_curr < T_end - 1e-12:
        u = step_rk4(u, t_curr, min(dt, T_end - t_curr), dx_a, c, dudx, g)
        t_curr += min(dt, T_end - t_curr)
        if not np.all(np.isfinite(u)):
            return False, 1e10

    err = np.sqrt(np.mean((u - exact_u(x, T_end))**2))
    return True, err


# ========== Augmented Objective ==========

def objective_stable(params, lam=3.0, cfl_target=0.8):
    """Augmented cost: accuracy + simulation-based stability penalty."""
    w1, w2 = get_weights(params)

    # Accuracy term
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

    # Reject non-monotonic error decay
    good = (errs > 1e-15) & (errs < 1e9)
    if good.sum() < 4:
        return 80.0
    for i in range(1, len(errs)):
        if good[i] and good[i-1] and errs[i] > errs[i-1] * 1.1:
            return 80.0

    loss_accuracy = (4.0 - order)**2

    # Stability penalty: run simulation at elevated CFL
    try:
        stable, err_high = check_simulation_stability(w1, w2, CFL=cfl_target, T_end=0.2)
        if not stable:
            stab_penalty = 10.0
        else:
            # Also check that the convergence holds at this CFL
            o_high, e_high = measure_order_rk4(w1, w2,
                                                dt_vals=np.array([0.008, 0.004, 0.002]),
                                                T_end=0.2, CFL=cfl_target)
            if not np.isfinite(o_high) or np.any(~np.isfinite(e_high)):
                stab_penalty = 10.0
            else:
                # Penalize if order drops significantly at higher CFL
                stab_penalty = max(0, order - o_high - 0.5)**2
    except Exception:
        stab_penalty = 10.0

    return loss_accuracy + lam * stab_penalty


# ========== Main ==========

def run():
    print("=== RK4 Stability-Augmented Stencil Optimisation ===\n")

    # Load accuracy-only stencils for warm-start
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    data = np.load(os.path.join(datadir, 'rk4_stencils.npz'))
    w1_acc = data['w1']
    w2_acc = data['w2']

    # Convert to free params for warm-start
    x_warm = np.array([w1_acc[2], w1_acc[3], w1_acc[4],
                       w2_acc[2], w2_acc[3], w2_acc[4]])

    print("Baseline (accuracy-only RK4 stencils):")
    o_acc, _ = measure_order_rk4(w1_acc, w2_acc)
    stable_07, _ = check_simulation_stability(w1_acc, w2_acc, CFL=0.7)
    stable_10, _ = check_simulation_stability(w1_acc, w2_acc, CFL=1.0)
    print(f"  Order: {o_acc:.2f}")
    print(f"  Stable at CFL=0.7: {stable_07}")
    print(f"  Stable at CFL=1.0: {stable_10}\n")

    LAMBDA = 3.0
    CFL_TARGET = 0.8

    print(f"Optimising with lambda={LAMBDA}, CFL_target={CFL_TARGET}...")
    bounds = [(-5, 5)] * 6

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
                    o, _ = measure_order_rk4(w1, w2,
                                             dt_vals=np.array([0.008, 0.004, 0.002, 0.001]),
                                             T_end=0.2)
            s, _ = check_simulation_stability(w1, w2, CFL=CFL_TARGET)
            print(f"  BEST  loss={val:.4f}  order={o:.2f}  "
                  f"stable(CFL={CFL_TARGET})={s}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    result = differential_evolution(
        lambda p: objective_stable(p, lam=LAMBDA, cfl_target=CFL_TARGET),
        bounds=bounds,
        x0=x_warm,
        maxiter=100, popsize=30, tol=1e-6, seed=42,
        callback=callback, workers=1, polish=False,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    x_best = best_x[0] if best_val[0] < result.fun else result.x
    w1_new, w2_new = get_weights(x_best)

    print("--- RK4 Stability-Aware Stencils ---")
    print(f"Node 1: {w1_new.round(6)}")
    print(f"Node 2: {w2_new.round(6)}")

    # Full evaluation
    dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    o_new, e_new = measure_order_rk4(w1_new, w2_new, dt_vals=dt_full)
    o_std, e_std = measure_order_rk4(W1_STANDARD, W2_STANDARD, dt_vals=dt_full)
    o_acc_full, e_acc = measure_order_rk4(w1_acc, w2_acc, dt_vals=dt_full)

    print(f"\n{'Stencil':<25s} {'Order':<8s} {'Stable CFL=0.7':<16s} {'Stable CFL=1.0':<16s}")
    print("-" * 70)
    for name, w1, w2, o in [('Standard', W1_STANDARD, W2_STANDARD, o_std),
                             ('Optimised (acc-only)', w1_acc, w2_acc, o_acc_full),
                             ('Optimised (acc+stab)', w1_new, w2_new, o_new)]:
        s07, _ = check_simulation_stability(w1, w2, CFL=0.7)
        s10, _ = check_simulation_stability(w1, w2, CFL=1.0)
        print(f"{name:<25s} {o:<8.2f} {str(s07):<16s} {str(s10):<16s}")

    # Find practical critical CFL via simulation
    crit = 0.0
    for cfl_test in np.arange(0.1, 2.0, 0.05):
        stable, _ = check_simulation_stability(w1_new, w2_new, CFL=cfl_test, T_end=0.5)
        if not stable:
            crit = cfl_test - 0.05
            break
        crit = cfl_test
    print(f"\nPractical critical CFL (stability-aware): {crit:.2f}")

    # Also find for acc-only
    crit_acc = 0.0
    for cfl_test in np.arange(0.1, 2.0, 0.05):
        stable, _ = check_simulation_stability(w1_acc, w2_acc, CFL=cfl_test, T_end=0.5)
        if not stable:
            crit_acc = cfl_test - 0.05
            break
        crit_acc = cfl_test
    print(f"Practical critical CFL (acc-only): {crit_acc:.2f}")

    outpath = os.path.join(datadir, 'rk4_stencils_stable.npz')
    np.savez(outpath, w1=w1_new, w2=w2_new,
             order=o_new, critical_cfl=crit,
             errors_std=e_std, errors_acc=e_acc, errors_stable=e_new,
             dt_vals=dt_full)
    print(f"Saved to {outpath}")


if __name__ == '__main__':
    run()
