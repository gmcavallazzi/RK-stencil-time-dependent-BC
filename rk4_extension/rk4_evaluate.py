"""
Convergence validation for RK4 with standard, accuracy-only, and
stability-augmented near-boundary stencils.

Test problems:
  (a) 1D linear advection
  (b) 1D Burgers MMS
  (c) 2D linear advection
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import warnings

from stencil_optimise import (make_dudx_func, get_weights,
                              W1_STANDARD, W2_STANDARD)
from rk4_optimise import step_rk4, measure_order_rk4


# ==================== 1D Burgers MMS ====================

def upwind5_dudx(u, dx):
    """5th-order upwind du/dx."""
    n = len(u)
    du = np.zeros(n)
    du[1] = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)
    if n > 3:
        du[2] = (u[0] - 6*u[1] + 3*u[2] + 2*u[3]) / (6*dx)
    du[3:n-2] = (-2*u[0:n-5] + 15*u[1:n-4] - 60*u[2:n-3]
                 + 20*u[3:n-2] + 30*u[4:n-1] - 3*u[5:n]) / (60*dx)
    if n > 4:
        du[-2] = (u[-4] - 6*u[-3] + 3*u[-2] + 2*u[-1]) / (6*dx)
    du[-1] = (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx)
    return du


def measure_order_rk4_burgers(w1, w2, CFL=0.5, T_end=0.5, dt_vals=None):
    """Burgers MMS convergence with RK4 and parameterised boundary stencils."""
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

    c_adv = 1.0
    L = 3.0
    errors = []

    for dt in dt_vals:
        dx = dt * 3.0 / CFL
        nx = int(L / dx) + 1
        x = np.linspace(0, L, nx)
        dx_a = x[1] - x[0]

        def exact(t, _x=x, _c=c_adv):
            return 2.0 + np.sin(np.pi * (_x - _c * t))

        g = lambda t, _c=c_adv: 2.0 + np.sin(-np.pi * _c * t)

        dudx_func = make_dudx_func(w1, w2)

        def rhs(u, t_stage, _dx=dx_a, _x=x, _c=c_adv, _dudx=dudx_func):
            phase = np.pi * (_x - _c * t_stage)
            S = np.pi * np.cos(phase) * (2.0 - _c + np.sin(phase))
            return -u * _dudx(u, _dx) + S

        u = exact(0.0)

        # RK4 stages manually for Burgers (nonlinear RHS)
        t_curr = 0.0
        blown = False
        while t_curr < T_end - 1e-12:
            step_dt = min(dt, T_end - t_curr)

            u[0] = g(t_curr)
            k1 = rhs(u, t_curr)

            u2 = u + 0.5 * step_dt * k1
            u2[0] = g(t_curr + 0.5 * step_dt)
            k2 = rhs(u2, t_curr + 0.5 * step_dt)

            u3 = u + 0.5 * step_dt * k2
            u3[0] = g(t_curr + 0.5 * step_dt)
            k3 = rhs(u3, t_curr + 0.5 * step_dt)

            u4 = u + step_dt * k3
            u4[0] = g(t_curr + step_dt)
            k4 = rhs(u4, t_curr + step_dt)

            u = u + (step_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            u[0] = g(t_curr + step_dt)
            t_curr += step_dt

            if not np.all(np.isfinite(u)):
                blown = True
                break

        if blown:
            errors.append(1e10)
        else:
            err = np.sqrt(np.mean((u - exact(T_end))**2))
            errors.append(err)

    errors = np.array(errors)
    mask = (errors > 1e-15) & (errors < 1e9)
    if mask.sum() < 2:
        return 0.0, errors
    order = np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0]
    return order, errors


# ==================== 2D Linear Advection ====================

def make_dudx_2d_func(w1, w2):
    """2D spatial derivatives with parameterised boundary stencils."""
    def compute_derivatives(u, dx):
        nx, ny = u.shape
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)

        # X-derivative
        du_dx[1, :] = (w1[0]*u[0,:] + w1[1]*u[1,:] + w1[2]*u[2,:] +
                       w1[3]*u[3,:] + w1[4]*u[4,:]) / dx
        du_dx[2, :] = (w2[0]*u[0,:] + w2[1]*u[1,:] + w2[2]*u[2,:] +
                       w2[3]*u[3,:] + w2[4]*u[4,:]) / dx
        du_dx[3:nx-2, :] = (-2*u[0:nx-5,:] + 15*u[1:nx-4,:] - 60*u[2:nx-3,:] +
                             20*u[3:nx-2,:] + 30*u[4:nx-1,:] - 3*u[5:nx,:]) / (60*dx)
        du_dx[nx-2, :] = (u[nx-4,:] - 6*u[nx-3,:] + 3*u[nx-2,:] + 2*u[nx-1,:]) / (6*dx)
        du_dx[nx-1, :] = (3*u[nx-1,:] - 4*u[nx-2,:] + u[nx-3,:]) / (2*dx)

        # Y-derivative
        du_dy[:, 1] = (w1[0]*u[:,0] + w1[1]*u[:,1] + w1[2]*u[:,2] +
                       w1[3]*u[:,3] + w1[4]*u[:,4]) / dx
        du_dy[:, 2] = (w2[0]*u[:,0] + w2[1]*u[:,1] + w2[2]*u[:,2] +
                       w2[3]*u[:,3] + w2[4]*u[:,4]) / dx
        du_dy[:, 3:ny-2] = (-2*u[:,0:ny-5] + 15*u[:,1:ny-4] - 60*u[:,2:ny-3] +
                             20*u[:,3:ny-2] + 30*u[:,4:ny-1] - 3*u[:,5:ny]) / (60*dx)
        du_dy[:, ny-2] = (u[:,ny-4] - 6*u[:,ny-3] + 3*u[:,ny-2] + 2*u[:,ny-1]) / (6*dx)
        du_dy[:, ny-1] = (3*u[:,ny-1] - 4*u[:,ny-2] + u[:,ny-3]) / (2*dx)

        return du_dx, du_dy
    return compute_derivatives


def exact_u_2d(X, Y, t, cx, cy):
    return np.sin(np.pi * (X + Y - (cx + cy)*t))


def apply_bc_2d(u, X, Y, t, cx, cy):
    u[0, :] = exact_u_2d(0.0, Y[0, :], t, cx, cy)
    u[:, 0] = exact_u_2d(X[:, 0], 0.0, t, cx, cy)
    return u


def step_rk4_2d(u, t, dt, dx, cx, cy, deriv_func, X, Y):
    """RK4 step for 2D advection."""
    # Stage 1
    u = apply_bc_2d(u, X, Y, t, cx, cy)
    du_dx, du_dy = deriv_func(u, dx)
    k1 = -(cx * du_dx + cy * du_dy)

    # Stage 2
    u2 = u + 0.5 * dt * k1
    u2 = apply_bc_2d(u2, X, Y, t + 0.5*dt, cx, cy)
    du2_dx, du2_dy = deriv_func(u2, dx)
    k2 = -(cx * du2_dx + cy * du2_dy)

    # Stage 3
    u3 = u + 0.5 * dt * k2
    u3 = apply_bc_2d(u3, X, Y, t + 0.5*dt, cx, cy)
    du3_dx, du3_dy = deriv_func(u3, dx)
    k3 = -(cx * du3_dx + cy * du3_dy)

    # Stage 4
    u4 = u + dt * k3
    u4 = apply_bc_2d(u4, X, Y, t + dt, cx, cy)
    du4_dx, du4_dy = deriv_func(u4, dx)
    k4 = -(cx * du4_dx + cy * du4_dy)

    u_new = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    u_new = apply_bc_2d(u_new, X, Y, t + dt, cx, cy)
    return u_new


def measure_order_rk4_2d(w1, w2, CFL=0.5, T_end=0.2, dt_vals=None):
    """2D advection convergence with RK4."""
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125])

    cx, cy = 1.0, 1.0
    deriv_func = make_dudx_2d_func(w1, w2)
    errors = []

    for dt in dt_vals:
        dx = dt * (cx + cy) / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        X, Y = np.meshgrid(x, x, indexing='ij')
        u = exact_u_2d(X, Y, 0.0, cx, cy)

        t_curr = 0.0
        blown = False
        while t_curr < T_end - 1e-12:
            step_dt = min(dt, T_end - t_curr)
            u = step_rk4_2d(u, t_curr, step_dt, dx_a, cx, cy, deriv_func, X, Y)
            t_curr += step_dt
            if not np.all(np.isfinite(u)):
                blown = True
                break

        if blown:
            errors.append(1e10)
        else:
            u_exact = exact_u_2d(X, Y, T_end, cx, cy)
            err = np.sqrt(np.mean((u - u_exact)**2))
            errors.append(err)

    errors = np.array(errors)
    mask = (errors > 1e-15) & (errors < 1e9)
    if mask.sum() < 2:
        return 0.0, errors
    return np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0], errors


# ==================== Main ====================

def run():
    print("=== RK4 Convergence Validation ===\n")

    datadir = os.path.join(os.path.dirname(__file__), 'data')
    dt_full = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])
    dt_2d = np.array([0.01, 0.005, 0.0025, 0.00125])

    # Load stencils
    data_acc = np.load(os.path.join(datadir, 'rk4_stencils.npz'))
    w1_acc, w2_acc = data_acc['w1'], data_acc['w2']

    data_stab = np.load(os.path.join(datadir, 'rk4_stencils_stable.npz'))
    w1_stab, w2_stab = data_stab['w1'], data_stab['w2']

    stencils = [
        ('Standard',           W1_STANDARD, W2_STANDARD),
        ('Optimised (acc.)',   w1_acc,      w2_acc),
        ('Optimised (stab.)',  w1_stab,     w2_stab),
    ]

    # --- (a) 1D Linear Advection ---
    print("(a) 1D Linear Advection: u_t + u_x = 0")
    print(f"{'Stencil':<22s}", end="")
    for dt in dt_full:
        print(f"  dt={dt:.5f}", end="")
    print("  Order")
    print("-" * 100)

    for name, w1, w2 in stencils:
        o, errs = measure_order_rk4(w1, w2, dt_vals=dt_full)
        print(f"{name:<22s}", end="")
        for e in errs:
            print(f"  {e:11.3e}", end="")
        print(f"  {o:5.2f}")
    print()

    # --- (b) 1D Burgers MMS ---
    print("(b) 1D Burgers MMS: u_t + u*u_x = S")
    print(f"{'Stencil':<22s}", end="")
    for dt in dt_full:
        print(f"  dt={dt:.5f}", end="")
    print("  Order")
    print("-" * 100)

    for name, w1, w2 in stencils:
        o, errs = measure_order_rk4_burgers(w1, w2, dt_vals=dt_full)
        print(f"{name:<22s}", end="")
        for e in errs:
            print(f"  {e:11.3e}", end="")
        print(f"  {o:5.2f}")
    print()

    # --- (c) 2D Linear Advection ---
    print("(c) 2D Linear Advection: u_t + u_x + u_y = 0")
    print(f"{'Stencil':<22s}", end="")
    for dt in dt_2d:
        print(f"  dt={dt:.5f}", end="")
    print("  Order")
    print("-" * 80)

    for name, w1, w2 in stencils:
        o, errs = measure_order_rk4_2d(w1, w2, dt_vals=dt_2d)
        print(f"{name:<22s}", end="")
        for e in errs:
            print(f"  {e:11.3e}", end="")
        print(f"  {o:5.2f}")
    print()


if __name__ == '__main__':
    run()
