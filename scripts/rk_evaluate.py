"""
Evaluate the convergence order of a given Butcher tableau on test PDEs
with static and time-dependent boundary conditions.

Supports: advection, diffusion, advection-diffusion.
"""

import numpy as np
import warnings


def upwind5_dudx(u, dx):
    """5th-order upwind du/dx (c > 0), vectorised."""
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
    du[-1] = (3*u[-1] - 4*u[-2] + u[-3]) / (2*dx)
    return du

def upwind5_dudx_2d(u, dx):
    """2D 5th order upwind derivatives using standard 1D stencils along axes."""
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    for j in range(u.shape[1]):
        du_dx[:, j] = upwind5_dudx(u[:, j], dx)
    for i in range(u.shape[0]):
        du_dy[i, :] = upwind5_dudx(u[i, :], dx)
    return du_dx, du_dy

def exact_u_2d(X, Y, t, cx, cy):
    return np.sin(np.pi * (X + Y - (cx + cy)*t))

def apply_bc_2d(u, X, Y, t, cx, cy):
    u[0, :] = exact_u_2d(0.0, Y[0,:], t, cx, cy)
    u[:, 0] = exact_u_2d(X[:,0], 0.0, t, cx, cy)
    return u


def central2_d2udx2(u, dx):
    """2nd-order central d²u/dx², interior only (boundary handled separately)."""
    n = len(u)
    d2u = np.zeros(n)
    d2u[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / dx**2
    return d2u


def step_generic_rk(u, t, dt, A_rk, b_rk, c_rk, rhs_func, g_func):
    """
    One step of a generic explicit RK method.

    At each stage, the boundary node u[0] is set to g(t + c_i * dt)
    before evaluating the RHS. This is the 'naive' boundary treatment
    that causes order reduction.
    """
    s = len(b_rk)
    n = len(u)
    K = np.zeros((s, n))

    for i in range(s):
        # Build intermediate state
        u_stage = u.copy()
        for j in range(i):
            u_stage += dt * A_rk[i, j] * K[j]

        # Set boundary at stage time
        t_stage = t + c_rk[i] * dt
        u_stage[0] = g_func(t_stage)

        # Evaluate RHS (time-dependent for MMS source terms)
        K[i] = rhs_func(u_stage, t_stage)

        # Early termination if unstable
        if not np.all(np.isfinite(K[i])):
            u_new = u.copy()
            u_new[:] = np.inf
            return u_new

    # Update
    u_new = u.copy()
    for i in range(s):
        u_new += dt * b_rk[i] * K[i]
    u_new[0] = g_func(t + dt)  # exact BC at full step

    return u_new

def step_generic_rk_2d(u, t, dt, dx, cx, cy, A_rk, b_rk, c_rk, X, Y):
    """Generic explicit RK step on 2D Advection."""
    s = len(b_rk)
    K = np.zeros((s, *u.shape))

    for i in range(s):
        u_stage = u.copy()
        for j in range(i):
            u_stage += dt * A_rk[i, j] * K[j]

        t_stage = t + c_rk[i] * dt
        u_stage = apply_bc_2d(u_stage, X, Y, t_stage, cx, cy)

        du_dx, du_dy = upwind5_dudx_2d(u_stage, dx)
        K[i] = -(cx * du_dx + cy * du_dy)

        if not np.all(np.isfinite(K[i])):
            u_new = u.copy()
            u_new[:] = np.inf
            return u_new

    u_new = u.copy()
    for i in range(s):
        u_new += dt * b_rk[i] * K[i]
    u_new = apply_bc_2d(u_new, X, Y, t + dt, cx, cy)

    return u_new


def measure_order(A_rk, b_rk, c_rk, pde='advection', bc_type='time_dependent',
                  CFL=0.5, T_end=0.5, dt_vals=None):
    """
    Run a convergence study and return measured order of convergence.
    """
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625])

    c_adv = 1.0
    nu = 0.01
    omega = 4 * np.pi
    L = 3.0

    errors = []
    for dt in dt_vals:
        dx = dt * c_adv / CFL
        nx = int(L / dx) + 1
        x = np.linspace(0, L, nx)
        dx = x[1] - x[0]

        # Define RHS and exact solution based on PDE type
        if pde == 'advection':
            rhs = lambda u, t_stage, _dx=dx: -c_adv * upwind5_dudx(u, _dx)
            if bc_type == 'static':
                def _pulse(y):
                    out = np.zeros_like(y)
                    m = (y > 0.3) & (y < 0.7)
                    out[m] = np.sin(np.pi * (y[m] - 0.3) / 0.4)**4
                    return out
                u = _pulse(x)
                g = lambda t: 0.0
                def _make_exact_static(x_arr, c_val):
                    def exact(t):
                        y = x_arr - c_val * t
                        out = np.zeros_like(y)
                        m = (y > 0.3) & (y < 0.7)
                        out[m] = np.sin(np.pi * (y[m] - 0.3) / 0.4)**4
                        return out
                    return exact
                exact = _make_exact_static(x, c_adv)
            else:
                u = np.zeros(nx)
                g = lambda t, _om=omega: np.sin(_om * t)
                def _make_exact_td(x_arr, c_val, om):
                    def exact(t):
                        out = np.zeros_like(x_arr)
                        mask = x_arr <= c_val * t
                        out[mask] = np.sin(om * (t - x_arr[mask] / c_val))
                        return out
                    return exact
                exact = _make_exact_td(x, c_adv, omega)

        elif pde == 'diffusion':
            rhs = lambda u, t_stage: nu * central2_d2udx2(u, dx)
            if bc_type == 'static':
                u = np.sin(np.pi * x / L)
                g = lambda t: 0.0
                exact = lambda t: np.sin(np.pi * x / L) * np.exp(-nu * (np.pi / L)**2 * t)
            else:
                u = np.sin(np.pi * x / L)
                g = lambda t, _om=omega: np.sin(_om * t)
                def exact(t, _om=omega, _nu=nu, _x=x, _L=L):
                    return np.sin(np.pi * _x / _L) * np.exp(-_nu * (np.pi/_L)**2 * t) + (1 - _x/_L) * np.sin(_om * t)

        elif pde == 'burgers_mms':
            # Local wave speed for burgers_mms maxes at ~3.0.
            dx = dt * 3.0 / CFL
            nx = int(L / dx) + 1
            x = np.linspace(0, L, nx)
            dx = x[1] - x[0]
            
            def exact(t, _x=x, _c=c_adv):
                return 2.0 + np.sin(np.pi * (_x - _c * t))
            g = lambda t, _c=c_adv: 2.0 + np.sin(-np.pi * _c * t)
            
            def rhs(u, t_stage, _dx=dx, _x=x, _c=c_adv):
                phase = np.pi * (_x - _c * t_stage)
                S = np.pi * np.cos(phase) * (2.0 - _c + np.sin(phase))
                return -u * upwind5_dudx(u, _dx) + S
                
            u = exact(0.0)

        t_curr = 0.0
        while t_curr < T_end - 1e-12:
            step_dt = min(dt, T_end - t_curr)
            u = step_generic_rk(u, t_curr, step_dt, A_rk, b_rk, c_rk, rhs, g)
            t_curr += step_dt

        err = np.sqrt(np.mean((u - exact(T_end))**2))
        errors.append(err)

    errors = np.array(errors)

    # Filter out any zero or negative errors (numerical noise)
    mask = errors > 1e-15
    if mask.sum() < 2:
        return 0.0, errors

    order = np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0]
    return order, errors

def measure_order_2d(A_rk, b_rk, c_rk, CFL=0.5, T_end=0.2, dt_vals=None):
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125])
        
    errors = []
    cx = 1.0
    cy = 1.0
    
    for dt in dt_vals:
        dx = dt * (cx + cy) / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        y = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        u = exact_u_2d(X, Y, 0.0, cx, cy)
        
        t_curr = 0.0
        while t_curr < T_end - 1e-12:
            step_dt = min(dt, T_end - t_curr)
            u = step_generic_rk_2d(u, t_curr, step_dt, dx_a, cx, cy, A_rk, b_rk, c_rk, X, Y)
            t_curr += step_dt
            
        u_exact = exact_u_2d(X, Y, T_end, cx, cy)
        err = np.sqrt(np.mean((u - u_exact)**2))
        errors.append(err)
        
    errors = np.array(errors)
    mask = errors > 1e-15
    if mask.sum() < 2:
        return 0.0, errors
    return np.polyfit(np.log(dt_vals[mask]), np.log(errors[mask]), 1)[0], errors


if __name__ == '__main__':
    from rk_parametrise import ssp_rk3, build_tableau_4s3p

    # Test SSP-RK3
    A, b, c = ssp_rk3()
    print("=== SSP-RK3 ===")
    o_s, _ = measure_order(A, b, c, pde='advection', bc_type='static')
    o_td, _ = measure_order(A, b, c, pde='advection', bc_type='time_dependent')
    print(f"  Static BC:    order {o_s:.2f}")
    print(f"  Time-dep BC:  order {o_td:.2f}")

    # Test a random 4-stage method
    print("\n=== Random 4-stage order-3 ===")
    free = [0.8, 0.4, 0.9, 0.2, 0.3, 0.35]
    A4, b4, c4, valid = build_tableau_4s3p(free)
    if valid:
        o_s4, _ = measure_order(A4, b4, c4, pde='advection', bc_type='static')
        o_td4, _ = measure_order(A4, b4, c4, pde='advection', bc_type='time_dependent')
        print(f"  Static BC:    order {o_s4:.2f}")
        print(f"  Time-dep BC:  order {o_td4:.2f}")
    else:
        print("  Invalid tableau")
