"""
Validate the explicitly optimised boundary stencils on the 2D Linear Advection equation.
u_t + c_x u_x + c_y u_y = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from stencil_optimise import W1_STANDARD, W2_STANDARD

# Load optimised stencils
data = np.load('optimised_stencils.npz')
W1_OPT = data['w1']
W2_OPT = data['w2']

def make_dudx_2d_func(w1, w2):
    """
    Returns a function that computes the 2D spatial derivatives (u_x and u_y)
    assuming a uniform grid dx = dy and identical boundary stencils on x=0 and y=0.
    """
    def compute_derivatives(u, dx):
        # u is shape (nx, ny)
        nx, ny = u.shape
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        
        # --- Derivative in X ---
        # For each y-slice: u[:, j]
        # x=0 (i=0): boundary condition, derivative not strictly needed if we don't update i=0.
        
        # Node 1 (i=1)
        # u[1, :] = (w1[0]*u[0,:] + w1[1]*u[1,:] + ...)/dx
        du_dx[1, :] = (w1[0]*u[0,:] + w1[1]*u[1,:] + w1[2]*u[2,:] + 
                       w1[3]*u[3,:] + w1[4]*u[4,:]) / dx
        
        # Node 2 (i=2)
        du_dx[2, :] = (w2[0]*u[0,:] + w2[1]*u[1,:] + w2[2]*u[2,:] + 
                       w2[3]*u[3,:] + w2[4]*u[4,:]) / dx
                       
        # Interior (i=3 to nx-3) - 5th order upwind
        # -2 u_{i-3} + 15 u_{i-2} - 60 u_{i-1} + 20 u_i + 30 u_{i+1} - 3 u_{i+2}
        du_dx[3:nx-2, :] = (-2*u[0:nx-5,:] + 15*u[1:nx-4,:] - 60*u[2:nx-3,:] + 
                             20*u[3:nx-2,:] + 30*u[4:nx-1,:] - 3*u[5:nx,:]) / (60 * dx)
                             
        # Outflow boundaries (standard biased 3rd/2nd)
        du_dx[nx-2, :] = (u[nx-4,:] - 6*u[nx-3,:] + 3*u[nx-2,:] + 2*u[nx-1,:]) / (6 * dx)
        du_dx[nx-1, :] = (3*u[nx-1,:] - 4*u[nx-2,:] + u[nx-3,:]) / (2 * dx)
        
        # --- Derivative in Y ---
        # Node 1 (j=1)
        du_dy[:, 1] = (w1[0]*u[:,0] + w1[1]*u[:,1] + w1[2]*u[:,2] + 
                       w1[3]*u[:,3] + w1[4]*u[:,4]) / dx
        
        # Node 2 (j=2)
        du_dy[:, 2] = (w2[0]*u[:,0] + w2[1]*u[:,1] + w2[2]*u[:,2] + 
                       w2[3]*u[:,3] + w2[4]*u[:,4]) / dx
                       
        # Interior
        du_dy[:, 3:ny-2] = (-2*u[:,0:ny-5] + 15*u[:,1:ny-4] - 60*u[:,2:ny-3] + 
                             20*u[:,3:ny-2] + 30*u[:,4:ny-1] - 3*u[:,5:ny]) / (60 * dx)
                             
        # Outflow
        du_dy[:, ny-2] = (u[:,ny-4] - 6*u[:,ny-3] + 3*u[:,ny-2] + 2*u[:,ny-1]) / (6 * dx)
        du_dy[:, ny-1] = (3*u[:,ny-1] - 4*u[:,ny-2] + u[:,ny-3]) / (2 * dx)
        
        return du_dx, du_dy

    return compute_derivatives

def exact_u_2d(X, Y, t, cx, cy):
    # Smooth exact solution
    return np.sin(np.pi * (X + Y - (cx + cy)*t))

def apply_bc_2d(u, X, Y, t, cx, cy):
    # Apply Dirichlet BC on the inflow faces: x=0 and y=0
    # u is shape (nx, ny)
    u[0, :] = exact_u_2d(0.0, Y[0,:], t, cx, cy)
    u[:, 0] = exact_u_2d(X[:,0], 0.0, t, cx, cy)
    return u

def step_rk3_2d(u, t, dt, dx, cx, cy, deriv_func, X, Y):
    # Stage 1
    u = apply_bc_2d(u, X, Y, t, cx, cy)
    du_dx, du_dy = deriv_func(u, dx)
    k1 = -(cx * du_dx + cy * du_dy)
    u1 = u + dt * k1
    u1 = apply_bc_2d(u1, X, Y, t + dt, cx, cy)
    
    # Stage 2
    du1_dx, du1_dy = deriv_func(u1, dx)
    k2 = -(cx * du1_dx + cy * du1_dy)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * k2
    u2 = apply_bc_2d(u2, X, Y, t + 0.5 * dt, cx, cy)
    
    # Stage 3
    du2_dx, du2_dy = deriv_func(u2, dx)
    k3 = -(cx * du2_dx + cy * du2_dy)
    u_new = (1.0/3.0) * u + (2.0/3.0) * u2 + (2.0/3.0) * dt * k3
    u_new = apply_bc_2d(u_new, X, Y, t + dt, cx, cy)
    
    return u_new

def run_eval_2d(w1, w2, T_end=0.2, dt_vals=None):
    deriv_func = make_dudx_2d_func(w1, w2)
    if dt_vals is None:
        dt_vals = np.array([0.01, 0.005, 0.0025, 0.00125])
    errors = []
    
    CFL = 0.5
    cx = 1.0
    cy = 1.0
    
    for dt in dt_vals:
        # For 2D advection, total speed is sqrt(cx^2 + cy^2). 
        # For simplicity, stability governed by cx*dt/dx + cy*dt/dy <= CFL
        # Since dx=dy, dt*(cx+cy)/dx <= CFL => dx = dt*(cx+cy)/CFL
        dx = dt * (cx + cy) / CFL
        nx = int(3.0 / dx) + 1
        x = np.linspace(0, 3.0, nx)
        y = np.linspace(0, 3.0, nx)
        dx_a = x[1] - x[0]
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        u = exact_u_2d(X, Y, 0.0, cx, cy)
        
        t = 0.0
        while t < T_end - 1e-12:
            step_dt = min(dt, T_end - t)
            u = step_rk3_2d(u, t, step_dt, dx_a, cx, cy, deriv_func, X, Y)
            t += step_dt
            
        u_exact = exact_u_2d(X, Y, T_end, cx, cy)
        err = np.sqrt(np.mean((u - u_exact)**2))
        errors.append(err)
        print(f"dt={dt}, err={err:.2e}")
        
    errors = np.array(errors)
    order = np.polyfit(np.log(dt_vals), np.log(errors), 1)[0]
    return dt_vals, errors, order

if __name__ == '__main__':
    print("=== 2D Linear Advection ===")
    print("Testing Standard Stencils:")
    dt_vals_std, err_std, ord_std = run_eval_2d(W1_STANDARD, W2_STANDARD)
    print(f"Standard  Order: {ord_std:.2f}\n")
    
    print("Testing Optimised Stencils:")
    dt_vals_opt, err_opt, ord_opt = run_eval_2d(W1_OPT, W2_OPT)
    print(f"Optimised Order: {ord_opt:.2f}\n")
    
    # Plotting with LaTeX formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pgf.preamble": r"\usepackage{amsmath}",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    
    plt.figure(figsize=(6, 5))
    plt.loglog(dt_vals_std, err_std, 'o-', color='#1f77b4', label=f'Standard ($\\mathcal{{O}}(\\Delta t^{{{ord_std:.2f}}})$)')
    plt.loglog(dt_vals_opt, err_opt, 's-', color='#ff7f0e', label=f'Optimised ($\\mathcal{{O}}(\\Delta t^{{{ord_opt:.2f}}})$)')
    plt.loglog(dt_vals_std, 1e-1 * dt_vals_std**2, 'k--', alpha=0.5, label='$\\mathcal{O}(\\Delta t^2)$')
    plt.loglog(dt_vals_std, 1e-1 * dt_vals_std**3, 'k:', alpha=0.5, label='$\\mathcal{O}(\\Delta t^3)$')
    
    plt.xlabel(r'Time Step $\Delta t$')
    plt.ylabel(r'$L_2$ Error')
    plt.title(r'2D Advection: $u_t + c_x u_x + c_y u_y = 0$')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('stencil_validation_2d.pdf', bbox_inches='tight')
    plt.savefig('stencil_validation_2d.pgf', bbox_inches='tight')
    print("Saved 2D validation plots to stencil_validation_2d.pdf/pgf")
