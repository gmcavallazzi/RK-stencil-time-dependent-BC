import sys, os
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ========== Build Spatial Operator Matrix ==========
def build_spatial_operator(N, w1, w2):
    D = np.zeros((N, N))
    if N > 4:
        D[1, 0:5] = w1
        D[2, 0:5] = w2
    coeff = np.array([-2, 15, -60, 20, 30, -3]) / 60.0
    for i in range(3, N-2):
        if i-3 >= 0 and i+2 < N:
            D[i, i-3:i+3] = coeff
    if N > 4:
        D[-2, -4:] = [1/6, -1, 1/2, 1/3]
    D[-1, -3:] = [1/2, -2, 3/2]
    return D

# Load weights
rk3_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'optimised_stencils_stable.npz')
rk3_data = np.load(rk3_data_path)
w1_rk3, w2_rk3 = rk3_data['w1'], rk3_data['w2']

rk4_data_path = os.path.join(os.path.dirname(__file__), '..', 'rk4_extension', 'data', 'rk4_stencils_stable.npz')
rk4_data = np.load(rk4_data_path)
w1_rk4, w2_rk4 = rk4_data['w1'], rk4_data['w2']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

N = 80
dx = 3.0 / (N - 1)

# Cartesian grid for stability region
x = np.linspace(-4.0, 1.0, 500)
y = np.linspace(-3.5, 3.5, 500)
X, Y = np.meshgrid(x, y)
Z_grid = X + 1j * Y

# SSP-RK3
ax = axes[0]
R_rk3 = np.abs(1 + Z_grid + Z_grid**2/2 + Z_grid**3/6)
mask_rk3 = R_rk3 <= 1.0
dist_rk3 = np.where(mask_rk3, np.abs(Z_grid), np.nan)

contour_rk3 = ax.contourf(X, Y, dist_rk3, levels=30, cmap='YlGnBu', alpha=0.85)
ax.contour(X, Y, R_rk3, levels=[1.0], colors='k', linewidths=2)
ax.plot([], [], 'k-', lw=2, label=r'Stability Boundary')

D_rk3 = build_spatial_operator(N, w1_rk3, w2_rk3)
eigs_rk3 = np.linalg.eigvals(-D_rk3 / dx)

cfls_rk3 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
max_imag_rk3 = []
min_imag_rk3 = []
for cfl in cfls_rk3:
    # Filter out boundary outliers: continuous spectrum has Re(z) > -3, Im(z) < 2
    # So we only consider roots inside the main stability lobe
    Z = (cfl * dx) * eigs_rk3
    valid_eigs = []
    for val in Z:
        if val.real > -3.5 and val.imag < 2.0:
            valid_eigs.append(val)
    valid_eigs = np.array(valid_eigs)    
    
    idx = np.argmax(np.abs(valid_eigs))
    z_max = valid_eigs[idx]
    
    # Force positive imaginary part for the 'max' list so lines don't cross
    if z_max.imag < 0:
        z_max = np.conj(z_max)
        
    max_imag_rk3.append(z_max)
    min_imag_rk3.append(np.conj(z_max))  # The conjugate pair

max_imag_rk3 = np.array(max_imag_rk3)
min_imag_rk3 = np.array(min_imag_rk3)

ax.plot(max_imag_rk3.real, max_imag_rk3.imag, 'r-o', lw=2, ms=5, 
        markeredgecolor='k', label=r'Trajectory of max $|Z|$')
ax.plot(min_imag_rk3.real, min_imag_rk3.imag, 'r-o', lw=2, ms=5, 
        markeredgecolor='k')

# Classical RK4
ax = axes[1]
R_rk4 = np.abs(1 + Z_grid + Z_grid**2/2 + Z_grid**3/6 + Z_grid**4/24)
mask_rk4 = R_rk4 <= 1.0
dist_rk4 = np.where(mask_rk4, np.abs(Z_grid), np.nan)

contour_rk4 = ax.contourf(X, Y, dist_rk4, levels=30, cmap='YlGnBu', alpha=0.85)
ax.contour(X, Y, R_rk4, levels=[1.0], colors='k', linewidths=2)

D_rk4 = build_spatial_operator(N, w1_rk4, w2_rk4)
eigs_rk4 = np.linalg.eigvals(-D_rk4 / dx)

cfls_rk4 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
max_imag_rk4 = []
min_imag_rk4 = []
for cfl in cfls_rk4:
    Z = (cfl * dx) * eigs_rk4
    valid_eigs = []
    for val in Z:
        if val.real > -3.5 and val.imag < 2.0:
            valid_eigs.append(val)
    valid_eigs = np.array(valid_eigs)    
    
    idx = np.argmax(np.abs(valid_eigs))
    z_max = valid_eigs[idx]
    
    if z_max.imag < 0:
        z_max = np.conj(z_max)
        
    max_imag_rk4.append(z_max)
    min_imag_rk4.append(np.conj(z_max))

max_imag_rk4 = np.array(max_imag_rk4)
min_imag_rk4 = np.array(min_imag_rk4)

ax.plot(max_imag_rk4.real, max_imag_rk4.imag, 'r-o', lw=2, ms=5, markeredgecolor='k', label=r'Trajectory of max $|Z|$')
ax.plot(min_imag_rk4.real, min_imag_rk4.imag, 'r-o', lw=2, ms=5, markeredgecolor='k')

# Formatting
for _ax in axes:
    _ax.set_aspect('equal')
    _ax.set_xlabel(r'$\mathrm{Re}(z)$')
    _ax.set_ylabel(r'$\mathrm{Im}(z)$')
    _ax.grid(True, ls='--', alpha=0.3)
    _ax.legend(loc='lower left', framealpha=0.9)
    _ax.set_xlim([-3.5, 1.0])
    _ax.set_ylim([-3.5, 3.5])

axes[0].set_title(r'(a) SSP-RK3 Stability Region Depth \& Trajectory')
axes[1].set_title(r'(b) Classical RK4 Stability Region Depth \& Trajectory')

# Add a single colorbar for the whole figure
cbar = fig.colorbar(contour_rk4, ax=axes, orientation='horizontal', fraction=0.08, pad=0.15, aspect=40)
cbar.set_label(r'Distance from origin $|Z|$')

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'fig_stability_distance.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved {out_path}")
