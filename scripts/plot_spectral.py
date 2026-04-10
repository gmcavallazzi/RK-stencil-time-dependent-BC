import numpy as np
import matplotlib.pyplot as plt
from stencil_optimise import W1_STANDARD, W2_STANDARD

# W1 operates at node 1 using nodes 0..4
# So offsets for W1 are [-1, 0, 1, 2, 3]

# W2 operates at node 2 using nodes 0..4
# So offsets for W2 are [-2, -1, 0, 1, 2]

offsets_1 = np.array([-1, 0, 1, 2, 3])
offsets_2 = np.array([-2, -1, 0, 1, 2])

def modified_wavenumber(weights, offsets, kdx):
    # ik_mod * dx = sum_j w_j * exp(i * offset_j * kdx)
    # So k_mod * dx = Im( sum_j w_j * exp(i * offset_j * kdx) )
    # Numerical dispersion = Re( ...) -> should be minimal
    
    val = np.zeros_like(kdx, dtype=complex)
    for j, w in enumerate(weights):
        val += w * np.exp(1j * offsets[j] * kdx)
    
    return val.imag, val.real

if __name__ == '__main__':
    # Load optimised stencils
    data = np.load('optimised_stencils.npz')
    W1_OPT = data['w1']
    W2_OPT = data['w2']
    kdx = np.linspace(0, np.pi, 200)

    kmod_1_std, disp_1_std = modified_wavenumber(W1_STANDARD, offsets_1, kdx)
    kmod_2_std, disp_2_std = modified_wavenumber(W2_STANDARD, offsets_2, kdx)

    kmod_1_opt, disp_1_opt = modified_wavenumber(W1_OPT, offsets_1, kdx)
    kmod_2_opt, disp_2_opt = modified_wavenumber(W2_OPT, offsets_2, kdx)

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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Node 1 Dispersion (Real part, ideal is 0)
    axes[0,0].plot(kdx, kdx, 'k:', label='Exact', alpha=0.5)
    axes[0,0].plot(kdx, kmod_1_std, '-', color='#1f77b4', label=f'Standard 2nd-Order')
    axes[0,0].plot(kdx, kmod_1_opt, '-', color='#ff7f0e', label=f'Optimised')
    axes[0,0].set_xlabel(r'Exact Wavenumber $\kappa \Delta x$')
    axes[0,0].set_ylabel(r'Modified Wavenumber $\kappa_{mod} \Delta x$')
    axes[0,0].set_title('Node 1 ($i=1$): Resolution Characteristics')
    axes[0,0].legend()
    axes[0,0].grid(True, ls='--', alpha=0.4)

    axes[1,0].plot(kdx, np.zeros_like(kdx), 'k:', label='Exact (0)', alpha=0.5)
    axes[1,0].plot(kdx, disp_1_std, '-', color='#1f77b4', label=f'Standard')
    axes[1,0].plot(kdx, disp_1_opt, '-', color='#ff7f0e', label=f'Optimised')
    axes[1,0].set_xlabel(r'Exact Wavenumber $\kappa \Delta x$')
    axes[1,0].set_ylabel(r'Numerical Dissipation $\operatorname{Re}(\hat{D})$')
    axes[1,0].set_title('Node 1 ($i=1$): Dissipation Characteristics')
    axes[1,0].legend()
    axes[1,0].grid(True, ls='--', alpha=0.4)

    # Node 2
    axes[0,1].plot(kdx, kdx, 'k:', label='Exact', alpha=0.5)
    axes[0,1].plot(kdx, kmod_2_std, '-', color='#1f77b4', label=f'Standard 3rd-Order')
    axes[0,1].plot(kdx, kmod_2_opt, '-', color='#ff7f0e', label=f'Optimised')
    axes[0,1].set_xlabel(r'Exact Wavenumber $\kappa \Delta x$')
    axes[0,1].set_ylabel(r'Modified Wavenumber $\kappa_{mod} \Delta x$')
    axes[0,1].set_title('Node 2 ($i=2$): Resolution Characteristics')
    axes[0,1].legend()
    axes[0,1].grid(True, ls='--', alpha=0.4)

    axes[1,1].plot(kdx, np.zeros_like(kdx), 'k:', label='Exact (0)', alpha=0.5)
    axes[1,1].plot(kdx, disp_2_std, '-', color='#1f77b4', label=f'Standard')
    axes[1,1].plot(kdx, disp_2_opt, '-', color='#ff7f0e', label=f'Optimised')
    axes[1,1].set_xlabel(r'Exact Wavenumber $\kappa \Delta x$')
    axes[1,1].set_ylabel(r'Numerical Dissipation $\operatorname{Re}(\hat{D})$')
    axes[1,1].set_title('Node 2 ($i=2$): Dissipation Characteristics')
    axes[1,1].legend()
    axes[1,1].grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('stencil_spectral.pdf', bbox_inches='tight')
    plt.savefig('stencil_spectral.pgf', bbox_inches='tight')
    print("Saved plots to stencil_spectral.pdf and stencil_spectral.pgf")
