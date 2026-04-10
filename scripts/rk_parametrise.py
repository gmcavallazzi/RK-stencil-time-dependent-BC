"""
Parameterisation of s-stage, order-p explicit Runge-Kutta methods.

For s=4, p=3: 10 tableau entries, 4 order conditions → 6 free parameters.

Free parameters: c2, c3, c4, a32, a42, a43
Derived:          a21 = c2
                  a31 = c3 - a32
                  a41 = c4 - a42 - a43
                  b1..b4 solved from 4 order conditions (linear system)
"""

import numpy as np


def build_tableau_4s3p(free_params):
    """
    Construct a 4-stage, order-3 explicit RK Butcher tableau.

    Parameters
    ----------
    free_params : array-like, shape (6,)
        [c2, c3, c4, a32, a42, a43]

    Returns
    -------
    A : (4,4) array   — strictly lower-triangular
    b : (4,) array    — weights
    c : (4,) array    — nodes
    valid : bool       — whether the order conditions could be solved
    """
    c2, c3, c4, a32, a42, a43 = free_params

    # Derived a_ij
    a21 = c2
    a31 = c3 - a32
    a41 = c4 - a42 - a43

    c = np.array([0.0, c2, c3, c4])

    A = np.array([
        [0,   0,   0,   0],
        [a21, 0,   0,   0],
        [a31, a32, 0,   0],
        [a41, a42, a43, 0],
    ])

    # Solve for b from order conditions (linear in b):
    # (1)  b1 + b2 + b3 + b4 = 1
    # (2)  c2*b2 + c3*b3 + c4*b4 = 1/2
    # (3)  c2^2*b2 + c3^2*b3 + c4^2*b4 = 1/3
    # (4)  a32*c2*b3 + (a42*c2 + a43*c3)*b4 = 1/6
    #
    # Rewrite as 3×3 for b2, b3, b4, then b1 = 1 - b2 - b3 - b4

    M = np.array([
        [c2,      c3,              c4             ],
        [c2**2,   c3**2,           c4**2          ],
        [0.0,     a32*c2,          a42*c2 + a43*c3],
    ])
    rhs = np.array([0.5, 1.0/3.0, 1.0/6.0])

    try:
        b234 = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return A, np.zeros(4), c, False

    b = np.array([1.0 - b234.sum(), b234[0], b234[1], b234[2]])

    return A, b, c, True


def stability_function(A, b, z):
    """
    Evaluate the stability function R(z) of an explicit RK method.
    R(z) = 1 + z b^T (I - zA)^{-1} e
    For explicit methods: R(z) = 1 + z*b1 + z^2*b^T*A*e + ...
    """
    s = len(b)
    e = np.ones(s)
    I = np.eye(s)
    try:
        R = 1.0 + z * b @ np.linalg.solve(I - z * A, e)
    except np.linalg.LinAlgError:
        R = np.inf
    return R


def max_cfl_imaginary(A, b, n_points=500):
    """
    Maximum CFL along the imaginary axis (advection stability).
    Find max |beta| such that |R(i*beta)| <= 1.
    """
    betas = np.linspace(0, 4, n_points)
    for beta in betas:
        z = 1j * beta
        if abs(stability_function(A, b, z)) > 1.0 + 1e-10:
            return max(0, beta - 4.0/n_points)
    return betas[-1]


def max_cfl_negative_real(A, b, n_points=500):
    """
    Maximum CFL along the negative real axis (diffusion stability).
    Find max |alpha| such that |R(-alpha)| <= 1.
    """
    alphas = np.linspace(0, 6, n_points)
    for alpha in alphas:
        z = -alpha
        if abs(stability_function(A, b, z)) > 1.0 + 1e-10:
            return max(0, alpha - 6.0/n_points)
    return alphas[-1]


def verify_order_conditions(A, b, c, tol=1e-12):
    """Check the first 4 order conditions for an order-3 method."""
    s = len(b)
    e = np.ones(s)
    checks = {
        'sum(b)=1':       abs(b.sum() - 1.0),
        'b·c=1/2':        abs(b @ c - 0.5),
        'b·c²=1/3':       abs(b @ (c**2) - 1.0/3.0),
        'b·A·c=1/6':      abs(b @ (A @ c) - 1.0/6.0),
    }
    all_ok = all(v < tol for v in checks.values())
    return checks, all_ok


# --- Well-known methods for comparison ---

def ssp_rk3():
    """Shu-Osher SSP-RK3 (3-stage, order 3)."""
    A = np.array([
        [0,    0,    0],
        [1,    0,    0],
        [0.25, 0.25, 0],
    ])
    b = np.array([1/6, 1/6, 2/3])
    c = np.array([0, 1, 0.5])
    return A, b, c


def classical_rk4():
    """Classical 4-stage RK4."""
    A = np.array([
        [0,   0,   0,   0],
        [0.5, 0,   0,   0],
        [0,   0.5, 0,   0],
        [0,   0,   0,   1],
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0, 0.5, 0.5, 1])
    return A, b, c


# --- WSO methods from Biswas et al. (arXiv:2310.02817) ---

def erk312():
    """
    ERK312: 4-stage, order 3, WSO 2.
    From Skvortsov (2006), referenced in Biswas et al. Eq. (3.5).
    c = [0, 1/2, 1, 1].

    Stability function: R(z) = 1 + z + z^2/2 + z^3/6 (same as all (4,3,2) methods).
    """
    A = np.array([
        [0,     0,    0,    0],
        [1/2,   0,    0,    0],
        [-1,    2,    0,    0],
        [1/6,  2/3,  1/6,   0],
    ])
    b = np.array([1/6, 2/3, 1/6, 0])
    c = np.array([0, 1/2, 1, 1])
    return A, b, c


def erk313():
    """
    ERK313: 5-stage, order 3, WSO 3.
    From Skvortsov (2006), Eq. (3.6) right in Biswas et al.
    c = [0, 1/3, 2/3, 1, 0].
    """
    A = np.array([
        [0,       0,      0,     0,    0],
        [1/3,     0,      0,     0,    0],
        [2/3,     0,      0,     0,    0],
        [1,       0,      0,     0,    0],
        [-11/12,  3/2,   -3/4,   1/6,  0],
    ])
    b = np.array([1/4, -3, 15/4, -1, 1])
    c = np.array([0, 1/3, 2/3, 1, 0])
    return A, b, c


def biswas_533():
    """
    Biswas (5,3,3): 5-stage, order 3, WSO 3.
    From Biswas et al. (arXiv:2310.02817), Eq. (3.6) left.
    c = [0, 3/11, 15/19, 5/6, 1].
    """
    c2 = 3/11
    c3 = 15/19

    A = np.array([
        [0,                  0,                   0,           0,           0],
        [3/11,               0,                   0,           0,           0],
        [285645/493487,      103950/493487,       0,           0,           0],
        [3075805/5314896,    1353275/5314896,      0,           0,           0],
        [196687/177710,      -129383023/426077496, 48013/42120, -2268/2405,  0],
    ])
    b = np.array([5626/4725, -25289/13608, 569297/340200, 324/175, -13/7])
    c = np.array([0, 3/11, 15/19, 5/6, 1])
    return A, b, c


def biswas_643():
    """
    Biswas (6,4,3): 6-stage, order 4, WSO 3.
    From Biswas et al. (arXiv:2310.02817v2), Appendix A.1.
    c = [0, 1, 1/7, 8/11, 5/9, 4/5].
    """
    A = np.array([
        [0,               0,               0,                  0,                0,               0],
        [1,               0,               0,                  0,                0,               0],
        [461/3920,        99/3920,         0,                  0,                0,               0],
        [314/605,         126/605,         0,                  0,                0,               0],
        [13193/197316,    39332/443961,    86632/190269,      -294151/5327532,   0,               0],
        [884721/773750,   52291/696375,   -155381744/135793125, -53297233/355151250, 74881422/85499375, 0],
    ])
    b = np.array([113/2880, 7/1296, 91238/363285, -1478741/1321920, 147987/194480, 77375/72864])
    c = np.array([0, 1, 1/7, 8/11, 5/9, 4/5])
    return A, b, c


def wso3_defects(A, b, c):
    """
    Compute WSO defect values for an RK method.
    Returns (defect_a, defect_b) — both should be 0 for WSO >= 3.
    """
    tau2 = c**2 / 2 - A @ c
    tau3 = c**3 / 6 - A @ (c**2 / 2)
    defect_a = b @ tau3
    defect_b = b @ (A @ tau2)
    return defect_a, defect_b


if __name__ == '__main__':
    # Quick test: try to recover a known method
    # SSP-RK3 has c = [0, 1, 0.5], a32 = 0.25
    # Let's build a 4-stage order-3 method with c = [0, 1, 0.5, 1]
    free = [1.0, 0.5, 1.0, 0.25, 0.5, 0.0]
    A, b, c, valid = build_tableau_4s3p(free)
    print(f"Valid: {valid}")
    print(f"A:\n{A}")
    print(f"b: {b}")
    print(f"c: {c}")

    checks, ok = verify_order_conditions(A, b, c)
    print(f"\nOrder conditions: {'PASS' if ok else 'FAIL'}")
    for name, val in checks.items():
        print(f"  {name}: {val:.2e}")

    cfl_im = max_cfl_imaginary(A, b)
    cfl_re = max_cfl_negative_real(A, b)
    print(f"\nMax CFL (imaginary axis): {cfl_im:.3f}")
    print(f"Max CFL (negative real):  {cfl_re:.3f}")
