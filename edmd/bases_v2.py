import numpy as np


def fourier_dict(x, n_basis):
    nf = (n_basis - 1) // 2
    Phi = np.ones((len(x), 1 + 2 * nf))
    for i in range(1, nf + 1):
        Phi[:, 2 * i - 1] = np.cos(2 * np.pi * i * x)
        Phi[:, 2 * i]     = np.sin(2 * np.pi * i * x)
    return Phi


def legendre_dict(x, n_basis):
    t = 2.0 * x - 1.0
    N = len(x)
    Phi = np.empty((N, n_basis))
    Phi[:, 0] = 1.0
    if n_basis > 1:
        Phi[:, 1] = t
    for n in range(2, n_basis):
        Phi[:, n] = ((2 * n - 1) * t * Phi[:, n - 1]
                     - (n - 1) * Phi[:, n - 2]) / n
    return Phi


def evaluate_dict(x, basis_type, n_basis):
    if basis_type == "fourier":  return fourier_dict(x, n_basis)
    if basis_type == "legendre": return legendre_dict(x, n_basis)
    raise ValueError(basis_type)
