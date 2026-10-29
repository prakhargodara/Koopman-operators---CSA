import numpy as np
from bases_v2 import evaluate_dict


def compute_edmd(x_data, y_data, basis_type, n_basis, reg=1e-8):
    Phi_X = evaluate_dict(x_data, basis_type, n_basis)
    Phi_Y = evaluate_dict(y_data, basis_type, n_basis)
    G  = Phi_X.T @ Phi_X + reg * np.eye(n_basis)
    KT = np.linalg.solve(G, Phi_X.T @ Phi_Y)
    return KT.T
