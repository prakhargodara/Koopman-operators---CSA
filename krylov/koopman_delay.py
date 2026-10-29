import numpy as np
from krylov_subspace import project_to_subspace


def fit_koopman_delay(H, info, reg=1e-8, rows_per_traj=None):
    X = H[:-1, :]   # (N-1, d)
    Y = H[1:, :]    # (N-1, d)
    if rows_per_traj is not None:
        # TODO drop the pairs (last row of one trajectory, first row of the next)
        keep = (np.arange(H.shape[0] - 1) + 1) % rows_per_traj != 0
        X, Y = X[keep], Y[keep]

    X_proj = project_to_subspace(X, info)  # (N-1, r)
    Y_proj = project_to_subspace(Y, info)  # (N-1, r)

    r = info.rank
    G = X_proj.T @ X_proj + reg * np.eye(r)
    KT = np.linalg.solve(G, X_proj.T @ Y_proj)   # (r, r)
    return KT.T   # K such that Y_proj^T ~ K X_proj^T
