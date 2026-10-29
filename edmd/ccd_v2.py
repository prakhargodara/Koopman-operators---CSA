import numpy as np
from scipy.optimize import minimize_scalar
from bases_v2 import evaluate_dict

THETA_MAX = 0.31       # just below 1/π ≈ 0.3183


def compute_ccd_distance(K1, K2, basis_type1, basis_type2, n_basis,
                          n_grid=2001, reg=1e-8):
    m = n_basis
    x_grid = np.linspace(0.00, 1., n_grid)

    Phi1 = evaluate_dict(x_grid, basis_type1, m)
    G    = Phi1.T @ Phi1 + reg * np.eye(m)
    GiPT = np.linalg.solve(G, Phi1.T)

    def objective(theta):
        h_x = x_grid + theta * np.sin(np.pi * x_grid)
        if h_x.min() < -0.01 or h_x.max() > 1.01:
            return 1e12
        h_x = np.clip(h_x, 0.0, 1.0)
        Phi2 = evaluate_dict(h_x, basis_type2, m)
        P    = (GiPT @ Phi2).T
        Pn   = np.linalg.norm(P, 'fro')
        if Pn < 1e-12:
            return 1e12
        diff = P @ K1 - K2 @ P
        return np.sum(diff ** 2) / (Pn ** 2)

    # Coarse grid search
    thetas = np.linspace(-THETA_MAX, THETA_MAX, 121)
    vals   = np.array([objective(t) for t in thetas])

    best_val   = np.min(vals)
    best_theta = thetas[np.argmin(vals)]

    # Refine around top-5 minima
    order = np.argsort(vals)
    for idx in order[:5]:
        lo = max(-THETA_MAX, thetas[idx] - 0.02)
        hi = min( THETA_MAX, thetas[idx] + 0.02)
        r  = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                             options={"maxiter": 2000, "xatol": 1e-12})
        if r.fun < best_val:
            best_val   = r.fun
            best_theta = r.x

    return np.sqrt(best_val), float(best_theta)
