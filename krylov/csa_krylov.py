import numpy as np

from krylov_subspace import (
    build_krylov_observables,
    project_to_subspace,
    SubspaceInfo,
)


def _build_projected_krylov(x, map_fn, param, info):
    H = build_krylov_observables(x, map_fn, param, info.delay_dim)
    return project_to_subspace(H, info)


def compute_csa_krylov_distance(
    K1, K2,
    map_fn1, param1, map_fn2, param2,
    info1, info2,
    conjugacy_fn,
    param_grids,
    n_grid=800, reg=1e-8,
    n_refine=10,
    x_grid=None,
):
    from itertools import product as cart_product

    if x_grid is None:
        x_grid = np.linspace(0.005, 0.995, n_grid)

    # Pre-compute system-1 Krylov observables on the grid
    Phi1 = _build_projected_krylov(x_grid, map_fn1, param1, info1)
    r1 = info1.rank
    GtG = Phi1.T @ Phi1
    # Adaptive regularization: scale by largest eigenvalue so that
    # the effective condition number is bounded by ~1/reg
    lam_max = np.linalg.norm(GtG, 2)
    reg_scaled = max(reg * lam_max, reg)
    G = GtG + reg_scaled * np.eye(r1)
    GiPT = np.linalg.solve(G, Phi1.T)

    def objective(params):
        h_x = conjugacy_fn(x_grid, params)
        if h_x.min() < -0.01 or h_x.max() > 1.01:
            return 1e12
        h_x = np.clip(h_x, 0.0, 1.0)

        Phi2 = _build_projected_krylov(h_x, map_fn2, param2, info2)
        P = (GiPT @ Phi2).T
        Pn = np.linalg.norm(P, 'fro')
        if Pn < 1e-12:
            return 1e12
        diff = P @ K1 - K2 @ P
        return np.sum(diff ** 2) / (Pn ** 2)

    # --- Coarse grid search over Cartesian product ---
    grid_points = list(cart_product(*param_grids))
    vals = np.array([objective(p) for p in grid_points])

    best_val = np.min(vals)
    best_params = grid_points[np.argmin(vals)]

    # --- Refine around top candidates ---
    bounds = [(float(g[0]), float(g[-1])) for g in param_grids]
    order = np.argsort(vals)
    n_dims = len(param_grids)

    if n_dims == 1:
        # 1-D: use minimize_scalar (Brent) — much more robust than L-BFGS-B
        # for objectives with noisy finite-difference gradients
        from scipy.optimize import minimize_scalar as _minimize_scalar
        lo_bound, hi_bound = bounds[0]
        for idx in order[:min(n_refine, len(order))]:
            if vals[idx] > best_val * 100 and vals[idx] > 1.0:
                continue
            p0 = float(grid_points[idx][0])
            # Search in a window around this candidate
            step = (hi_bound - lo_bound) / max(len(param_grids[0]) - 1, 1)
            bracket_lo = max(lo_bound, p0 - 1.5 * step)
            bracket_hi = min(hi_bound, p0 + 1.5 * step)
            r = _minimize_scalar(
                lambda a: objective((a,)),
                bounds=(bracket_lo, bracket_hi), method="bounded",
                options={"maxiter": 2000, "xatol": 1e-13})
            if r.fun < best_val:
                best_val = r.fun
                best_params = (r.x,)
    else:
        # Multi-D: use L-BFGS-B
        from scipy.optimize import minimize as _minimize
        for idx in order[:min(n_refine, len(order))]:
            if vals[idx] > best_val * 100 and vals[idx] > 1.0:
                continue
            p0 = np.array(grid_points[idx], dtype=float)
            r = _minimize(lambda p: objective(tuple(p)), p0,
                          method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-14})
            if r.fun < best_val:
                best_val = r.fun
                best_params = tuple(r.x)

    return np.sqrt(max(best_val, 0.0)), tuple(float(v) for v in best_params)
