import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "edmd"))

from bases_v2 import evaluate_dict            # noqa: E402
from bijections import h_sin                  # noqa: E402
from koopman_v2 import compute_edmd           # noqa: E402

# ── fixed-dictionary experiment conventions (main.py) ────────────
N_K        = 10
K_VALUES   = np.linspace(2.0, 4.0, N_K)
N_BASIS    = 25
N_DATA     = 100000
REG_EDMD   = 1e-6
SEED       = 42
BASIS      = "fourier"

# ── CSA objective conventions (ccd_v2.py) ────────────────────────
N_GRID     = 2001
REG_PROJ   = 1e-8
THETA_MAX  = 0.31

# ── experiment grids ─────────────────────────────────────────────
THETA_STAR_GRID = np.round(np.linspace(0.0, 0.30, 31), 10)            # true parameter
THETA_GRID      = np.round(np.linspace(-THETA_MAX, THETA_MAX, 621), 10)  # candidates
CAPS            = [0.05, 0.10, 0.15, 0.20, THETA_MAX]                  # search caps
THETA_STAR_REF  = 0.10

OUT_DIR = os.path.join(HERE, "Results")
os.makedirs(OUT_DIR, exist_ok=True)


def logistic(x, k):
    return k * x * (1.0 - x)


# ── operators K1(k) and K2(k, theta*) from the same data draw ────
print("Building EDMD operators ...")
t0 = time.time()
np.random.seed(SEED)
K1 = []
K2 = np.zeros((N_K, len(THETA_STAR_GRID), N_BASIS, N_BASIS))
for ik, k in enumerate(K_VALUES):
    x = np.random.uniform(0.0, 1.0, N_DATA)
    y = np.clip(logistic(x, k), 1e-10, 1 - 1e-10)
    K1.append(compute_edmd(x, y, BASIS, N_BASIS, REG_EDMD))
    for its, ts in enumerate(THETA_STAR_GRID):
        K2[ik, its] = compute_edmd(h_sin(x, ts), h_sin(y, ts), BASIS, N_BASIS, REG_EDMD)
print(f"  done in {time.time()-t0:.1f}s")

# ── P_theta on the candidate grid (as in ccd_v2.compute_ccd_distance) ─
x_grid = np.linspace(0.0, 1.0, N_GRID)
Phi1 = evaluate_dict(x_grid, BASIS, N_BASIS)
GiPT = np.linalg.solve(Phi1.T @ Phi1 + REG_PROJ * np.eye(N_BASIS), Phi1.T)


def transfer(theta):
    return (GiPT @ evaluate_dict(np.clip(h_sin(x_grid, theta), 0.0, 1.0),
                                 BASIS, N_BASIS)).T


P_all = np.stack([transfer(th) for th in THETA_GRID])
Pn_all = np.linalg.norm(P_all, axis=(1, 2))


def objective(K1m, K2m):
    def obj(theta):
        P = transfer(theta)
        return np.sum((P @ K1m - K2m @ P) ** 2) / np.linalg.norm(P) ** 2
    return obj


# ── landscapes J(h_theta) for every k and theta* ──────────────────
print("Computing landscapes ...")
t0 = time.time()
J = np.zeros((N_K, len(THETA_STAR_GRID), len(THETA_GRID)))
for ik in range(N_K):
    for its in range(len(THETA_STAR_GRID)):
        diff = P_all @ K1[ik] - K2[ik, its] @ P_all
        J[ik, its] = np.sqrt(np.sum(diff ** 2, axis=(1, 2))) / Pn_all
print(f"  done in {time.time()-t0:.1f}s")

# ── capped minimisation: grid argmin + bounded Brent refinement ───
print("Capped minimisation ...")
t0 = time.time()
DMIN = np.zeros((len(CAPS), N_K, len(THETA_STAR_GRID)))
THHAT = np.zeros_like(DMIN)
for ic, cap in enumerate(CAPS):
    mask = np.abs(THETA_GRID) <= cap + 1e-12
    th_m = THETA_GRID[mask]
    for ik in range(N_K):
        for its in range(len(THETA_STAR_GRID)):
            vals = J[ik, its][mask]
            i0 = int(np.argmin(vals))
            best_th, best_d = th_m[i0], vals[i0]
            lo, hi = max(-cap, best_th - 0.002), min(cap, best_th + 0.002)
            if hi > lo + 1e-13:
                r = minimize_scalar(objective(K1[ik], K2[ik, its]), bounds=(lo, hi),
                                    method="bounded",
                                    options={"maxiter": 300, "xatol": 1e-12})
                if np.sqrt(r.fun) < best_d:
                    best_d, best_th = np.sqrt(r.fun), float(r.x)
            DMIN[ic, ik, its] = best_d
            THHAT[ic, ik, its] = best_th
print(f"  done in {time.time()-t0:.1f}s")

its_ref = int(np.argmin(np.abs(THETA_STAR_GRID - THETA_STAR_REF)))
th_rec = THHAT[CAPS.index(THETA_MAX), :, its_ref]
print(f"theta* = {THETA_STAR_REF}: max |theta_hat - theta*| over k = "
      f"{np.max(np.abs(th_rec - THETA_STAR_REF)):.2e}")

path = os.path.join(OUT_DIR, "theta_landscape.npz")
np.savez_compressed(
    path,
    k_values=K_VALUES, theta_star_grid=THETA_STAR_GRID, theta_grid=THETA_GRID,
    caps=np.array(CAPS), theta_star_ref=THETA_STAR_REF,
    J_ref=J[:, its_ref], DMIN=DMIN, THHAT=THHAT,
)
print(f"Saved {path}")
