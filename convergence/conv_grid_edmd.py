
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PC = os.path.abspath(os.path.join(_HERE, "..", "edmd"))
sys.path.insert(0, _PC)

from koopman_v2 import compute_edmd
from bases_v2 import evaluate_dict


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------
K_GRID  = np.round(np.linspace(2.0, 4.0, 11), 3)       # 2.0, 2.2, ..., 4.0
N_BASIS = [11, 21, 41, 81, 121]                         # dictionary size sweep
BASES   = ["fourier", "legendre"]                       # EDMD dictionaries

N_DATA   = 100_000     # training samples per fit
N_TEST   = 50_000      # held-out samples for rho
REG      = 1.0e-6      # EDMD ridge
SEED     = 42


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def logistic(x, k):
    return k * x * (1.0 - x)


def sample_xy(k, n, seed):
    """Training pairs for f_k: x ~ Uniform(0,1), y = f_k(x)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.clip(logistic(x, k), 1e-10, 1.0 - 1e-10)
    return x, y


def fit_state_readout(x_tr, basis, n_basis):
    Phi_X = evaluate_dict(x_tr, basis, n_basis)
    c, *_ = np.linalg.lstsq(Phi_X, x_tr, rcond=None)
    return c


def residual(x, y, K, c, basis, n_basis):
    Phi_X = evaluate_dict(x, basis, n_basis)
    pred  = (Phi_X @ K.T) @ c
    return float(np.sum((y - pred) ** 2)
                 / max(np.sum(y ** 2), 1.0e-30))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def run():
    out_dir = os.path.join(_HERE, "Results")
    os.makedirs(out_dir, exist_ok=True)

    n_basis_arr = np.asarray(N_BASIS, dtype=int)
    n_k = len(K_GRID)
    n_nb = len(N_BASIS)
    n_bases = len(BASES)

    rho_f = np.empty((n_bases, n_nb, n_k))

    t_start = time.time()
    print(f"EDMD grid sweep: {n_bases} bases x {n_nb} truncations x {n_k} k-values")
    print(f"  K_GRID  = {list(K_GRID)}")
    print(f"  N_BASIS = {N_BASIS}")
    print(f"  BASES   = {BASES}")

    for bi, basis in enumerate(BASES):
        for ni, nb in enumerate(N_BASIS):
            for ki, k in enumerate(K_GRID):
                t0 = time.time()

                x_tr, y_tr = sample_xy(k, N_DATA, SEED + ki)
                x_te, y_te = sample_xy(k, N_TEST, SEED + 10_000 + ki)

                K_f = compute_edmd(x_tr, y_tr, basis, nb, REG)
                c_f = fit_state_readout(x_tr, basis, nb)
                rho_f[bi, ni, ki] = residual(x_te, y_te, K_f, c_f, basis, nb)

                print(f"  [{basis:8s} nb={nb:3d} k={k:.2f}]  "
                      f"rho_f={rho_f[bi,ni,ki]:.3e}  "
                      f"({time.time()-t0:.1f}s)")

    out_path = os.path.join(out_dir, "conv_grid_edmd.npz")
    np.savez(
        out_path,
        K_GRID=np.asarray(K_GRID, dtype=float),
        N_BASIS=n_basis_arr,
        BASES=np.asarray(BASES),
        N_DATA=N_DATA, N_TEST=N_TEST, REG=REG,
        rho_f=rho_f,
    )
    print(f"\nsaved {out_path}  ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    run()
