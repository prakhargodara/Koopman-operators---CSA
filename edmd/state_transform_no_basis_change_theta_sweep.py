
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from koopman_v2 import compute_edmd
from dsa import compute_dsa_distance
from ccd_v2 import compute_ccd_distance
from bijections import h_sin

# Same configuration as main.py
N_K          = 10
K_VALUES     = np.linspace(2.0, 4.0, N_K)
BASIS        = "fourier"
N_BASIS      = 25
N_DATA       = 100000
REG          = 1e-6
SEED         = 42
THETA_VALUES = np.round(np.linspace(0.0, 0.30, 16), 10)   # 0, 0.02, ..., 0.30
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "StateTransformNoBasisChangeResults")


def logistic(x, k):
    return k * x * (1.0 - x)


# -----------------------------------------------------------------------------
# Metrics (Appendix D): all N^2 entries
# -----------------------------------------------------------------------------

def fit_affine_distortion(D, B):
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()
    a, c = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    if a < 0:
        a, c = 0.0, float(np.mean(y))
    return float(np.sqrt(np.mean((y - a * x - c) ** 2)) / np.std(y))


def matrix_pearson(D, B):
    return float(np.corrcoef(np.ravel(B), np.ravel(D))[0, 1])


# -----------------------------------------------------------------------------
# Distance matrices
# -----------------------------------------------------------------------------

def distance_matrices(ops1, ops2, same):
    """Entry (i, j) = d(ops1[i], ops2[j]) for all ordered pairs.
    If same, ops1 and ops2 are one configuration and d(K, K) = 0."""
    n = len(ops1)
    D_dsa, D_ccd, Th = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if same and i == j:
                continue
            D_dsa[i, j] = compute_dsa_distance(ops1[i], ops2[j])
            D_ccd[i, j], Th[i, j] = compute_ccd_distance(
                ops1[i], ops2[j], BASIS, BASIS, N_BASIS)
    return D_dsa, D_ccd, Th


def build_operators():
    """Original operators and transformed operators for every theta,
    from the same samples as main.py."""
    np.random.seed(SEED)
    ops_ref, ops_trans = [], {th: [] for th in THETA_VALUES}
    for k in K_VALUES:
        x = np.random.uniform(0.0, 1.0, N_DATA)
        y = np.clip(logistic(x, k), 1e-10, 1 - 1e-10)
        ops_ref.append(compute_edmd(x, y, BASIS, N_BASIS, REG))
        for th in THETA_VALUES:
            ops_trans[th].append(
                compute_edmd(h_sin(x, th), h_sin(y, th), BASIS, N_BASIS, REG))
    return ops_ref, ops_trans


def _task(args):
    ops1, ops2, same = args
    return distance_matrices(ops1, ops2, same)


def run_experiment():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    ops_ref, ops_trans = build_operators()

    # Task 0: within-original reference; then, per theta, the comparison
    # matrix (orig vs trans) and the within-transformed reference.
    tasks = [(ops_ref, ops_ref, True)]
    for th in THETA_VALUES:
        tasks.append((ops_ref, ops_trans[th], False))
        tasks.append((ops_trans[th], ops_trans[th], True))
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(_task, tasks))

    O_dsa, O_ccd, _ = out[0]
    mats = {"D_ref_dsa": O_dsa, "D_ref_ccd": O_ccd}
    res = {key: [] for key in ("dsa_nrmse", "ccd_nrmse", "dsa_pearson",
                               "ccd_pearson", "ccd_theta_mean", "ccd_theta_std")}
    for it, th in enumerate(THETA_VALUES):
        D_dsa, D_ccd, Theta_hat = out[1 + 2 * it]
        T_dsa, T_ccd, _ = out[2 + 2 * it]

        # Symmetrized reference (Appendix D)
        B_dsa = 0.5 * (O_dsa + T_dsa)
        B_ccd = 0.5 * (O_ccd + T_ccd)

        res["dsa_nrmse"].append(fit_affine_distortion(D_dsa, B_dsa))
        res["ccd_nrmse"].append(fit_affine_distortion(D_ccd, B_ccd))
        res["dsa_pearson"].append(matrix_pearson(D_dsa, B_dsa))
        res["ccd_pearson"].append(matrix_pearson(D_ccd, B_ccd))
        # Recovered parameter on the conjugate pairs only
        res["ccd_theta_mean"].append(float(np.mean(np.diag(Theta_hat))))
        res["ccd_theta_std"].append(float(np.std(np.diag(Theta_hat))))

        for name, M in [("D_dsa", D_dsa), ("D_ccd", D_ccd), ("T_dsa", T_dsa),
                        ("T_ccd", T_ccd), ("theta_hat", Theta_hat)]:
            mats[f"{name}_{th:.2f}"] = M

        print(f"theta={th:.2f}  DSA nRMSE={res['dsa_nrmse'][-1]:.4f} "
              f"Pearson={res['dsa_pearson'][-1]:.4f} | "
              f"CSA nRMSE={res['ccd_nrmse'][-1]:.4f} "
              f"Pearson={res['ccd_pearson'][-1]:.4f} "
              f"theta_hat={res['ccd_theta_mean'][-1]:.4f}"
              f"+-{res['ccd_theta_std'][-1]:.4f}")

    np.save(os.path.join(OUT_DIR, "theta_values.npy"), THETA_VALUES)
    for key, vals in res.items():
        np.save(os.path.join(OUT_DIR, f"{key}.npy"), np.array(vals))
    np.savez(os.path.join(OUT_DIR, "distance_matrices.npz"),
             theta_values=THETA_VALUES, k_values=K_VALUES, **mats)
    print(f"\nSaved outputs to {OUT_DIR}/  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    run_experiment()
