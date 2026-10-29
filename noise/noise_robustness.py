import os
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "edmd"))

from bases_v2 import evaluate_dict   # noqa: E402
from bijections import h_sin         # noqa: E402
from koopman_v2 import compute_edmd  # noqa: E402

NK, M, REG, EPS = 10, 25, 1e-6, 0.1
KV = np.linspace(2.0, 4.0, NK)
REG_P, NGRID = 1e-8, 2001
THG = np.round(np.linspace(-0.31, 0.31, 621), 10)
N_LIST = [1000, 10000, 100000]
SIG_LIST = [0.0, 0.01, 0.02, 0.05]
SEEDS = [42, 43, 44, 45, 46]

OUT_DIR = os.path.join(HERE, "Results")
os.makedirs(OUT_DIR, exist_ok=True)

xg = np.linspace(0.0, 1.0, NGRID)
Phi1 = evaluate_dict(xg, "fourier", M)
G = Phi1.T @ Phi1 + REG_P * np.eye(M)
W = np.linalg.solve(G, Phi1.T)
P_grid = np.empty((len(THG), M, M))
for i, th in enumerate(THG):
    P_grid[i] = (W @ evaluate_dict(np.clip(h_sin(xg, th), 0, 1), "fourier", M)).T
Pn_grid = np.linalg.norm(P_grid, axis=(1, 2))


def logistic(x, k):
    return k * x * (1.0 - x)


def csa(K1, K2):
    diff = P_grid @ K1 - K2 @ P_grid
    d = np.sqrt(np.sum(diff**2, axis=(1, 2))) / Pn_grid
    i0 = int(np.argmin(d))
    best_d, best_th = d[i0], THG[i0]

    def obj(th):
        P = (W @ evaluate_dict(np.clip(h_sin(xg, th), 0, 1), "fourier", M)).T
        return np.sum((P @ K1 - K2 @ P)**2) / np.linalg.norm(P)**2

    r = minimize_scalar(obj, bounds=(max(-0.31, best_th - 0.002),
                                     min(0.31, best_th + 0.002)),
                        method="bounded", options={"xatol": 1e-10})
    if np.sqrt(r.fun) < best_d:
        best_d, best_th = np.sqrt(r.fun), float(r.x)
    return best_d, best_th


def one_run(N, sig, seed):
    rng = np.random.default_rng(seed)
    K1s, K2s = [], []
    for k in KV:
        x = rng.uniform(0.0, 1.0, N)
        y = np.clip(logistic(x, k), 1e-10, 1 - 1e-10)
        u, v = h_sin(x, EPS), h_sin(y, EPS)
        e = [rng.standard_normal(N) for _ in range(4)]   # drawn even if sig=0
        xo = np.clip(x + sig * e[0], 0, 1); yo = np.clip(y + sig * e[1], 0, 1)
        uo = np.clip(u + sig * e[2], 0, 1); vo = np.clip(v + sig * e[3], 0, 1)
        K1s.append(compute_edmd(xo, yo, "fourier", M, REG))
        K2s.append(compute_edmd(uo, vo, "fourier", M, REG))
    D = np.zeros((NK, NK)); TH = np.zeros((NK, NK))
    OO = np.zeros((NK, NK)); TT = np.zeros((NK, NK))
    for i in range(NK):
        for j in range(NK):
            D[i, j], TH[i, j] = csa(K1s[i], K2s[j])
            if i != j:
                OO[i, j], _ = csa(K1s[i], K1s[j])
                TT[i, j], _ = csa(K2s[i], K2s[j])
    return D, TH, OO, TT


if __name__ == "__main__":
    out = {"N_list": np.array(N_LIST), "sigma_list": np.array(SIG_LIST),
           "seeds": np.array(SEEDS), "k_values": KV, "theta_true": EPS}
    for N in N_LIST:
        for sig in SIG_LIST:
            t0 = time.time()
            runs = [one_run(N, sig, s) for s in SEEDS]
            for name, idx in [("D", 0), ("TH", 1), ("OO", 2), ("TT", 3)]:
                out[f"{name}_N{N}_s{sig}"] = np.stack([r[idx] for r in runs])
            d0 = np.mean([np.diag(r[0]).mean() for r in runs])
            err = np.mean([np.abs(np.diag(r[1]) - EPS).mean() for r in runs])
            print(f"N={N:6d} sig={sig:5.2f}  d0={d0:.3f}  "
                  f"|th-th*|={err:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    path = os.path.join(OUT_DIR, "noise_robustness.npz")
    np.savez(path, **out)
    print(f"\nSaved {path}")
