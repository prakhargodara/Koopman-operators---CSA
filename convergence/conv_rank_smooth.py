import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_KRY = os.path.abspath(os.path.join(_HERE, "..", "krylov"))
sys.path.insert(0, _KRY)

from maps import generate_time_series
from krylov_subspace import build_delay_matrix, project_to_subspace
from paper_section3_smooth_afreq import (
    build_systems, build_data, fit_at_rank,
    A_REF, N_REF, N_SERIES, DELAY_DIM, TS_LENGTH, SEED,
)


# ---------------------------------------------------------------------
# Configuration: Figure 5 setup at (a*, n*) = (0.5, 4)
# ---------------------------------------------------------------------
A_STAR, N_STAR = A_REF, N_REF
RANKS   = [2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 70, 80, 99]   # includes all Fig. 5c ranks
N_TEST  = 100        # held-out trajectories per system
TS_TEST = TS_LENGTH


def gen_trajs(map_fn, params, n_series, ts_length, seed):
    rng = np.random.default_rng(seed)
    return [generate_time_series(map_fn, params, rng.uniform(0.01, 0.99),
                                 ts_length)
            for _ in range(n_series)]


def fit_state_readout(H, info):
    c, *_ = np.linalg.lstsq(project_to_subspace(H, info), H[:, 0],
                            rcond=None)
    return c


def residual(test_trajs, info, K, c):
    num = 0.0
    den = 0.0
    for ts in test_trajs:
        H = build_delay_matrix(ts, DELAY_DIM)
        Z = project_to_subspace(H, info)
        x_pred = (Z[:-1] @ K.T) @ c
        x_true = H[1:, 0]
        num += float(np.sum((x_true - x_pred) ** 2))
        den += float(np.sum(x_true ** 2))
    return num / den


def run():
    out_dir = os.path.join(_HERE, "Results")
    os.makedirs(out_dir, exist_ok=True)

    systems = build_systems(A_STAR, N_STAR)
    n_sys = len(systems)
    rho = np.empty((n_sys, len(RANKS)))
    eff_rank = np.empty((n_sys, len(RANKS)), dtype=int)

    print("=== Smooth-bijection rank-residual sweep (Figure 5 construction) ===")
    print(f"(a*, n*) = ({A_STAR}, {N_STAR}), ranks = {RANKS}, "
          f"delay_dim = {DELAY_DIM}, {N_SERIES} training trajectories\n",
          flush=True)

    t_start = time.time()
    H = build_data(systems, seed=SEED)          # same data as Fig. 5c
    test = [gen_trajs(mf, p, N_TEST, TS_TEST, seed=SEED + 100_000 + si)
            for si, (_, mf, p) in enumerate(systems)]

    for ri, r in enumerate(RANKS):
        Ks, infos = fit_at_rank(H, r)
        for si, (label, _, _) in enumerate(systems):
            c = fit_state_readout(H[si], infos[si])
            rho[si, ri] = residual(test[si], infos[si], Ks[si], c)
            eff_rank[si, ri] = infos[si].rank
            print(f"  r={r:>3}  r'={eff_rank[si, ri]:>3}  [{label:>28s}]  "
                  f"rho={rho[si, ri]:.3e}", flush=True)

    out_path = os.path.join(out_dir, "conv_rank_smooth.npz")
    np.savez(
        out_path,
        labels=np.array([s[0] for s in systems]),
        rates=np.asarray([s[2] for s in systems], dtype=float),
        kinds=np.array(["lin", "lin", "smooth", "smooth"]),
        ranks=np.asarray(RANKS, dtype=int),
        eff_rank=eff_rank,
        rho=rho,
        a_star=A_STAR, n_star=N_STAR, delay_dim=DELAY_DIM,
        n_series=N_SERIES, ts_length=TS_LENGTH,
    )
    print(f"\nsaved {out_path}  ({time.time()-t_start:.1f}s total)",
          flush=True)


if __name__ == "__main__":
    run()
