import os
import numpy as np, time
from koopman_v2 import compute_edmd
from dsa import compute_dsa_distance
from ccd_v2 import compute_ccd_distance
from bijections import h_sin

# ═══════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════
N_K          = 10
K_VALUES     = np.linspace(2.0, 4.0, N_K)
EPS_TRUE     = 0.1
N_BASIS      = 25
N_DATA       = 100000
REG          = 1e-6
SEED         = 42
DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")


def logistic(x, k):
    return k * x * (1.0 - x)


# ═══════════════════════════════════════════════════════════════
#  Build operators
# ═══════════════════════════════════════════════════════════════
np.random.seed(SEED)
n_sys = 2 * N_K

BASIS_LIST = ["fourier", "legendre"]

ops = {b: [] for b in BASIS_LIST}

for k in K_VALUES:
    x = np.random.uniform(0.0, 1.0, N_DATA)
    y = np.clip(logistic(x, k), 1e-10, 1 - 1e-10)

    hx = h_sin(x, EPS_TRUE)
    hy = h_sin(y, EPS_TRUE)

    for b in BASIS_LIST:
        Ko = compute_edmd(x,  y,  b, N_BASIS, REG)
        Kt = compute_edmd(hx, hy, b, N_BASIS, REG)
        ops[b].append(Ko)
        ops[b].append(Kt)

print(f"Built {n_sys} systems × {len(BASIS_LIST)} bases")
print(f"  N_K={N_K}  m={N_BASIS}  N={N_DATA}  eps={EPS_TRUE}\n")


# ═══════════════════════════════════════════════════════════════
#  Pairwise distances
# ═══════════════════════════════════════════════════════════════
def run(ops1, ops2, b1, b2, tag, nb):
    """Return DSA matrix and CCD matrix for all ordered pairs (i, j)."""
    n = len(ops1)
    same = ops1 is ops2          # within-basis run: d(K, K) = 0 on the diagonal
    pairs = [(i, j) for i in range(n) for j in range(n)
             if not (same and i == j)]
    np_ = len(pairs)
    D = np.zeros((n, n))
    C = np.zeros((n, n))
    print(f"── {tag}  ({np_} pairs, m={nb}) ──")

    t0 = time.time()
    for idx, (i, j) in enumerate(pairs):
        D[i, j] = compute_dsa_distance(ops1[i], ops2[j])
        if (idx + 1) % 50 == 0 or idx + 1 == np_:
            print(f"   DSA {idx+1}/{np_}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    for idx, (i, j) in enumerate(pairs):
        C[i, j], _ = compute_ccd_distance(ops1[i], ops2[j], b1, b2, nb)
        if (idx + 1) % 50 == 0 or idx + 1 == np_:
            print(f"   CCD {idx+1}/{np_}  ({time.time()-t0:.1f}s)", flush=True)

    return D, C


# ── within-basis runs ────────────────────────────────────────
for b in BASIS_LIST:
    d, c = run(ops[b], ops[b], b, b, f"within-{b}", N_BASIS)
    np.save(os.path.join(DIR, f"dsa_distances_{b}.npy"), d)
    np.save(os.path.join(DIR, f"ccd_distances_{b}.npy"), c)

# ── cross-basis run (Fourier ↔ Legendre), original systems only ──
d_x, c_x = run(ops["fourier"][0::2], ops["legendre"][0::2],
               "fourier", "legendre",
               "cross (FOURIER→LEGENDRE)", N_BASIS)
np.save(os.path.join(DIR, "dsa_distances_cross_fourier_legendre.npy"), d_x)
np.save(os.path.join(DIR, "ccd_distances_cross_fourier_legendre.npy"), c_x)

np.save(os.path.join(DIR, "k_values.npy"), K_VALUES)
print("\nDone.")
