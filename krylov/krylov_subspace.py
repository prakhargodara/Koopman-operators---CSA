
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class SubspaceInfo:
    V_r:        np.ndarray
    S:          np.ndarray
    rank:       int
    delay_dim:  int
    energy:     float
    Vt_full:    np.ndarray
    H_mean:     Optional[np.ndarray] = None
    centered:   bool = False


# ─────────────────────────────────────────────────────────
#  Delay matrix construction
# ─────────────────────────────────────────────────────────

def build_delay_matrix(ts, delay_dim):
    T = len(ts)
    N = T - delay_dim + 1
    H = np.empty((N, delay_dim))
    for j in range(delay_dim):
        H[:, j] = ts[j : j + N]
    return H


def build_delay_matrix_multi(trajectories, delay_dim):
    blocks = []
    for ts in trajectories:
        if len(ts) >= delay_dim:
            blocks.append(build_delay_matrix(ts, delay_dim))
    return np.vstack(blocks)


# ─────────────────────────────────────────────────────────
#  Krylov observable evaluation from known dynamics
# ─────────────────────────────────────────────────────────

def build_krylov_observables(x, map_fn, param, delay_dim):

    x = np.asarray(x, dtype=float)
    N = len(x)
    H = np.empty((N, delay_dim))
    H[:, 0] = x
    for j in range(1, delay_dim):
        H[:, j] = map_fn(H[:, j - 1], param)
    return H


# ─────────────────────────────────────────────────────────
#  SVD truncation with subspace tracking
# ─────────────────────────────────────────────────────────

def svd_truncate(H, rank=None, energy_threshold=0.999, center=False):
    H_work = H.copy()
    H_mean = None
    if center:
        H_mean = H_work.mean(axis=0)
        H_work -= H_mean

    U, S, Vt = np.linalg.svd(H_work, full_matrices=False)
    total_energy = np.sum(S ** 2)

    if rank is None:
        cum_energy = np.cumsum(S ** 2) / total_energy
        rank = int(np.searchsorted(cum_energy, energy_threshold)) + 1
        rank = min(rank, len(S))

    rank = min(rank, len(S))
    V_r = Vt[:rank, :].T   # (delay_dim, rank)
    captured = np.sum(S[:rank] ** 2) / total_energy

    return SubspaceInfo(
        V_r=V_r,
        S=S,
        rank=rank,
        delay_dim=H.shape[1],
        energy=captured,
        Vt_full=Vt,
        H_mean=H_mean,
        centered=center,
    )


# ─────────────────────────────────────────────────────────
#  Subspace comparison
# ─────────────────────────────────────────────────────────

def principal_angles(V1, V2):
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    _, sv, _ = np.linalg.svd(Q1.T @ Q2, full_matrices=False)
    # Clamp for numerical safety
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


def grassmann_distance(V1, V2):

    angles = principal_angles(V1, V2)
    return np.sqrt(np.sum(np.sin(angles) ** 2))


def project_to_subspace(H, info):
    if info.centered and info.H_mean is not None:
        H = H - info.H_mean
    return H @ info.V_r
