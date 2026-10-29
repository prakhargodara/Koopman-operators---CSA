import numpy as np
from scipy.linalg import schur, svd as _scipy_svd


# ─────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────

def _polar(X):
    try:
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        U, _, Vt = _scipy_svd(X, full_matrices=False, lapack_driver='gesvd')
    return U @ Vt


def _obj_grad(Q, K1, K2):
    R = K1 - Q @ K2 @ Q.T
    f = 0.5 * np.sum(R ** 2)
    G = -(R @ Q @ K2.T + R.T @ Q @ K2)       
    sym = 0.5 * (Q.T @ G + G.T @ Q)
    return f, G - Q @ sym                       


def _spectral_init(K1, K2):
    """Align real Schur bases of K1 and K2."""
    _, U1 = schur(K1, output='real')
    _, U2 = schur(K2, output='real')
    return _polar(U1 @ U2.T)


# ─────────────────────────────────────────────────────────
#  Optimiser  (BB step + backtracking)
# ─────────────────────────────────────────────────────────

def _optimize(K1, K2, Q0, maxiter=600, tol=1e-11):
    Q = Q0.copy()
    f, Grad = _obj_grad(Q, K1, K2)
    eta = 0.02

    for _ in range(maxiter):
        if np.linalg.norm(Grad, 'fro') < tol:
            break

        # Backtracking
        eta_try = eta
        accepted = False
        for _ in range(20):
            Q_new = _polar(Q - eta_try * Grad)
            f_new, G_new = _obj_grad(Q_new, K1, K2)
            if f_new <= f:
                accepted = True
                break
            eta_try *= 0.5

        if not accepted:
            break

        # Update for next step
        dQ = Q_new - Q
        dG = G_new - Grad
        dQdG = np.sum(dQ * dG)
        if abs(dQdG) > 1e-30:
            eta = np.clip(np.sum(dQ ** 2) / abs(dQdG), 1e-6, 0.5)

        Q, f, Grad = Q_new, f_new, G_new

    f, _ = _obj_grad(Q, K1, K2)
    return np.sqrt(max(2.0 * f, 0.0))


def compute_dsa_distance(K1, K2, n_restarts=3, seed=0):
    m = K1.shape[0]
    rng = np.random.default_rng(seed)

    F = np.eye(m)
    F[0, 0] = -1.0

    # Deterministic inits: identity, flip, spectral, spectral-flip
    Q_spec = _spectral_init(K1, K2)
    inits = [np.eye(m), F, Q_spec, F @ Q_spec]

    # Random inits in both components
    for _ in range(n_restarts):
        A = rng.normal(size=(m, m))
        Q, _ = np.linalg.qr(A)
        inits.append(Q)
        inits.append(F @ Q)

    return float(min(_optimize(K1, K2, Q0) for Q0 in inits))
