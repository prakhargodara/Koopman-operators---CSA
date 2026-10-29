import os
import time
import numpy as np
from scipy.linalg import svd, eigh

from dsa import compute_dsa_distance


# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

def logistic(x, k):
    return k * x * (1.0 - x)


# -----------------------------------------------------------------------------
# Raw endpoint families
# -----------------------------------------------------------------------------

def raw_fourier_basis(x, n_harmonics):
    x = np.asarray(x)
    cols = [np.ones_like(x)]
    for n in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * n * x))
        cols.append(np.sin(2.0 * np.pi * n * x))
    return np.column_stack(cols)


def raw_legendre_basis(x, max_degree):
    x = np.asarray(x)
    t = 2.0 * x - 1.0
    N = len(x)
    P = np.empty((N, max_degree + 1))
    P[:, 0] = 1.0
    if max_degree >= 1:
        P[:, 1] = t
    for n in range(2, max_degree + 1):
        P[:, n] = ((2 * n - 1) * t * P[:, n - 1] - (n - 1) * P[:, n - 2]) / n
    return P[:, :max_degree]


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------

def sym_inv_sqrt(M, reg=1e-10):
    w, V = eigh(0.5 * (M + M.T))
    w = np.maximum(w, reg)
    return (V / np.sqrt(w)) @ V.T


def sym_sqrt(M, reg=1e-10):
    w, V = eigh(0.5 * (M + M.T))
    w = np.maximum(w, reg)
    return (V * np.sqrt(w)) @ V.T


def gram_on_grid(Phi):
    return (Phi.T @ Phi) / Phi.shape[0]


def orthonormalize_coeffs(C, G, reg=1e-10):
    M = C.T @ G @ C
    return C @ sym_inv_sqrt(M, reg=reg)


def principal_geodesic(U, V, t, tol=1e-10):
    A = U.T @ V
    Pu, s, Qvt = svd(A, full_matrices=False)
    Qv = Qvt.T
    Up = U @ Pu
    Vp = V @ Qv
    angles = np.arccos(np.clip(s, -1.0, 1.0))

    W = np.zeros_like(Up)
    for j, ang in enumerate(angles):
        if ang < tol:
            W[:, j] = 0.0
        else:
            W[:, j] = (Vp[:, j] - np.cos(ang) * Up[:, j]) / np.sin(ang)

    Ut = Up * np.cos(t * angles)[None, :] + W * np.sin(t * angles)[None, :]
    Q, _ = np.linalg.qr(Ut)
    return Q[:, :U.shape[1]]


def projector_distance(U1, U2):
    m = U1.shape[1]
    P1 = U1 @ U1.T
    P2 = U2 @ U2.T
    return np.linalg.norm(P1 - P2, "fro") / np.sqrt(2.0 * m)


# -----------------------------------------------------------------------------
# Affine / correlation metrics
# -----------------------------------------------------------------------------

def fit_affine_distortion(D, B):
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    X = np.column_stack([x, np.ones_like(x)])
    a, c = np.linalg.lstsq(X, y, rcond=None)[0]

    if a < 0:
        a = 0.0
        c = float(np.mean(y))

    resid = y - (a * x + c)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    scale = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    if scale < 1e-15:
        scale = max(float(np.mean(np.abs(y))), 1.0)

    return rmse / scale


def matrix_pearson(D, B):
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return float(np.sum(x * y) / den) if den > 1e-15 else np.nan


# -----------------------------------------------------------------------------
# Basis family on Grassmannian
# -----------------------------------------------------------------------------

class FourierLegendreFamily:
    def __init__(self, target_dim=21, ortho_grid=None):
        if target_dim < 3 or target_dim % 2 == 0:
            raise ValueError("target_dim must be odd and >= 3")

        self.target_dim = target_dim
        self.n_harmonics = (target_dim - 1) // 2
        self.max_degree = target_dim

        def ambient_eval(x):
            F = raw_fourier_basis(x, self.n_harmonics)
            L = raw_legendre_basis(x, self.max_degree)
            return np.concatenate([F, L], axis=1)

        self.ambient_eval = ambient_eval

        A = self.ambient_eval(ortho_grid)
        self.G = gram_on_grid(A)
        self.S = sym_sqrt(self.G)
        self.Sinv = np.linalg.inv(self.S)

        p = 2 * target_dim
        CX = np.eye(p)[:, :target_dim]
        CY = np.eye(p)[:, target_dim:]

        self.CX = orthonormalize_coeffs(CX, self.G)
        self.CY = orthonormalize_coeffs(CY, self.G)
        self.UX = self.S @ self.CX
        self.UY = self.S @ self.CY

        eigvals = np.linalg.eigvalsh(self.G)
        n_eff = np.sum(eigvals > 1e-8 * eigvals.max())
        Auv = self.UX.T @ self.UY
        _, s, _ = svd(Auv, full_matrices=False)
        angles = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))

        print(f"Ambient dim={p}, effective rank={n_eff}")
        print(f"Max principal angle between endpoints = {angles.max():.2f} degrees")

    def coeffs(self, theta):
        U = principal_geodesic(self.UX, self.UY, theta)
        C = self.Sinv @ U
        return orthonormalize_coeffs(C, self.G)

    def evaluate(self, x, theta):
        return self.ambient_eval(x) @ self.coeffs(theta)

    def euclidean_basis(self, theta):
        return self.S @ self.coeffs(theta)


# -----------------------------------------------------------------------------
# EDMD
# -----------------------------------------------------------------------------

def compute_edmd_generic(x_data, y_data, basis_eval, reg=1e-8):
    Phi_X = basis_eval(x_data)
    Phi_Y = basis_eval(y_data)
    m = Phi_X.shape[1]
    G = Phi_X.T @ Phi_X + reg * np.eye(m)
    KT = np.linalg.solve(G, Phi_X.T @ Phi_Y)
    return KT.T


# -----------------------------------------------------------------------------
# Basis-transport CCD
# -----------------------------------------------------------------------------

def compute_basis_ccd_distance(K1, K2, basis_eval1, basis_eval2,
                               n_grid=800, reg=1e-8):
    x_grid = np.linspace(0.005, 0.995, n_grid)
    Phi1 = basis_eval1(x_grid)
    Phi2 = basis_eval2(x_grid)

    G = Phi1.T @ Phi1 + reg * np.eye(Phi1.shape[1])
    P = np.linalg.solve(G, Phi1.T @ Phi2).T

    Pn = np.linalg.norm(P, "fro")
    if Pn < 1e-12:
        return 1e12

    diff = P @ K1 - K2 @ P
    return float(np.linalg.norm(diff, "fro") / Pn)


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

def run_experiment(
    target_dim=21,
    theta_values=np.linspace(0.0, 1.0, 11),
    k_values=np.linspace(2.0, 4.0, 10),
    theta_ref=0.0,
    n_data=5000,
    reg=1e-8,
    out_dir="PureBasisChangeResults",
):
    os.makedirs(out_dir, exist_ok=True)

    ortho_grid = np.linspace(0.0, 1.0, 4000)
    x_data = np.linspace(1e-6, 1.0 - 1e-6, n_data)

    print(f"Building family with target_dim={target_dim} ...")
    family = FourierLegendreFamily(target_dim=target_dim, ortho_grid=ortho_grid)

    basis_evals = {
        th: (lambda x, th=th: family.evaluate(x, th))
        for th in theta_values
    }

    ops = {th: [] for th in theta_values}

    t0 = time.time()
    for th in theta_values:
        be = basis_evals[th]
        for k in k_values:
            y = np.clip(logistic(x_data, k), 1e-10, 1 - 1e-10)
            K = compute_edmd_generic(x_data, y, be, reg=reg)
            ops[th].append(K)
    print(f"Built all operators in {time.time() - t0:.1f}s")

    nK = len(k_values)
    be_ref = basis_evals[theta_ref]

    print("Computing reference matrices ...")
    D_ref_dsa = np.zeros((nK, nK))
    D_ref_ccd = np.zeros((nK, nK))

    for i in range(nK):
        for j in range(nK):
            K1 = ops[theta_ref][i]
            K2 = ops[theta_ref][j]
            D_ref_dsa[i, j] = compute_dsa_distance(K1, K2)
            D_ref_ccd[i, j] = compute_basis_ccd_distance(K1, K2, be_ref, be_ref)

    basis_dist = []
    dsa_nrmse = []
    ccd_nrmse = []
    dsa_pearson = []
    ccd_pearson = []

    for th in theta_values:
        print(f"theta={th:.3f} vs theta_ref={theta_ref:.3f}")

        be_th = basis_evals[th]
        D_dsa = np.zeros((nK, nK))
        D_ccd = np.zeros((nK, nK))

        for i in range(nK):
            for j in range(nK):
                K1 = ops[theta_ref][i]
                K2 = ops[th][j]
                D_dsa[i, j] = compute_dsa_distance(K1, K2)
                D_ccd[i, j] = compute_basis_ccd_distance(K1, K2, be_ref, be_th)

        nr_dsa = fit_affine_distortion(D_dsa, D_ref_dsa)
        nr_ccd = fit_affine_distortion(D_ccd, D_ref_ccd)

        pear_dsa = matrix_pearson(D_dsa, D_ref_dsa)
        pear_ccd = matrix_pearson(D_ccd, D_ref_ccd)

        U_ref = family.euclidean_basis(theta_ref)
        U_th = family.euclidean_basis(th)
        dist = projector_distance(U_ref, U_th)

        basis_dist.append(dist)
        dsa_nrmse.append(nr_dsa)
        ccd_nrmse.append(nr_ccd)
        dsa_pearson.append(pear_dsa)
        ccd_pearson.append(pear_ccd)

        print(
            f"  basis_dist={dist:.4f} | "
            f"DSA nRMSE={nr_dsa:.4f}, Pearson={pear_dsa:.4f} | "
            f"CCD nRMSE={nr_ccd:.4f}, Pearson={pear_ccd:.4f}"
        )

    basis_dist = np.array(basis_dist)
    dsa_nrmse = np.array(dsa_nrmse)
    ccd_nrmse = np.array(ccd_nrmse)
    dsa_pearson = np.array(dsa_pearson)
    ccd_pearson = np.array(ccd_pearson)

    np.save(os.path.join(out_dir, "basis_distance.npy"), basis_dist)
    np.save(os.path.join(out_dir, "dsa_nrmse.npy"), dsa_nrmse)
    np.save(os.path.join(out_dir, "ccd_nrmse.npy"), ccd_nrmse)
    np.save(os.path.join(out_dir, "dsa_pearson.npy"), dsa_pearson)
    np.save(os.path.join(out_dir, "ccd_pearson.npy"), ccd_pearson)

    print(f"\nSaved outputs to: {out_dir}/")


if __name__ == "__main__":
    run_experiment(
        target_dim=25,
        theta_values=np.linspace(0.0, 1.0, 11),
        k_values=np.linspace(2.0, 4.0, 10),
        theta_ref=0.0,
        n_data=5000,
        reg=1e-8,
        out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "PureBasisChangeResults"),
    )
