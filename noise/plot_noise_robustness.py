import os



import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "Results", "noise_robustness.npz")
PLOT_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def fit_affine_nrmse(D, B):
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()
    a, c = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    if a < 0:
        a, c = 0.0, float(np.mean(y))
    return np.sqrt(np.mean((y - a * x - c) ** 2)) / np.std(y)


def matrix_pearson(D, B):
    return float(np.corrcoef(np.ravel(B), np.ravel(D))[0, 1])


z = np.load(RES)
N_LIST = [int(n) for n in z["N_list"]]
SIG_LIST = [float(s) for s in z["sigma_list"]]
THETA_TRUE = float(z["theta_true"])

# stats[N] -> array (n_sigma, n_replicates, 5): d0, theta err, nRMSE, Pearson, twin
stats = {}
for N in N_LIST:
    cells = []
    for sig in SIG_LIST:
        D, TH, OO, TT = (z[f"{name}_N{N}_s{sig}"] for name in ("D", "TH", "OO", "TT"))
        rows = []
        for r in range(D.shape[0]):
            B = 0.5 * (OO[r] + TT[r])
            rows.append([
                np.diag(D[r]).mean(),
                np.abs(np.diag(TH[r]) - THETA_TRUE).mean(),
                fit_affine_nrmse(D[r], B),
                matrix_pearson(D[r], B),
                sum(int(np.argmin(D[r][i]) == i) for i in range(D.shape[1])),
            ])
        cells.append(rows)
    stats[N] = np.array(cells)

print("| N | sigma | d0 | |theta_hat - theta*| | nRMSE | Pearson | twin ID |")

for N in N_LIST:
    for i_s, sig in enumerate(SIG_LIST):
        r = stats[N][i_s]
        m, s = r.mean(0), r.std(0)
        print(f"| {N:g} | {sig:g} | {m[0]:.3f} ± {s[0]:.3f} | {m[1]:.4f} ± {s[1]:.4f} "
              f"| {m[2]:.3f} ± {s[2]:.3f} | {m[3]:.3f} ± {s[3]:.3f} "
              f"| {int(r[:, 4].sum())}/{r.shape[0] * len(z['k_values'])} |")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
panels = [
    (1, r"(a) parameter error $|\hat\theta - \theta^*|$"),
    (3, r"(b) Pearson$(D, B)$"),
    (0, r"(c) conjugate-pair score $d_0$"),
]
for ax, (col, title) in zip(axes, panels):
    for N in N_LIST:
        vals = stats[N][:, :, col]
        ax.errorbar(SIG_LIST, vals.mean(1), yerr=vals.std(1), fmt="o-", ms=3.5,
                    lw=1.3, capsize=2, label=rf"$N = 10^{{{int(np.log10(N))}}}$")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"noise level $\sigma$")
    ax.set_xticks(SIG_LIST)

axes[0].set_yscale("log")
axes[0].axhline(THETA_TRUE, color="gray", ls="--", lw=0.8)
axes[0].text(SIG_LIST[-1], THETA_TRUE * 1.15, rf"$\theta^* = {THETA_TRUE:g}$",
             color="gray", fontsize=8, ha="right", va="bottom")
axes[0].set_ylim(2e-4, 0.4)
axes[0].legend(fontsize=8, frameon=False, loc="center left")
axes[1].set_ylim(0.85, 1.0)
axes[2].set_ylim(0.0, 0.25)

fig.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(PLOT_DIR, f"appendix_noise_robustness.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
plt.close(fig)
