import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(_HERE, "Results")
PLOTS_DIR = os.path.join(_HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
_RHO_FLOOR = 0.0


def _load():
    edmd = np.load(os.path.join(RES_DIR, "conv_grid_edmd.npz"),
                   allow_pickle=False)
    smooth_rho = np.load(os.path.join(RES_DIR, "conv_rank_smooth.npz"),
                         allow_pickle=False)
    return edmd, smooth_rho


def _panel_label(ax, label):
    ax.text(-0.02, 1.05, f"({label})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right")


# ─────────────────────────────────────────────────────────────────
#  Operator convergence: rho_x
# ─────────────────────────────────────────────────────────────────

def fig_residual(edmd, smooth_rho):
    K_e = edmd["K_GRID"]; nb_arr = edmd["N_BASIS"]
    rho_e = np.maximum(edmd["rho_f"], _RHO_FLOOR)   # (n_basis_types, n_nb, n_k)

    rh_ranks  = smooth_rho["ranks"]
    rh_rho    = np.maximum(smooth_rho["rho"], _RHO_FLOOR)   # (n_sys, n_r)
    rh_kinds  = smooth_rho["kinds"]
    rh_rates  = smooth_rho["rates"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    cmap = cm.viridis

    # (a) Fourier
    ax = axes[0]
    cs = cmap(np.linspace(0.15, 0.92, len(nb_arr)))
    for j, nb in enumerate(nb_arr):
        ax.plot(K_e, rho_e[0, j], color=cs[j], label=f"$n_b={int(nb)}$")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_x$  (Fourier-EDMD)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best", frameon=False)
    _panel_label(ax, "a")

    # (b) Legendre
    ax = axes[1]
    for j, nb in enumerate(nb_arr):
        ax.plot(K_e, rho_e[1, j], color=cs[j])
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\rho_x$  (Legendre-EDMD)")
    ax.grid(True, alpha=0.2)
    _panel_label(ax, "b")

    # (c) Hankel-DMD (smooth bijection): rho_x vs rank, one line per system
    ax = axes[2]
    cs2 = cmap(np.linspace(0.15, 0.85, 2))
    rate_idx = {0.5: 0, 0.8: 1}
    for si in range(rh_rho.shape[0]):
        kind = str(rh_kinds[si])
        rate = float(rh_rates[si])
        marker = "o" if kind == "lin" else "s"
        ls     = "-" if kind == "lin" else "--"
        label = (rf"$f_{{{rate:g}}}$" if kind == "lin"
                 else rf"$g^{{({rate:g})}}$")
        ax.plot(rh_ranks, rh_rho[si], marker=marker, linestyle=ls,
                color=cs2[rate_idx[rate]], label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 99])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "99"])
    ax.minorticks_off()
    ax.set_xlabel("SVD rank $r$  (Hankel-DMD)")
    ax.set_ylabel(r"$\rho_x$")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7, loc="best", frameon=False)
    _panel_label(ax, "c")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "appendix_residual.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    edmd, smooth_rho = _load()
    fig_residual(edmd, smooth_rho)


if __name__ == "__main__":
    main()
