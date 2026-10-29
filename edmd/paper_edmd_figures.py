import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# ══════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "Results")
ST_DIR = os.path.join(BASE, "StateTransformNoBasisChangeResults")
BC_DIR = os.path.join(BASE, "PureBasisChangeResults")
PLOT_DIR = os.path.join(BASE, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
#  Global plotting style
# ══════════════════════════════════════════════════════════

STYLE = {
    # Font sizes
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "annotation.fontsize": 18,
    "rowlabel.fontsize": 18,
    "colorbar.ticksize": 18,

    # Line / marker sizes
    "lines.linewidth": 3.6,
    "lines.markersize": 12,
    "scatter.size": 18,

    # Figure sizes
    "heatmap.figsize": (5.4, 4.8),
    "scatter.figsize": (4.8, 4.4),
    "curve.figsize": (5.4, 4.8),
    "rowwise.figsize": (19.0, 9.6),

    # Heatmap ticks
    "k_tick_show_every": 2,

    # Colorbars
    "colorbar.decimals": 1,
    "colorbar.nbins": 5,

    # Export
    "figure.dpi": 150,
    "savefig.dpi": 220,
}

plt.rcParams.update({
    "font.size": STYLE["font.size"],
    "axes.labelsize": STYLE["axes.labelsize"],
    "axes.titlesize": STYLE["axes.titlesize"],
    "xtick.labelsize": STYLE["xtick.labelsize"],
    "ytick.labelsize": STYLE["ytick.labelsize"],
    "legend.fontsize": STYLE["legend.fontsize"],
    "lines.linewidth": STYLE["lines.linewidth"],
    "lines.markersize": STYLE["lines.markersize"],
    "figure.dpi": STYLE["figure.dpi"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_DSA = "#1f77b4"
C_CCD = "#d62728"


# ══════════════════════════════════════════════════════════
#  Index helpers
# ══════════════════════════════════════════════════════════

def _idx_orig(nk):
    """
    main.py stores systems as:
      original(k_1), transformed(k_1), original(k_2), transformed(k_2), ...

    Original/no-bijection systems are even indices.
    """
    return np.arange(0, 2 * nk, 2)


def _idx_trans(nk):
    """
    State-transformed systems are odd indices.
    """
    return np.arange(1, 2 * nk, 2)


def _block(mat, ri, ci):
    return mat[np.ix_(ri, ci)]


def _same_block(mat, idx):
    return mat[np.ix_(idx, idx)]


def save_current(name):
    png = os.path.join(PLOT_DIR, f"{name}.png")
    pdf = os.path.join(PLOT_DIR, f"{name}.pdf")
    plt.savefig(png, dpi=STYLE["savefig.dpi"], bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {png}")
    print(f"Saved {pdf}")


# ══════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════

def fit_affine(D, B, mask=None):
    """
    Fit D ≈ a B + c.

    Returns:
      a, c, normalized affine RMSE, R^2
    """
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, bool).ravel()

    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return 0.0, 0.0, np.nan, np.nan

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

    nrmse = rmse / scale

    denom = max(np.sum((y - np.mean(y)) ** 2), 1e-15)
    r2 = 1.0 - np.sum(resid ** 2) / denom

    return float(a), float(c), float(nrmse), float(r2)


def matrix_pearson(D, B, mask=None):
    """
    Matrix-wise Pearson correlation between flattened D and B.
    """
    x = np.asarray(B, float).ravel()
    y = np.asarray(D, float).ravel()

    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, bool).ravel()

    x = x[valid]
    y = y[valid]

    if x.size == 0:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return float(np.sum(x * y) / den) if den > 1e-15 else np.nan


# ══════════════════════════════════════════════════════════
#  Plot helpers
# ══════════════════════════════════════════════════════════

def _format_colorbar(cbar):
    """
    Keep colorbar labels compact and non-scientific.
    """
    cbar.locator = MaxNLocator(nbins=STYLE["colorbar.nbins"])
    cbar.formatter = FormatStrFormatter(f"%.{STYLE['colorbar.decimals']}f")
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=STYLE["colorbar.ticksize"])


def _sparse_k_ticks(ax, kv):
    """
    Show only k = 2, 3, and 4 on both axes.

    Uses the nearest available grid indices, so it works even if kv is
    linspace(2, 4, 10), where 3.0 may not be exactly present.
    """
    desired = np.array([2.0, 3.0, 4.0])

    shown = []
    labels = []
    for val in desired:
        idx = int(np.argmin(np.abs(kv - val)))
        if idx not in shown:
            shown.append(idx)
            labels.append(f"{val:g}")

    ax.set_xticks(shown)
    ax.set_yticks(shown)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

def plot_heatmap(M, kv, xlabel, ylabel, out_name):
    """
    Save one heatmap as an independent figure.

    Uses linear color scale so:
      - zero/diagonal values are shown naturally,
      - colorbar labels stay readable,
      - no scientific notation appears.
    """
    M = np.array(M, dtype=float, copy=True)

    fig, ax = plt.subplots(1, 1, figsize=STYLE["heatmap.figsize"])

    vmax = np.nanmax(M)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        M,
        aspect="equal",
        origin="lower",
        cmap="gist_heat",
        vmin=0.0,
        vmax=vmax,
    )

    _sparse_k_ticks(ax, kv)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for i in range(1, len(kv)):
        ax.axhline(i - 0.5, color="w", lw=0.35, alpha=0.35)
        ax.axvline(i - 0.5, color="w", lw=0.35, alpha=0.35)

    ax.plot(
        np.arange(len(kv)),
        np.arange(len(kv)),
        "w--",
        lw=1.0,
        alpha=0.9,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _format_colorbar(cbar)

    plt.tight_layout()
    save_current(out_name)


def plot_scatter(D, B, color, out_name):
    """
    Save one scatter plot (all entries of D against B) with affine best-fit line.

    Annotation reports:
      - normalized affine RMSE
      - matrix-wise Pearson correlation
      - R^2 of affine fit
    """
    bv = np.asarray(B, float).ravel()
    dv = np.asarray(D, float).ravel()

    valid = np.isfinite(bv) & np.isfinite(dv)
    bv = bv[valid]
    dv = dv[valid]

    a, c, nrmse, r2 = fit_affine(D, B)
    pear = matrix_pearson(D, B)

    fig, ax = plt.subplots(1, 1, figsize=STYLE["scatter.figsize"])

    ax.scatter(
        bv,
        dv,
        s=STYLE["scatter.size"],
        alpha=0.7,
        color=color,
        edgecolors="none",
    )

    if bv.size:
        xx = np.linspace(bv.min(), bv.max(), 200)
        ax.plot(xx, a * xx + c, color=color)

    txt = (
        f"nRMSE = {nrmse:.3f}\n"
        f"Pearson = {pear:.3f}\n"
        f"$R^2$ = {r2:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STYLE["annotation.fontsize"],
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.7",
        ),
    )

    ax.set_xlabel("Reference distance")
    ax.set_ylabel("Comparison distance")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_current(out_name)


def plot_two_method_curve(
    x,
    y_dsa,
    y_ccd,
    xlabel,
    ylabel,
    out_name,
    ylim=None,
    ccd_label="CSA",
):
    fig, ax = plt.subplots(1, 1, figsize=STYLE["curve.figsize"])

    ax.plot(x, y_dsa, "o-", color=C_DSA, label="DSA")
    ax.plot(x, y_ccd, "s-", color=C_CCD, label=ccd_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    save_current(out_name)


def plot_recovered_theta(theta_vals, theta_mean, theta_std):
    fig, ax = plt.subplots(1, 1, figsize=STYLE["curve.figsize"])

    # theta_hat on the ten conjugate pairs (K_k^orig, K_k^theta): mean +- std over k
    ax.plot(theta_vals, theta_mean, "o-", color=C_CCD, label="mean $\\hat\\theta$")
    ax.fill_between(
        theta_vals,
        theta_mean - theta_std,
        theta_mean + theta_std,
        alpha=0.15,
        color=C_CCD,
        label="$\\pm 1$ std",
    )
    ax.plot(theta_vals, theta_vals, "k--", lw=1.2, label="true $\\theta$")

    ax.set_xlabel("True $\\theta$")
    ax.set_ylabel("Recovered $\\hat\\theta$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    save_current("state_transform_recovered_theta")


def plot_rowwise_8_heatmaps(kv, heatmaps):
    """
    Appendix figure: the eight within-transformation distance matrices
    D^M[(Phi, theta), (Phi, theta)].

      Rows:     DSA, CSA
      Columns:  Fourier (Original | Transformed), Legendre (Original | Transformed)

    Every panel has its own colorbar; the colour scale is shared within a row.
    Entry (i, j) = d(K_{k_i}, K_{k_j}): k_1 on the vertical axis, k_2 horizontal.
    """
    order = [
        ("dsa_fourier_orig", "dsa_fourier_trans", "dsa_legendre_orig", "dsa_legendre_trans"),
        ("ccd_fourier_orig", "ccd_fourier_trans", "ccd_legendre_orig", "ccd_legendre_trans"),
    ]
    row_labels = ["DSA", "CSA"]
    col_labels = ["Original", "Transformed", "Original", "Transformed"]

    fig, axes = plt.subplots(2, 4, figsize=STYLE["rowwise.figsize"])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.13, top=0.90,
                        wspace=0.45, hspace=0.35)

    for r, keys in enumerate(order):
        vmax = max(np.nanmax(heatmaps[key]) for key in keys)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        for c, key in enumerate(keys):
            ax = axes[r, c]
            im = ax.imshow(
                heatmaps[key],
                origin="lower",
                aspect="equal",
                cmap="gist_heat",
                vmin=0.0,
                vmax=vmax,
            )
            _sparse_k_ticks(ax, kv)

            for i in range(1, len(kv)):
                ax.axhline(i - 0.5, color="w", lw=0.35, alpha=0.35)
                ax.axvline(i - 0.5, color="w", lw=0.35, alpha=0.35)

            ax.plot(np.arange(len(kv)), np.arange(len(kv)), "w--", lw=1.0, alpha=0.9)

            ax.set_xlabel("$k_2$")
            ax.set_ylabel("$k_1$")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _format_colorbar(cbar)

    # Column titles above the top row
    for c, label in enumerate(col_labels):
        pos = axes[0, c].get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.02, label,
                 ha="center", va="bottom",
                 fontsize=STYLE["axes.labelsize"], fontweight="bold")

    # Row labels on the left
    for r, label in enumerate(row_labels):
        pos = axes[r, 0].get_position()
        fig.text(0.02, 0.5 * (pos.y0 + pos.y1), label, rotation=90,
                 ha="center", va="center",
                 fontsize=STYLE["rowlabel.fontsize"], fontweight="bold")

    # Dictionary group labels below the bottom row
    for c0, label in [(0, "Fourier"), (2, "Legendre")]:
        p0 = axes[1, c0].get_position()
        p1 = axes[1, c0 + 1].get_position()
        fig.text(0.5 * (p0.x0 + p1.x1), 0.015, label,
                 ha="center", va="bottom",
                 fontsize=STYLE["axes.labelsize"], fontweight="bold")

    save_current("within_basis_8_heatmaps_rowwise")
# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    kv = np.load(os.path.join(RES, "k_values.npy"))
    nk = len(kv)

    io = _idx_orig(nk)
    it = _idx_trans(nk)

    dsa_fixed = np.load(os.path.join(RES, "dsa_distances_fourier.npy"))
    ccd_fixed = np.load(os.path.join(RES, "ccd_distances_fourier.npy"))

    dsa_leg = np.load(os.path.join(RES, "dsa_distances_legendre.npy"))
    ccd_leg = np.load(os.path.join(RES, "ccd_distances_legendre.npy"))

    dsa_cross = np.load(os.path.join(RES, "dsa_distances_cross_fourier_legendre.npy"))
    ccd_cross = np.load(os.path.join(RES, "ccd_distances_cross_fourier_legendre.npy"))

    # ─────────────────────────────────────────────────────
    #  Fixed-basis state-transform heatmaps/scatters
    # ─────────────────────────────────────────────────────

    D_dsa_fixed = _block(dsa_fixed, io, it)
    D_ccd_fixed = _block(ccd_fixed, io, it)

    # Symmetrized reference (Appendix D):
    #   B = ( D^M[(Phi,0),(Phi,0)] + D^M[(Phi,theta),(Phi,theta)] ) / 2
    B_dsa_fixed = 0.5 * (_block(dsa_fixed, io, io) + _block(dsa_fixed, it, it))
    B_ccd_fixed = 0.5 * (_block(ccd_fixed, io, io) + _block(ccd_fixed, it, it))

    plot_heatmap(
        D_dsa_fixed,
        kv,
        xlabel="k (transformed)",
        ylabel="k (original)",
        out_name="fixed_basis_transform_heatmap_dsa",
    )

    plot_heatmap(
        D_ccd_fixed,
        kv,
        xlabel="k (transformed)",
        ylabel="k (original)",
        out_name="fixed_basis_transform_heatmap_ccd",
    )

    plot_scatter(
        D_dsa_fixed,
        B_dsa_fixed,
        C_DSA,
        out_name="fixed_basis_scatter_dsa",
    )

    plot_scatter(
        D_ccd_fixed,
        B_ccd_fixed,
        C_CCD,
        out_name="fixed_basis_scatter_ccd",
    )

    # ─────────────────────────────────────────────────────
    #  State-transform robustness curves
    # ─────────────────────────────────────────────────────

    theta_vals = np.load(os.path.join(ST_DIR, "theta_values.npy"))
    dsa_nrmse_st = np.load(os.path.join(ST_DIR, "dsa_nrmse.npy"))
    ccd_nrmse_st = np.load(os.path.join(ST_DIR, "ccd_nrmse.npy"))
    dsa_pearson_st = np.load(os.path.join(ST_DIR, "dsa_pearson.npy"))
    ccd_pearson_st = np.load(os.path.join(ST_DIR, "ccd_pearson.npy"))
    ccd_theta_mean = np.load(os.path.join(ST_DIR, "ccd_theta_mean.npy"))
    ccd_theta_std = np.load(os.path.join(ST_DIR, "ccd_theta_std.npy"))

    plot_two_method_curve(
        theta_vals,
        dsa_nrmse_st,
        ccd_nrmse_st,
        xlabel=" $\\theta$",
        ylabel="nRMSE",
        out_name="state_transform_nrmse",
    )

    plot_two_method_curve(
        theta_vals,
        dsa_pearson_st,
        ccd_pearson_st,
        xlabel=" $\\theta$",
        ylabel="Correlation",
        out_name="state_transform_pearson",
        ylim=(-0.05, 1.05),
    )

    plot_recovered_theta(theta_vals, ccd_theta_mean, ccd_theta_std)

    # ─────────────────────────────────────────────────────
    #  Cross-basis heatmaps/scatters
    # ─────────────────────────────────────────────────────

    # main.py stores the cross-basis matrices over the original systems only
    D_dsa_cross = dsa_cross
    D_ccd_cross = ccd_cross

    B_dsa_cross = 0.5 * (_block(dsa_fixed, io, io) + _block(dsa_leg, io, io))
    B_ccd_cross = 0.5 * (_block(ccd_fixed, io, io) + _block(ccd_leg, io, io))

    # D[i, j] = d(K_i^Fourier, K_j^Legendre); transpose for display so that
    # Fourier runs along the horizontal axis and Legendre along the vertical.
    plot_heatmap(
        D_dsa_cross.T,
        kv,
        xlabel="k (Fourier)",
        ylabel="k (Legendre)",
        out_name="cross_basis_heatmap_dsa",
    )

    plot_heatmap(
        D_ccd_cross.T,
        kv,
        xlabel="k (Fourier)",
        ylabel="k (Legendre)",
        out_name="cross_basis_heatmap_ccd",
    )

    plot_scatter(
        D_dsa_cross,
        B_dsa_cross,
        C_DSA,
        out_name="cross_basis_scatter_dsa",
    )

    plot_scatter(
        D_ccd_cross,
        B_ccd_cross,
        C_CCD,
        out_name="cross_basis_scatter_ccd",
    )

    # ─────────────────────────────────────────────────────
    #  Pure basis-change robustness curves
    # ─────────────────────────────────────────────────────

    basis_dist = np.load(os.path.join(BC_DIR, "basis_distance.npy"))
    dsa_nrmse_bc = np.load(os.path.join(BC_DIR, "dsa_nrmse.npy"))
    ccd_nrmse_bc = np.load(os.path.join(BC_DIR, "ccd_nrmse.npy"))
    dsa_pearson_bc = np.load(os.path.join(BC_DIR, "dsa_pearson.npy"))
    ccd_pearson_bc = np.load(os.path.join(BC_DIR, "ccd_pearson.npy"))

    order = np.argsort(basis_dist)
    x_bc = basis_dist[order]

    plot_two_method_curve(
        x_bc,
        dsa_nrmse_bc[order],
        ccd_nrmse_bc[order],
        xlabel="Basis distance",
        ylabel="nRMSE",
        out_name="basis_change_nrmse",
    )

    plot_two_method_curve(
        x_bc,
        dsa_pearson_bc[order],
        ccd_pearson_bc[order],
        xlabel="Basis distance",
        ylabel="Correlation",
        out_name="basis_change_pearson",
        ylim=(-0.05, 1.05),
    )

    # ─────────────────────────────────────────────────────
    #  Extra diagnostic: row-wise 8 heatmaps
    # ─────────────────────────────────────────────────────

    heatmaps = {
        "dsa_fourier_orig": _same_block(dsa_fixed, io),
        "dsa_fourier_trans": _same_block(dsa_fixed, it),
        "dsa_legendre_orig": _same_block(dsa_leg, io),
        "dsa_legendre_trans": _same_block(dsa_leg, it),
        "ccd_fourier_orig": _same_block(ccd_fixed, io),
        "ccd_fourier_trans": _same_block(ccd_fixed, it),
        "ccd_legendre_orig": _same_block(ccd_leg, io),
        "ccd_legendre_trans": _same_block(ccd_leg, it),
    }

    plot_rowwise_8_heatmaps(kv, heatmaps)

    print(f"\nAll plots saved independently to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
