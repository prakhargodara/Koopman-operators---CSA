import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(_HERE, "Results")
PLOT_DIR = os.path.join(_HERE, "plots")
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
    "legend.title_fontsize": 18,

    # Line / marker sizes
    "lines.linewidth": 3.6,
    "lines.markersize": 12,
    "scatter.size": 18,

     # Figure sizes
    "figsize": (5.4, 4.8),


    # Grid
    "grid.alpha": 0.25,
    # Heatmap ticks
    "k_tick_show_every": 2,

    # Colorbars
    "colorbar.decimals": 1,
    "colorbar.nbins": 5,

    # Export
    "figure.dpi": 150,
    "savefig.dpi": 220,
    # Legend
    "legend.frameon": False,
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


# ══════════════════════════════════════════════════════════
#  Colors
# ══════════════════════════════════════════════════════════

C_N = "#1b7837"
C_A = "#762a83"
C_DSA = "#1f77b4"
C_CCD = "#d62728"


# ══════════════════════════════════════════════════════════
#  Dynamics / bijection
# ══════════════════════════════════════════════════════════

def h_smooth(x, a, n):
    x = np.asarray(x, dtype=float)
    n = int(round(n))
    return x - (a / (2.0 * np.pi * n)) * np.sin(
        2.0 * np.pi * n * (x - 0.5)
    )


# ══════════════════════════════════════════════════════════
#  IO helpers
# ══════════════════════════════════════════════════════════

def save_current(name):
    png = os.path.join(PLOT_DIR, f"{name}.png")
    pdf = os.path.join(PLOT_DIR, f"{name}.pdf")

    plt.savefig(png, dpi=STYLE["savefig.dpi"], bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print(f"wrote {png}")
    print(f"wrote {pdf}")


def load_data():
    rank_d = np.load(os.path.join(DIR, "section3_smooth_rank_sweep.npz"))
    n_d = np.load(os.path.join(DIR, "section3_smooth_n_sweep.npz"))
    a_d = np.load(os.path.join(DIR, "section3_smooth_a_freq_sweep.npz"))

    # Rank sweep
    ranks_R = rank_d["ranks"].astype(float)
    a_ref = float(rank_d["a_ref"])
    n_ref = int(rank_d["n_ref"])
    dsa_R = (rank_d["dsa_conj"] / np.sqrt(rank_d["dsa_dim_conj"])).mean(axis=1)
    ccd_R = rank_d["ccd_conj"].mean(axis=1)

    # n sweep
    n_vals = n_d["n_values"].astype(float)
    ranks_N = n_d["ranks"]
    a_freq = float(n_d["a_freq"])
    ri_top = int(np.argmax(ranks_N))
    rank_top = int(ranks_N[ri_top])
    dsa_N = (n_d["dsa_conj"][ri_top]
             / np.sqrt(n_d["dsa_dim_conj"][ri_top])).mean(axis=1)
    ccd_N = n_d["ccd_conj"][ri_top].mean(axis=1)

    # a sweep
    a_vals = a_d["a_freqs"].astype(float)
    ranks_A = a_d["ranks"]
    n_for_a = int(a_d["n_ref"])
    ri_top_a = int(np.argmax(ranks_A))
    rank_top_a = int(ranks_A[ri_top_a])
    dsa_A = (a_d["dsa_conj"][ri_top_a]
             / np.sqrt(a_d["dsa_dim_conj"][ri_top_a])).mean(axis=1)
    ccd_A = a_d["ccd_conj"][ri_top_a].mean(axis=1)

    return {
        "rank": {
            "ranks": ranks_R,
            "dsa": dsa_R,
            "ccd": ccd_R,
            "a_ref": a_ref,
            "n_ref": n_ref,
        },
        "n_sweep": {
            "n_values": n_vals,
            "dsa": dsa_N,
            "ccd": ccd_N,
            "a_freq": a_freq,
            "rank": rank_top,
        },
        "a_sweep": {
            "a_values": a_vals,
            "dsa": dsa_A,
            "ccd": ccd_A,
            "n_ref": n_for_a,
            "rank": rank_top_a,
        },
    }


# ══════════════════════════════════════════════════════════
#  Plot 1: bijection deviation
# ══════════════════════════════════════════════════════════

def plot_bijection_deviation(data):
    a_freq = data["n_sweep"]["a_freq"]

    fig, ax = plt.subplots(1, 1, figsize=STYLE["figsize"])

    x_plot = np.linspace(0.005, 0.995, 4000)

    cmap = plt.cm.viridis
    n_demo = [1, 2, 4, 8]
    cs_demo = cmap(np.linspace(0.15, 0.85, len(n_demo)))

    for nn, c in zip(n_demo, cs_demo):
        ax.plot(
            x_plot,
            h_smooth(x_plot, a_freq, nn) - x_plot,
            color=c,
            label=fr"$n={nn}$",
        )

    ax.axhline(0, color="k", ls=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$h_{a,n}(x) - x$")
    ax.grid(True, alpha=STYLE["grid.alpha"])

    ax.legend(
        loc="best",
        frameon=STYLE["legend.frameon"],
        title=fr"$a={a_freq}$",
        title_fontsize=STYLE["legend.title_fontsize"],
    )

    plt.tight_layout()
    save_current("smooth_bijection_deviation")


# ══════════════════════════════════════════════════════════
#  Plot 2: n and a sweeps
# ══════════════════════════════════════════════════════════

def plot_n_and_a_sweeps(data):
    n_values = data["n_sweep"]["n_values"]
    dsa_N = data["n_sweep"]["dsa"]
    ccd_N = data["n_sweep"]["ccd"]
    a_freq = data["n_sweep"]["a_freq"]
    rank_N = data["n_sweep"]["rank"]

    a_values = data["a_sweep"]["a_values"]
    dsa_A = data["a_sweep"]["dsa"]
    ccd_A = data["a_sweep"]["ccd"]
    n_ref = data["a_sweep"]["n_ref"]
    rank_A = data["a_sweep"]["rank"]

    fig, ax = plt.subplots(1, 1, figsize=STYLE["figsize"])

    # n sweep on bottom axis
    ax.plot(n_values, dsa_N, "o-", color=C_N)
    ax.plot(n_values, ccd_N, "s--", color=C_N)

    ax.set_xlabel(rf"$n$  ", color=C_N) #(rank = {rank_N}, $a={a_freq}$)
    ax.set_ylabel("Conjugate distance")
    ax.tick_params(axis="x", labelcolor=C_N)
    ax.spines["bottom"].set_color(C_N)
    ax.grid(True, alpha=STYLE["grid.alpha"])

    # a sweep on top axis
    ax_top = ax.twiny()
    ax_top.plot(a_values, dsa_A, "^-", color=C_A)
    ax_top.plot(a_values, ccd_A, "D--", color=C_A)

    ax_top.set_xlabel(rf"$a$  ", color=C_A) #(rank = {rank_A}, $n={n_ref}$)
    ax_top.tick_params(axis="x", labelcolor=C_A)
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["top"].set_color(C_A)
    ax_top.spines["right"].set_visible(False)

    # Legend shows line style only: solid = DSA, dashed = CCD.
    proxy_dsa = Line2D(
        [0],
        [0],
        color="0.25",
        ls="-",
        marker=None,
    )
    proxy_ccd = Line2D(
        [0],
        [0],
        color="0.25",
        ls="--",
        marker=None,
    )

    ax.legend(
        [proxy_dsa, proxy_ccd],
        [r"DSA / $\sqrt{r'}$", "CSA"],
        loc="best",
        frameon=STYLE["legend.frameon"],
    )

    plt.tight_layout()
    save_current("smooth_bijection_n_and_a_sweeps")

# ══════════════════════════════════════════════════════════
#  Plot 3: rank sweep
# ══════════════════════════════════════════════════════════

def plot_rank_sweep(data):
    ranks = data["rank"]["ranks"]
    dsa = data["rank"]["dsa"]
    ccd = data["rank"]["ccd"]
    a_ref = data["rank"]["a_ref"]
    n_ref = data["rank"]["n_ref"]

    fig, ax = plt.subplots(1, 1, figsize=STYLE["figsize"])

    ax.plot(ranks, dsa, "o-", color=C_DSA, label=r"DSA / $\sqrt{r'}$")
    ax.plot(ranks, ccd, "s--", color=C_CCD, label="CSA")

    ax.set_xlabel(rf"SVD rank $r$  ") #($a={a_ref}$, $n={n_ref}$)
    ax.set_ylabel("Conjugate distance")
    ax.set_yscale("log")
    ax.grid(True, alpha=STYLE["grid.alpha"], which="both")
    ax.legend(loc="best", frameon=STYLE["legend.frameon"])

    plt.tight_layout()
    save_current("smooth_bijection_rank_sweep")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    data = load_data()

    plot_bijection_deviation(data)
    plot_n_and_a_sweeps(data)
    plot_rank_sweep(data)

    print(f"\nAll plots saved independently to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
