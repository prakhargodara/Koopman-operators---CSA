import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "Results", "theta_landscape.npz")
PLOT_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

z = np.load(RES)
kv = z["k_values"]
th_grid = z["theta_grid"]
ts_grid = z["theta_star_grid"]
caps = z["caps"]
ts_ref = float(z["theta_star_ref"])
J_ref = z["J_ref"]          # (n_k, n_theta)
DMIN = z["DMIN"]            # (n_caps, n_k, n_theta_star)
full_cap = caps.max()

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 3.5))

# (a) score landscape at theta* = 0.10
for ik, k in enumerate(kv):
    ax_a.plot(th_grid, J_ref[ik], color=plt.cm.viridis(ik / (len(kv) - 1)),
              lw=1.3, label=f"$k = {k:.2f}$")
ax_a.axvline(ts_ref, color="r", ls="--", lw=1.5)
ax_a.text(ts_ref + 0.01, 0.97, r"$\theta^*$", color="r", fontsize=12,
          transform=ax_a.get_xaxis_transform(), ha="left", va="top")
ax_a.set_xlabel(r"candidate parameter $\theta$")
ax_a.set_ylabel(r"CSA objective $J(h_\theta)$")
ax_a.set_title(rf"(a) score landscape, $\theta^* = {ts_ref:.2f}$", fontsize=11)
ax_a.legend(fontsize=6.5, ncol=2, loc="lower left", frameon=False)

# (b) misspecified search domain
for ic, cap in enumerate(caps):
    col = plt.cm.plasma(ic / (len(caps) - 1) * 0.85)
    mean = DMIN[ic].mean(axis=0)
    lab = (rf"cap $|\theta| \leq {cap:.2f}$" if cap < full_cap else "full domain")
    ax_b.plot(ts_grid, mean, "o-", color=col, ms=3, lw=1.5, label=lab)
    ax_b.fill_between(ts_grid, DMIN[ic].min(axis=0), DMIN[ic].max(axis=0),
                      color=col, alpha=0.15, lw=0)
    if cap < full_cap:
        ax_b.axvline(cap, color=col, ls=":", lw=1)
ax_b.set_xlabel(r"true parameter $\theta^*$")
ax_b.set_ylabel(r"$d_{\rm CSA} = \min_{|\theta| \leq c} J(h_\theta)$")
ax_b.set_title("(b) misspecified search domain", fontsize=11)
ax_b.legend(fontsize=7, loc="upper left", frameon=False)

fig.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(PLOT_DIR, f"appendix_theta_landscape.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
plt.close(fig)
