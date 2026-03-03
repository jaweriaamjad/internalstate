"""
Figure M.2: Prior-induced gap vs assumed prior (supplementary).
Single panel showing gap vs phat_r for different sigma values.
Pure analytical computation — no data files needed.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
import seaborn as sns

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
import config_utils

# Contrast values
cs = [0.0, 0.0625, 0.125, 0.25, 1.0]


def calc_fs(sigx, xis):
    """Compute f(xi, sigma) = (1/N_c) * sum_i exp(-(xi - c_i)^2 / (2*sigma^2))"""
    fs = np.zeros(np.shape(xis))
    for c in cs:
        fs += (1 / len(cs)) * np.exp(-np.multiply(xis - c, xis - c) / (2 * sigx * sigx))
    return fs


def calc_pright_rewarded(pr, sigx, xis):
    """Compute P(right|rewarded) for given prior and noise."""
    pl = 1.0 - pr
    fs = calc_fs(sigx, xis)
    fs_neg = calc_fs(sigx, -xis)
    pright_rewarded = np.divide(pr * fs, pr * fs + pl * fs_neg)
    return pright_rewarded


def calc_xi_star(pr, sigx):
    """Find xi* where P(right|rewarded) = 0.5."""
    dxi = 0.0003
    xis = np.arange(-3, 3 + 0.5 * dxi, dxi)
    pright_rewarded = calc_pright_rewarded(pr, sigx, xis)
    xi_idx = np.argmin(np.abs(pright_rewarded - 0.5))
    return xis[xi_idx]


def Phi(x):
    """Standard normal CDF."""
    return scist.norm.cdf(x)


def calc_gap(xi_starR, xi_starL, sigx):
    """Compute prior-induced gap."""
    return Phi(-xi_starR / sigx) - Phi(-xi_starL / sigx)


def main():
    import os

    paths = config_utils.get_paths()
    figure_dir = str(paths["figures_dir"])
    os.makedirs(figure_dir, exist_ok=True)

    colors = ["#000000", "#404040", "#555555", "#7F7F7F", "#A9A9A9", "#D3D3D3"]
    markers = ["o", "s", "^", "v", "D", "<"]

    sigxs = 1.0 / (np.sqrt(2 * np.pi) * np.arange(1, 7, 1))
    prs = np.arange(0.5, 0.996, 0.005)

    plt.style.use("default")
    sns.set_style("whitegrid", {"axes.grid": False})

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    for sidx, sigx in enumerate(sigxs):
        gaps = np.zeros(np.shape(prs))
        for pridx, pr in enumerate(prs):
            xi_starR = calc_xi_star(pr, sigx)
            xi_starL = calc_xi_star(1.0 - pr, sigx)
            gaps[pridx] = calc_gap(xi_starR, xi_starL, sigx)
        sigma_val = sidx + 1
        ax.plot(
            prs,
            gaps,
            color=colors[sidx],
            marker=markers[sidx],
            markersize=3,
            markevery=5,
            linewidth=2.5,
            label=f"$1/({sigma_val}\\sqrt{{2\\pi}})$",
        )
    ax.set_xlabel(r"$\hat{p}_r$", fontsize=18, fontweight="normal", fontfamily="Arial")
    ax.set_ylabel("Gap", fontsize=16, fontweight="normal", fontfamily="Arial")
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5, length=6)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    out_path = os.path.join(figure_dir, "figM2.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
