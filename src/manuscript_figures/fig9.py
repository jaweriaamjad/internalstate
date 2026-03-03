"""
Figure 9: Fraction correct versus assumed prior (Bayesian model performance).
Plots theoretical performance curves for different noise levels (sigma).
No data files needed — pure analytical computation.
"""
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
import config_utils

# Contrast values used in the model
cs = [0.0, 0.0625, 0.125, 0.25, 1.0]

# True prior (right block)
pr_true = 0.8


def calc_fs(sigx, xis):
    """Compute f(xi, sigma) = (1/N_c) * sum_i exp(-(xi - c_i)^2 / (2*sigma^2))"""
    fs = np.zeros(np.shape(xis))
    for c in cs:
        fs += (1 / len(cs)) * np.exp(-np.multiply(xis - c, xis - c) / (2 * sigx * sigx))
    return fs


def calc_pright_choice(pr, sigx, xis, atype):
    """Compute probability of choosing right given prior, noise, and action type."""
    pl = 1.0 - pr
    fs = calc_fs(sigx, xis)
    fs_neg = calc_fs(sigx, -xis)
    pright_rewarded = np.divide(pr * fs, pr * fs + pl * fs_neg)
    if atype == "matching":
        return pright_rewarded
    elif atype == "argmax":
        return np.heaviside(pright_rewarded - 0.5, 0.5)


def calc_pcorrect(pr, sigx, atype):
    """Compute fraction correct for given assumed prior and noise."""
    dxi = 0.001
    xis = np.arange(-3, 3 + 0.5 * dxi, dxi)
    pcorrect = 0.0
    for c in cs:
        pright_choice = calc_pright_choice(pr, sigx, xis, atype)
        pxi_c = np.exp(-np.multiply(xis - c, xis - c) / (2 * sigx * sigx)) / sqrt(2 * np.pi * sigx * sigx)
        pxi_cneg = np.exp(-np.multiply(xis + c, xis + c) / (2 * sigx * sigx)) / sqrt(2 * np.pi * sigx * sigx)
        pcorrect += pr_true * np.dot(pright_choice, pxi_c) * dxi / len(cs)
        pcorrect += (1 - pr_true) * np.dot(1 - pright_choice, pxi_cneg) * dxi / len(cs)
    return pcorrect


def main():
    paths = config_utils.get_paths()
    figure_dir = str(paths["figures_dir"])

    import os
    os.makedirs(figure_dir, exist_ok=True)

    colors = ["#000000", "#404040", "#555555", "#7F7F7F", "#A9A9A9", "#D3D3D3"]
    markers = ["o", "s", "^", "v", "D", "<"]

    sigxs = 1.0 / (np.sqrt(2 * np.pi) * np.arange(1, 7, 1))
    prs = np.arange(0.5, 0.996, 0.005)
    prlen = len(prs)
    pcorrect = np.zeros(prlen)

    plt.style.use("default")
    sns.set_style("whitegrid", {"axes.grid": False})

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

    atype = "argmax"
    for sidx, sigx in enumerate(sigxs):
        for pridx, pr in enumerate(prs):
            pcorrect[pridx] = calc_pcorrect(pr, sigx, atype)
        sigma_val = sidx + 1
        ax.plot(
            prs,
            pcorrect,
            color=colors[sidx],
            marker=markers[sidx],
            markersize=3,
            markevery=5,
            linewidth=2.5,
            label=f"$1/({sigma_val}\\sqrt{{2\\pi}})$",
        )

    ax.set_xlabel(r"$\hat{p}_r$", fontsize=18, fontweight="normal", fontfamily="Arial")
    ax.set_ylabel("Fraction of Correct Choices", fontsize=16, fontweight="normal", fontfamily="Arial")
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5, length=6)
    ax.tick_params(axis="both", which="minor", width=1, length=3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    out_path = os.path.join(figure_dir, "fig9.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
