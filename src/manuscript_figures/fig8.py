"""
Figure 8: Prior vs slope (8a) and prior vs noise (8b).
Uses KDE results from fig7 cache; fig8b uses C binary in fig8b/ to convert (z, gap, slope) -> (z, sigma, prior).
Run from repo root: python src/manuscript_figures/fig8.py [kernel_width]
Default kernel_width=0.4 to match fig7 cache (kde_*_0.4_v2.csv).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
import config_utils


def read_match_file(filename):
    """Read .match file: 3 columns (z, sigma, prior) or 5 columns (z, sigma, prior, slope_error, gap_error)."""
    data = []
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    if len(values) in (3, 5):
                        data.append(values)
        return np.array(data)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def create_prior_vs_slope_plot(results_dir, figure_dir, K, kernel_width=0.1):
    """Fig 8a: prior vs slope from KDE CSV files (fig7 cache)."""
    cache_dir = os.path.join(results_dir, "fig7_cache")
    suffix = "_v2"
    pc1_file = os.path.join(cache_dir, f"kde_PC1_{kernel_width}{suffix}.csv")
    weighted_stim_file = os.path.join(cache_dir, f"kde_weightedStim_{K}states_{kernel_width}{suffix}.csv")

    weighted_stim_color = "#1f77b4"
    pc1_color = "#ff7f0e"

    if not os.path.exists(pc1_file):
        print(f"ERROR: File not found: {pc1_file}")
        return
    if not os.path.exists(weighted_stim_file):
        print(f"ERROR: File not found: {weighted_stim_file}")
        return

    pc1_data = pd.read_csv(pc1_file)
    weighted_stim_data = pd.read_csv(weighted_stim_file)

    z_range = (-2, 2)
    pc1_mask = (pc1_data["z"] >= z_range[0]) & (pc1_data["z"] <= z_range[1])
    ws_mask = (weighted_stim_data["z"] >= z_range[0]) & (weighted_stim_data["z"] <= z_range[1])

    pc1_gap = pc1_data.loc[pc1_mask, "gap"].values
    pc1_slope = pc1_data.loc[pc1_mask, "slope"].values
    ws_gap = weighted_stim_data.loc[ws_mask, "gap"].values
    ws_slope = weighted_stim_data.loc[ws_mask, "slope"].values

    line_width = 2.5
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300, facecolor="w", edgecolor="k")
    ax.plot(pc1_slope, pc1_gap, color=pc1_color, lw=line_width, label="Engagement Index")
    ax.plot(ws_slope, ws_gap, color=weighted_stim_color, lw=line_width, label="GLM Stimulus Coeff.")
    ax.set_xlabel("Slope", fontsize=label_fontsize, fontfamily="Arial", fontweight="normal")
    ax.set_ylabel("Prior-Induced Gap", fontsize=label_fontsize, fontfamily="Arial", fontweight="normal")
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 6.5)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=1.5, length=6)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc="best")
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, "fig8a.pdf"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {os.path.join(figure_dir, 'fig8a.pdf')}")


def run_gap_to_match(results_dir, fig8b_dir, kernel_width=0.1):
    """
    If ei.match / sc.match are missing, build them from fig7 cache:
    write ei.txt and sc.txt (z, gap, slope), run gap binary, save .match to results_dir.
    """
    K = config_utils.get_paper_config().get("glmhmm_K", 3)
    cache_dir = os.path.join(results_dir, "fig7_cache")
    suffix = "_v2"
    pc1_file = os.path.join(cache_dir, f"kde_PC1_{kernel_width}{suffix}.csv")
    ws_file = os.path.join(cache_dir, f"kde_weightedStim_{K}states_{kernel_width}{suffix}.csv")
    gap_bin = os.path.join(fig8b_dir, "gap")

    if not os.path.exists(gap_bin):
        print(f"fig8b: gap binary not found at {gap_bin}; cannot generate ei.match / sc.match")
        return False
    if not os.path.exists(pc1_file) or not os.path.exists(ws_file):
        print(f"fig8b: KDE cache not found ({pc1_file}, {ws_file}); run fig7 first")
        return False

    for name, path in [("ei", pc1_file), ("sc", ws_file)]:
        df = pd.read_csv(path)
        # Input format: z, gap, slope (space-separated)
        txt_path = os.path.join(fig8b_dir, f"{name}.txt")
        with open(txt_path, "w") as f:
            for _, row in df.iterrows():
                f.write(f"{row['z']} {row['gap']} {row['slope']}\n")
        with open(txt_path, "r") as f:
            out = subprocess.run(
                [gap_bin],
                stdin=f,
                capture_output=True,
                text=True,
                cwd=fig8b_dir,
                timeout=60,
            )
        if out.returncode != 0:
            print(f"fig8b: gap failed for {name}: {out.stderr}")
            return False
        # gap writes to out.convert (or stdout? readme says "mv out.convert ei.match")
        out_convert = os.path.join(fig8b_dir, "out.convert")
        match_path = os.path.join(results_dir, f"{name}.match")
        if os.path.exists(out_convert):
            with open(out_convert, "r") as src:
                with open(match_path, "w") as dst:
                    dst.write(src.read())
        elif out.stdout.strip():
            with open(match_path, "w") as dst:
                dst.write(out.stdout)
        else:
            print(f"fig8b: no output from gap for {name}")
            return False
    print("fig8b: generated ei.match and sc.match")
    return True


def main():
    parser = argparse.ArgumentParser(description="Figure 8: prior vs slope (8a), prior vs noise (8b)")
    parser.add_argument("--kernel_width", type=float, default=0.4, help="KDE kernel width (must match fig7 cache)")
    parser.add_argument("K", nargs="?", default=3, type=int, help="GLM-HMM number of states (default 3)")
    args = parser.parse_args()
    if isinstance(args.K, str):
        args.K = int(args.K)
    K = args.K
    kernel_width = args.kernel_width

    paths = config_utils.get_paths()
    results_dir = str(paths["results_dir"])
    figure_dir = str(paths["figures_dir"])
    fig8b_dir = Path(__file__).resolve().parent / "fig8b"

    ei_match = os.path.join(results_dir, "ei.match")
    sc_match = os.path.join(results_dir, "sc.match")
    if not os.path.exists(ei_match) or not os.path.exists(sc_match):
        if not run_gap_to_match(results_dir, str(fig8b_dir), kernel_width):
            print("Could not generate ei.match / sc.match. Exiting.")
            return 1

    ei_data = read_match_file(ei_match)
    sc_data = read_match_file(sc_match)
    if ei_data is None or sc_data is None:
        print("Could not read match files. Exiting.")
        return 1

    z_range = (-2, 2)
    ei_z, ei_sigma, ei_prior = ei_data[:, 0], ei_data[:, 1], ei_data[:, 2]
    sc_z, sc_sigma, sc_prior = sc_data[:, 0], sc_data[:, 1], sc_data[:, 2]
    ei_mask = (ei_z >= z_range[0]) & (ei_z <= z_range[1])
    sc_mask = (sc_z >= z_range[0]) & (sc_z <= z_range[1])
    ei_sigma_f = ei_sigma[ei_mask]
    ei_prior_f = ei_prior[ei_mask]
    sc_sigma_f = sc_sigma[sc_mask]
    sc_prior_f = sc_prior[sc_mask]

    sqrt_2pi = np.sqrt(2 * np.pi)
    ei_precision = 1.0 / (sqrt_2pi * ei_sigma_f)
    sc_precision = 1.0 / (sqrt_2pi * sc_sigma_f)

    weighted_stim_color = "#1f77b4"
    pc1_color = "#ff7f0e"
    line_width = 2.5
    label_fontsize = 14
    tick_fontsize = 12

    plt.rcParams["mathtext.fontset"] = "dejavusans"
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300, facecolor="w", edgecolor="k")
    ax.plot(ei_precision, ei_prior_f, color=pc1_color, lw=line_width, label="Engagement Index")
    ax.plot(sc_precision, sc_prior_f, color=weighted_stim_color, lw=line_width, label="GLM Stimulus Coeff.")
    ax.set_xlabel(r"$1/\sqrt{2\pi}\sigma$", fontsize=label_fontsize, fontweight="normal")
    ax.set_ylabel(r"$\hat{p}_r$", fontsize=label_fontsize, fontweight="bold")
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0.5, 0.7)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=1.5, length=6)
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, "fig8b.pdf"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved {os.path.join(figure_dir, 'fig8b.pdf')}")

    create_prior_vs_slope_plot(results_dir, figure_dir, K, kernel_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
