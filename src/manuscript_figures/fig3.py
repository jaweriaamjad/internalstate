"""
Figure 3: PC latent space — scatter (fig3b) and wheel trajectories in PC space (fig3c).
Requires VAE latents: datalatents_3PC.pqt under results_dir/vae_logs/latents/.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src" / "manuscript_figures"))
import config_utils

from plotting_utils import pc_rotation, scatterplot_latents, plot_wheelInPCspace


def print_data_summary(data_latents):
    print(data_latents.columns.tolist())
    print(f"Subjects: {data_latents.subject.nunique()}, Sessions: {data_latents.eid.nunique()}")
    eid_counts = data_latents.groupby('subject')['eid'].nunique()
    print(f"Min sessions per subject: {eid_counts.min()}")


def main():
    _paths = config_utils.get_paths()
    results_dir = str(_paths["results_dir"])
    figures_dir = str(_paths["figures_dir"])

    data_path = join(results_dir, "vae_logs", "latents")
    os.makedirs(figures_dir, exist_ok=True)

    data_df = pd.read_parquet(join(data_path, "datalatents_3PC.pqt"))
    print_data_summary(data_df)

    plt.style.use("default")
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    pc_rotation(data_df, 30)

    print("Creating PC plots (fig3b, fig3c)...")
    selected_indices = plot_wheelInPCspace(
        data_df, 8, 5, 3, embed_type="PC",
        out_path=join(figures_dir, "fig3c.pdf"), z_score=True
    )
    scatterplot_latents(data_df, selected_indices, out_path=join(figures_dir, "fig3b.png"))
    print("Done.")


if __name__ == "__main__":
    main()
