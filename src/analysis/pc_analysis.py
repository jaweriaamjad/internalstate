"""
PC analysis: load data_with_vaelatents, run PCA, save datalatents_3PC.pqt.
No figure plotting — use datalatents_3PC.pqt in figure scripts (e.g. fig3.py) for scatterplot_latents, plot_wheelInPCspace, etc.
Reads: results_dir/vaelogs/all_latents/{folder}/data_with_vaelatents.pqt
Writes: results_dir/vaelogs/all_latents/{folder}/datalatents_3PC.pqt
"""

import os
import sys
import argparse
from pathlib import Path
from os.path import join

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_utils


def load_data(results_dir, folder):
    file_path = join(results_dir, "vaelogs", "all_latents", folder, "data_with_vaelatents.pqt")
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    raise FileNotFoundError(f"File not found: {file_path}")


def perform_pca(data_latents, n_components=3):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(np.concatenate((
        np.stack(data_latents['z_mean'].values),
        np.stack(data_latents['z_log_sigma'].values)), axis=1))
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    data_latents = data_latents.copy()
    data_latents[pc_columns] = pcs
    print(f"Variance explained: {pca.explained_variance_}")
    print(f"Variance ratio: {pca.explained_variance_ratio_}")
    return data_latents


def save_embeddings(data_latents, results_dir, folder, n_components=3):
    out_path = join(results_dir, "vaelogs", "all_latents", folder, f"datalatents_{n_components}PC.pqt")
    data_latents.to_parquet(out_path)
    print(f"Saved: {out_path}")


def main(folder, n_components=3):
    paths = config_utils.get_paths()
    results_dir = paths["results_dir"]
    data_latents = load_data(results_dir, folder)
    data_latents = perform_pca(data_latents, n_components=n_components)
    save_embeddings(data_latents, results_dir, folder, n_components=n_components)
    print(f"Subjects: {data_latents.subject.nunique()}, sessions: {data_latents.eid.nunique()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute PCA on VAE latents and save datalatents_3PC.pqt")
    parser.add_argument('--folder', type=str, default=None, help="VAE run folder (default: from paper_config)")
    parser.add_argument('--n_components', type=int, default=3, help="Number of PC components")
    args = parser.parse_args()
    paper = config_utils.get_paper_config()
    folder = args.folder or paper["vae_run_folder"]
    main(folder, n_components=args.n_components)
