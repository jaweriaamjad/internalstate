"""
Step 3 (VAE): Extract latents for all trials using a trained encoder.
Folder (model/run name) and probL come from config; no command-line args.
Reads: data_dir/processed_for_vae/{tag}/allsubjects/allsubjects.pqt
       Encoder from: vae_weights/vaelstm_enc (if exists) else results_dir/vae_logs/model_weights/vaelstm_enc
Writes: results_dir/vaelogs/all_latents/{folder}/data_with_vaelatents.pqt
"""

import os
import sys
from pathlib import Path
from os.path import join

import pandas as pd

_VAE_DIR = Path(__file__).resolve().parent
_REPO = _VAE_DIR.parent.parent  # src/vae -> src -> repo
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_VAE_DIR))

import config_utils
import vae_utils

paths = config_utils.get_paths()
paper = config_utils.get_paper_config()
# Folder where the trained model is saved (same name as in 2-fit_vae output)
folder = paper["vae_run_folder"]
probL = "all"
probLeft = [0.5, 0.2, 0.8]

tag = paper["tag"]
results_dir = paths["results_dir"]
data_dir = paths["data_dir"]
data_path = join(data_dir, "processed_for_vae", tag)

# Check for encoder weights: vae_weights/ first, then results/vae_logs/model_weights/
vae_weights_dir = _REPO / "vae_weights"
if (vae_weights_dir / "vaelstm_enc").exists():
    encoder_dir = str(vae_weights_dir)
    print(f"Using encoder from vae_weights/")
else:
    encoder_dir = join(results_dir, "vae_logs", "model_weights")
    print(f"Using encoder from {encoder_dir}")

os.makedirs(join(results_dir, "vaelogs", "all_latents", folder), exist_ok=True)

df = pd.read_parquet(join(data_path, "allsubjects", "allsubjects.pqt"))
vae_utils.get_vae_latents(df, probLeft, folder, results_dir, encoder_dir=encoder_dir)
