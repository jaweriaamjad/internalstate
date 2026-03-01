"""
Figure 6: Psychometric curves binned by behavioral variable (e.g. GLM stimulus coeff.).
Single main output: fig6.pdf (binned psychometric curves).
"""
import argparse
import os
import sys
from pathlib import Path
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src" / "manuscript_figures"))
import config_utils
from plotting_utils import pc_rotation, plot_psychometric


def _get_stim_weight(posterior_probs, glmweights_subject, weighted_stim=True, prob_only=False, K=3):
    """Stimulus weights from GLM-HMM posterior (one subject).
    glmweights_subject has K rows (one per state); use full 'stimulus' column."""
    K_actual = posterior_probs.shape[1]
    stim_arr = np.asarray(glmweights_subject["stimulus"].values, dtype=float).flatten()[:K_actual]
    if prob_only:
        max_row_index = np.argmax(stim_arr)
        return posterior_probs[:, max_row_index]
    if weighted_stim:
        return np.dot(posterior_probs, stim_arr)
    state = np.argmax(posterior_probs, axis=1) + 1
    return np.array([stim_arr[s - 1] for s in state])


def _bias_probs_match(bias_vals, target_probs, atol=0.015):
    """Match bias_probs to block targets (e.g. 0.2, 0.5, 0.8) with tolerance."""
    if np.isscalar(bias_vals):
        return any(np.isclose(bias_vals, t, atol=atol) for t in target_probs)
    return np.array([any(np.isclose(b, t, atol=atol) for t in target_probs) for b in bias_vals])


def get_filtered_data(bin_data, pLeft):
    filtered_data = bin_data[bin_data["contrast"] == 0]
    mean_right_choice = {}
    for key, probs in pLeft.items():
        mask = _bias_probs_match(filtered_data["bias_probs"].values, probs)
        subset = filtered_data.loc[mask]
        if len(subset) == 0:
            mean_right_choice[key] = np.nan
        else:
            mean_right_choice[key] = subset["right_choice"].mean()
    return mean_right_choice


def plot_binned_psychometric_curves(trials, n_bins, pLeft, bin_edges, behav_colors, behav, behav_dict, figure_dir, out_filename):
    fig, axes = plt.subplots(1, n_bins, figsize=(3 * n_bins, 4), sharey=True)
    axes = np.atleast_1d(axes)
    binned_slope, prior = [], []
    pleft_ls = ["dashed", "solid", "dotted"]
    for i in range(n_bins):
        bin_data = trials[trials[f"{behav}_bin"] == i]
        ax = axes[i]
        block_slope = []
        for idx, p in enumerate(pLeft):
            mask = _bias_probs_match(bin_data["bias_probs"].values, pLeft[p])
            subset = bin_data.loc[mask]
            if len(subset) > 0:
                slope = plot_psychometric(subset, ax=ax, color=behav_colors[behav], linestyle=pleft_ls[idx], label=p)
                block_slope.append(slope * 100)
        if i >= 0:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            if ax.get_legend() is not None:
                ax.get_legend().set_visible(False)
        else:
            ax.set_ylabel("Right choices", fontsize=20)
            ax.set_xlabel("Contrast (%)", fontsize=20)
            ax.set_xticklabels(["-100", "0", "100"])
            ax.set_yticklabels(["0", "25", "50", "75", "100"])
        mean_right_choice = get_filtered_data(bin_data, pLeft)
        prior.append(mean_right_choice["right"] - mean_right_choice["left"])
        binned_slope.append(np.mean(block_slope) if block_slope else np.nan)

    sns.despine(trim=True)
    ax2 = plt.axes(facecolor=(1, 1, 1, 0))
    ax2.set(xlim=(bin_edges[0], bin_edges[-1]), xticks=np.linspace(bin_edges[0], bin_edges[-1], n_bins + 1))
    ax2.set_xticklabels([round(edge, 2) for edge in bin_edges], fontsize=24)
    ax2.set_yticklabels("")
    ax2.tick_params(axis="x", labelsize=24)
    ax2.spines[["right", "left", "top"]].set_visible(False)
    ax2.spines[["bottom"]].set_color("black")
    ax2.spines[["bottom"]].set_linewidth(2)
    ax2.tick_params(colors="black")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.spines.bottom.set_position(("outward", 40))
    ax2.grid(False)
    ax2.set_xlabel(r"$\longrightarrow$" + behav_dict[behav] + r"$\longrightarrow$", fontsize=30, color="black")
    fig.savefig(join(figure_dir, out_filename), transparent=False, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_filename)
    return binned_slope, prior


def main():
    parser = argparse.ArgumentParser(description="Figure 6: binned psychometric curves")
    parser.add_argument("--prob_engaged", action="store_true", help="Use probability engaged (prob_only) for weightedStim")
    parser.add_argument("--z_score", action="store_true", help="Z-score the behavior variable before binning")
    parser.add_argument("K", type=int, nargs="?", default=3, help="Number of GLM-HMM states (default: 3)")
    args = parser.parse_args()

    _paths = config_utils.get_paths()
    _paper = config_utils.get_paper_config()
    data_dir = str(_paths["data_dir"])
    results_dir = str(_paths["results_dir"])
    K = getattr(args, "K", _paper.get("glmhmm_K", 3))

    figure_dir = str(_paths["figures_dir"])
    os.makedirs(figure_dir, exist_ok=True)

    behav_list = ["PC1","weightedStim","pupil"]
    behav_colors = {
        "weightedStim": "#1f77b4",
        "PC1": "#ff7f0e",
        "duration": "#2ca02c",
        "pupil": "#d62728",
    }
    if args.prob_engaged:
        behav_dict = {
            "PC1": "Engagement Index",
            "duration": "Response Time (s)",
            "pupil": "Pupil Diameter",
            "weightedStim": "Probability Engaged",
        }
    else:
        behav_dict = {
            "PC1": "Engagement Index",
            "duration": "Response Time (s)",
            "pupil": "Pupil Diameter",
            "weightedStim": "GLM Stimulus Coeff.",
        }

    for bidx, behav in enumerate(behav_list):
        if behav == "pupil":
            pupil_data = join(results_dir, "pupil_logs")
            hq_file = join(pupil_data, "high_quality_pupil_sessions.txt")
            try:
                with open(hq_file) as f:
                    sessions = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                sessions = []
            if not sessions:
                print(f"Skipping {behav}: no high-quality pupil sessions in {hq_file}")
                continue
            data_df = pd.read_parquet(join(pupil_data, "prestim_pupil_w_PCs.pqt"))
            data_df["RT"] = data_df["FMOT_time"] - data_df["stimOn_time"]
            trials = data_df[data_df["eid"].isin(sessions)][["RT", "bias_probs", "contrast", "choice", "feedback", "starting_diameters"]].copy()
            trials["pupil"] = trials["starting_diameters"]
            if len(trials) < 100:
                print(f"Skipping {behav}: too few pupil trials ({len(trials)})")
                continue
        elif behav == "weightedStim":
            glmhmm_datadir = join(data_dir, "glmhmm", "bysubject")
            glmhmm_weightsdir = join(results_dir, "glmhmm_logs")
            glmweights = pd.read_parquet(join(glmhmm_weightsdir, f"indivfit_{K}statesglmweights.pqt"))
            subjects = glmweights["subject"].unique()
            subject_dfs = []
            for subject in subjects:
                glmhmm_subject_df = pd.read_parquet(join(glmhmm_datadir, f"{subject}_trials_table.pqt"))
                glmhmm_subject_df["RT"] = glmhmm_subject_df["firstMovement_times"] - glmhmm_subject_df["goCueTrigger_times"]
                selected_df = glmhmm_subject_df[["RT", "choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft"]].copy()
                posterior_probs = np.stack(glmhmm_subject_df[f"glm-hmm_{K}"])
                glmweights_subject = glmweights.loc[glmweights["subject"] == subject].reset_index(drop=True)
                stim_weights = _get_stim_weight(
                    posterior_probs, glmweights_subject,
                    weighted_stim=not args.prob_engaged,
                    prob_only=args.prob_engaged,
                    K=K,
                )
                selected_df["weightedStim"] = stim_weights
                subject_dfs.append(selected_df)
            trials = pd.concat(subject_dfs, ignore_index=True)
            trials["contrast"] = trials["contrastRight"]
            trials.loc[trials["contrast"].isnull(), "contrast"] = -trials["contrastLeft"]
            trials["bias_probs"] = trials["probabilityLeft"]
            trials["feedback"] = trials["feedbackType"]
        else:
            vae_latents_path = join(results_dir, "vae_logs", "latents")
            data_df = pd.read_parquet(join(vae_latents_path, "datalatents_3PC.pqt"))
            data_df["RT"] = data_df["FMOT_time"] - data_df["stimOn_time"]
            trials = data_df[["bias_probs", "contrast", "choice", "feedback", "RT", "duration", "PC1", "PC2", "PC3"]].copy()
            pc_rotation(trials, 30)

        trials["correct"] = trials["feedback"].astype(int)
        trials.loc[trials["correct"] == -1, "correct"] = 0
        trials["right_choice"] = -trials["choice"]
        trials.loc[trials["right_choice"] == -1, "right_choice"] = 0
        pLeft = {"right": [0.2], "basic": [0.5], "left": [0.8]}

        trials["RT"] = trials["RT"].clip(upper=60, lower=-0.4)
        n_bins = 6
        # Always z-score the behavior variable for comparable binning across behav types
        scaler = StandardScaler()
        trials[behav] = scaler.fit_transform(trials[[behav]]).flatten()

        trials[f"{behav}_bin"], bin_edges = pd.qcut(trials[behav], q=n_bins, labels=False, retbins=True, duplicates="drop")
        n_bins_actual = trials[f"{behav}_bin"].nunique()
        if n_bins_actual < 2:
            print("Skipping fig6: too few bins after qcut")
            return
        # Recompute edges from data in case qcut dropped duplicate edges
        bin_edges = [trials.loc[trials[f"{behav}_bin"] == i, behav].min() for i in range(n_bins_actual)]
        bin_edges.append(trials.loc[trials[f"{behav}_bin"] == n_bins_actual - 1, behav].max())
        bin_edges = np.array(bin_edges)
        fig_letter = chr(ord("a") + bidx)
        plot_binned_psychometric_curves(
            trials, n_bins_actual, pLeft, bin_edges, behav_colors, behav, behav_dict, figure_dir, f"fig6{fig_letter}.pdf"
        )

    print("Figures saved to", figure_dir)


if __name__ == "__main__":
    main()
