"""
Figure 7: KDE-based overlaid plots (gap, slope, performance vs z).
Same logic as wheel-movement-project manuscript fig6; paths for internalstate_code.
Outputs: fig7a.pdf (prior-induced gap), fig7b.pdf (slope), fig7c.pdf (fraction correct).
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
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src" / "manuscript_figures"))
import config_utils
from plotting_utils import pc_rotation


def get_stim_weight(posterior_probs, glmweights_subject, weighted_stim=True, prob_only=False, K=3):
    """Stimulus weights from GLM-HMM posterior. glmweights_subject has K rows; use full 'stimulus' column."""
    K_actual = posterior_probs.shape[1]
    stim_arr = np.asarray(glmweights_subject["stimulus"].values, dtype=float).flatten()[:K_actual]
    if prob_only:
        max_row_index = np.argmax(stim_arr)
        return posterior_probs[:, max_row_index]
    if weighted_stim:
        return np.dot(posterior_probs, stim_arr)
    state = np.argmax(posterior_probs, axis=1) + 1
    return np.array([stim_arr[s - 1] for s in state])


def _match_float(arr, target, atol=0.015):
    """Match array to target with tolerance (for contrast/block)."""
    return np.isclose(arr, target, atol=atol)


def kde_choice_prob(trials, z_grid, contrast, block, kernel_width=0.1, return_sem=False):
    if block is None:
        mask = _match_float(trials["contrast"].values, contrast)
    else:
        mask = _match_float(trials["contrast"].values, contrast) & _match_float(trials["block"].values, block)
    z_i = trials.loc[mask, "z"].values
    x_i = trials.loc[mask, "right_choice"].values
    p_right = np.zeros_like(z_grid)
    sem = np.zeros_like(z_grid)
    for j, z in enumerate(z_grid):
        weights = norm.pdf(z - z_i, scale=kernel_width)
        Z = np.sum(weights)
        if Z > 0:
            xbar = np.sum(weights * x_i) / Z
            p_right[j] = xbar
            if return_sem:
                var = np.sum(weights * (x_i - xbar) ** 2) / Z
                sem2 = var * np.sum((weights / Z) ** 2)
                sem[j] = np.sqrt(sem2)
        else:
            p_right[j] = np.nan
            sem[j] = np.nan
    return p_right, sem


def kde_performance(trials, z_grid, kernel_width=0.1, return_sem=False):
    z_i = trials["z"].values
    correct = trials["correct"].values
    perf = np.zeros_like(z_grid)
    sem = np.zeros_like(z_grid)
    for j, z in enumerate(z_grid):
        weights = norm.pdf(z - z_i, scale=kernel_width)
        Z = np.sum(weights)
        if Z > 0:
            xbar = np.sum(weights * correct) / Z
            perf[j] = xbar
            if return_sem:
                var = np.sum(weights * (correct - xbar) ** 2) / Z
                sem2 = var * np.sum((weights / Z) ** 2)
                sem[j] = np.sqrt(sem2)
        else:
            perf[j] = np.nan
            sem[j] = np.nan
    return perf, sem


def compute_gap_slope(trials, z_grid, kernel_width=0.1, rightblock=0.2, leftblock=0.8, return_sem=False):
    p_right_0_right, sem_0_right = kde_choice_prob(
        trials, z_grid, contrast=0, block=rightblock, kernel_width=kernel_width, return_sem=return_sem
    )
    p_right_0_left, sem_0_left = kde_choice_prob(
        trials, z_grid, contrast=0, block=leftblock, kernel_width=kernel_width, return_sem=return_sem
    )
    gap = p_right_0_right - p_right_0_left
    gap_sem = np.sqrt(sem_0_right ** 2 + sem_0_left ** 2) if return_sem else np.zeros_like(gap)
    p_right_p, sem_p = kde_choice_prob(
        trials, z_grid, contrast=0.0625, block=None, kernel_width=kernel_width, return_sem=return_sem
    )
    p_right_m, sem_m = kde_choice_prob(
        trials, z_grid, contrast=-0.0625, block=None, kernel_width=kernel_width, return_sem=return_sem
    )
    slope = (p_right_p - p_right_m) / 0.125
    slope_sem = np.sqrt(sem_p ** 2 + sem_m ** 2) / 0.125 if return_sem else np.zeros_like(slope)
    if return_sem:
        return gap, gap_sem, slope, slope_sem
    return gap, slope


def main():
    parser = argparse.ArgumentParser(description="Figure 7: KDE overlaid plots (gap, slope, perf vs z)")
    parser.add_argument("--z_score", action="store_true", help="Z-score the behavioral variable")
    parser.add_argument("--kernel_width", type=float, default=0.4, help="Kernel width for KDE")
    parser.add_argument("--prob_engaged", action="store_true", help="Use probability engaged for weightedStim")
    parser.add_argument("K", type=int, nargs="?", default=3, help="Number of GLM-HMM states (default: 3)")
    args = parser.parse_args()

    _paths = config_utils.get_paths()
    _paper = config_utils.get_paper_config()
    data_dir = str(_paths["data_dir"])
    results_dir = str(_paths["results_dir"])
    K = getattr(args, "K", _paper.get("glmhmm_K", 3))

    figure_dir = str(_paths["figures_dir"])
    os.makedirs(figure_dir, exist_ok=True)
    cache_dir = join(results_dir, "fig7_cache")
    os.makedirs(cache_dir, exist_ok=True)

    prob_engaged = args.prob_engaged
    if prob_engaged:
        behav_list = ["prob_engaged"]
        behav_dict = {"PC1": "Engagement Index", "prob_engaged": "Probability Engaged", "pupil": "Pupil Diameter"}
    else:
        behav_list = ["weightedStim", "PC1", "pupil"]
        behav_dict = {
            "PC1": "Engagement Index",
            "weightedStim": "GLM Stimulus Coeff.",
            "pupil": "Pupil Diameter",
        }
    behav_colors = {
        "weightedStim": "#1f77b4",
        "PC1": "#ff7f0e",
        "pupil": "#d62728",
        "prob_engaged": "#1f77b4",
    }

    rightblock = 0.2
    leftblock = 0.8
    all_results = {}

    for behav in behav_list:
        print(f"Analysing {behav_dict[behav]}.........")
        # Cache key includes v2 = always z-score + fixed z_grid [-2,2] for paper-matching overlay
        cache_suffix = "_v2"
        if behav in ["weightedStim", "prob_engaged"]:
            out_csv = join(cache_dir, f"kde_{behav}_{K}states_{args.kernel_width}{cache_suffix}.csv")
        else:
            out_csv = join(cache_dir, f"kde_{behav}_{args.kernel_width}{cache_suffix}.csv")

        if os.path.exists(out_csv):
            print(f"Loading existing results from {out_csv}")
            df_out = pd.read_csv(out_csv)
            z_grid = df_out["z"].values
            gap = df_out["gap"].values
            gap_sem = df_out["gap_sem"].values
            slope = df_out["slope"].values
            slope_sem = df_out["slope_sem"].values
            perf = df_out["performance"].values
            perf_sem = df_out["performance_sem"].values
        else:
            if behav == "PC1":
                data_df = pd.read_parquet(join(results_dir, "vae_logs", "latents", "datalatents_3PC.pqt"))
                data_df["RT"] = data_df["FMOT_time"] - data_df["stimOn_time"]
                trials = data_df[["RT", "bias_probs", "contrast", "choice", "feedback", "PC1", "PC2", "PC3"]].copy()
                trials["block"] = trials["bias_probs"]
                pc_rotation(trials, 30)
            elif behav == "pupil":
                pupil_dir = join(results_dir, "pupil_logs")
                hq_file = join(pupil_dir, "high_quality_pupil_sessions.txt")
                try:
                    with open(hq_file) as f:
                        session_list = [line.strip() for line in f if line.strip()]
                except FileNotFoundError:
                    print(f"Skipping {behav}: {hq_file} not found")
                    continue
                if not session_list:
                    print(f"Skipping {behav}: no high-quality pupil sessions")
                    continue
                data_df = pd.read_parquet(join(pupil_dir, "prestim_pupil_w_PCs.pqt"))
                data_df = data_df[data_df["eid"].isin(session_list)]
                data_df["RT"] = data_df["FMOT_time"] - data_df["stimOn_time"]
                trials = data_df[["RT", "bias_probs", "contrast", "choice", "feedback", "starting_diameters"]].copy()
                trials["RT"] = trials["RT"].clip(upper=60, lower=-0.4)
                trials["fs"] = (trials["RT"] < 0.08).astype(int)
                trials = trials[trials["fs"] == 0]
                trials["pupil"] = trials["starting_diameters"]
                trials["block"] = trials["bias_probs"]
            else:
                glmhmm_datadir = join(data_dir, "glmhmm", "bysubject")
                glmhmm_weightsdir = join(results_dir, "glmhmm_logs")
                glmweights = pd.read_parquet(join(glmhmm_weightsdir, f"indivfit_{K}statesglmweights.pqt"))
                subjects = glmweights["subject"].unique()
                subject_dfs = []
                for subject in subjects:
                    glmhmm_subject_df = pd.read_parquet(join(glmhmm_datadir, f"{subject}_trials_table.pqt"))
                    glmhmm_subject_df["RT"] = (
                        glmhmm_subject_df["firstMovement_times"] - glmhmm_subject_df["goCueTrigger_times"]
                    )
                    selected_df = glmhmm_subject_df[
                        ["RT", "choice", "contrastLeft", "contrastRight", "feedbackType", "probabilityLeft"]
                    ].copy()
                    posterior_probs = np.stack(glmhmm_subject_df[f"glm-hmm_{K}"])
                    glmweights_subject = glmweights.loc[glmweights["subject"] == subject].reset_index(drop=True)
                    stim_weights = get_stim_weight(
                        posterior_probs, glmweights_subject,
                        weighted_stim=not prob_engaged, prob_only=prob_engaged, K=K
                    )
                    selected_df[behav] = stim_weights
                    subject_dfs.append(selected_df)
                trials = pd.concat(subject_dfs, ignore_index=True)
                trials["contrast"] = trials["contrastRight"]
                trials.loc[trials["contrast"].isnull(), "contrast"] = -trials["contrastLeft"]
                trials["block"] = trials["probabilityLeft"]
                trials["feedback"] = trials["feedbackType"]

            trials["correct"] = trials["feedback"].astype(int)
            trials.loc[trials["correct"] == -1, "correct"] = 0
            trials["right_choice"] = -trials["choice"]
            trials.loc[trials["right_choice"] == -1, "right_choice"] = 0
            trials["RT"] = trials["RT"].clip(upper=60, lower=-0.4)
            trials["fs"] = (trials["RT"] < 0.08).astype(int)
            trials = trials[trials["fs"] == 0]

            if len(trials) < 100:
                print(f"Skipping {behav}: too few trials ({len(trials)})")
                continue

            # Always z-score so overlaid curves share the same x-scale (paper uses xlim -2 to 2)
            scaler = StandardScaler()
            trials[behav] = scaler.fit_transform(trials[[behav]]).flatten()
            trials["z"] = trials[behav]

            # Contrast in fraction for KDE (0.0625 = 6.25%); convert if data are in percentage
            if trials["contrast"].abs().max() > 1.5:
                trials["contrast"] = trials["contrast"] / 100.0

            # Common z grid for all behaviors so overlaid plots align (paper: xlim -2, 2)
            z_grid = np.linspace(-2, 2, 100)
            gap, gap_sem, slope, slope_sem = compute_gap_slope(
                trials, z_grid, args.kernel_width, rightblock, leftblock, return_sem=True
            )
            perf, perf_sem = kde_performance(trials, z_grid, args.kernel_width, return_sem=True)
            pd.DataFrame({
                "z": z_grid,
                "gap": gap,
                "gap_sem": gap_sem,
                "slope": slope,
                "slope_sem": slope_sem,
                "performance": perf,
                "performance_sem": perf_sem,
            }).to_csv(out_csv, index=False)
            print(f"Saved {out_csv}")

        all_results[behav] = {
            "z_grid": z_grid,
            "gap": gap,
            "slope": slope,
            "perf": perf,
            "gap_ci": (gap - gap_sem, gap + gap_sem),
            "slope_ci": (slope - slope_sem, slope + slope_sem),
            "perf_ci": (perf - perf_sem, perf + perf_sem),
        }

    if not all_results:
        print("No results to plot.")
        return

    print("Creating overlaid plots (fig7a, fig7b, fig7c)...")
    plt.style.use("default")
    sns.set_style("whitegrid", {"axes.grid": False})
    line_width = 2.5
    alpha_ci = 0.3
    label_fontsize = 16
    tick_fontsize = 12
    legend_fontsize = 12
    metrics = ["gap", "slope", "perf"]
    y_labels = ["Prior-Induced Gap", "Slope", "Fraction of Correct Choices"]
    y_limits = [(-0.05, 0.5), (0, None), (0, 1.05)]
    fig_names = ["fig7a.pdf", "fig7b.pdf", "fig7c.pdf"]

    for metric, y_label, y_lim, fig_name in zip(metrics, y_labels, y_limits, fig_names):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5), dpi=300, facecolor="w", edgecolor="k")
        for behav_key, behav_data in all_results.items():
            color = behav_colors.get(behav_key, "#000000")
            label = behav_dict.get(behav_key, behav_key)
            if metric == "gap":
                data, ci = behav_data["gap"], behav_data.get("gap_ci")
            elif metric == "slope":
                data, ci = behav_data["slope"], behav_data.get("slope_ci")
            else:
                data, ci = behav_data["perf"], behav_data.get("perf_ci")
            z_grid = behav_data["z_grid"]
            ax.plot(z_grid, data, color=color, lw=line_width, label=label)
            if ci is not None:
                ax.fill_between(z_grid, ci[0], ci[1], color=color, alpha=alpha_ci)
        ax.set_ylabel(y_label, fontsize=label_fontsize, fontfamily="Arial", fontweight="normal")
        ax.set_xlabel("z", fontsize=label_fontsize, fontfamily="Arial", fontweight="normal")
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=1.5, length=6)
        ax.set_xlim(-2, 2)
        if metric == "slope":
            ax.legend(fontsize=legend_fontsize, frameon=False, loc="best")
        if y_lim[1] is None:
            max_val = max(
                max(behav_data[metric].max() for behav_data in all_results.values()),
                0.0
            )
            ax.set_ylim(y_lim[0], max_val * 1.1)
        else:
            ax.set_ylim(y_lim)
        if metric == "perf":
            ax.set_ylim(0.5, 1.0)
        sns.despine(ax=ax)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)
        plt.savefig(join(figure_dir, fig_name), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()
        print(f"Saved {fig_name}")

    print(f"Figure 7 saved to {figure_dir}/ (fig7a, fig7b, fig7c)")


if __name__ == "__main__":
    main()
