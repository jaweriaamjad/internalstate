"""
Figure 5: KDE plots — EI vs Stim, EI vs Pupil, Stim vs Pupil (raw and autocorrelation).
Same logic as manuscript fig4; paths point to internalstate_code layout.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, gaussian_kde
from sklearn.preprocessing import StandardScaler

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src" / "manuscript_figures"))
import config_utils
from plotting_utils import pc_rotation


def get_stim_weight(posterior_probs, glmweights_subject, weighted_stim=True):
    """Same as paper utils.get_stim_weight (weighted_stim branch).
    glmweights_subject has K rows (one per state); use full 'stimulus' column."""
    K = posterior_probs.shape[1]
    stim_arr = np.asarray(glmweights_subject["stimulus"].values, dtype=float).flatten()
    stim_arr = stim_arr[:K]
    if weighted_stim:
        return np.dot(posterior_probs, stim_arr)
    state = np.argmax(posterior_probs, axis=1) + 1
    return np.array([stim_arr[s - 1] for s in state])


def lag1_autocorr(x):
    if len(x) < 2:
        return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def style_axes(ax):
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def create_kde_plot(data, x_col, y_col, xlabel, ylabel, title, filename, figure_dir, cmap="Blues"):
    clean_data = data.dropna(subset=[x_col, y_col])
    if len(clean_data) < 10:
        print(f"Warning: Not enough data for KDE plot ({len(clean_data)} points)")
        return
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    x = clean_data[x_col].values
    y = clean_data[y_col].values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    xx, yy = np.mgrid[
        (x_min - x_pad) : (x_max + x_pad) : 100j,
        (y_min - y_pad) : (y_max + y_pad) : 100j,
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    try:
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f_normalized = f / f.max()
        im = ax.contourf(xx, yy, f_normalized, levels=20, cmap=cmap, alpha=0.8)
        ax.contour(xx, yy, f_normalized, levels=10, colors="black", alpha=0.3, linewidths=0.5)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label("Density", fontsize=14, fontfamily="Arial", fontweight="normal")
        cbar.ax.tick_params(labelsize=10)
    except Exception as e:
        print(f"Error creating KDE: {e}")
        ax.scatter(x, y, alpha=0.6, s=30, color="#1f77b4", edgecolors="black", linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=14, fontfamily="Arial", fontweight="normal")
    ax.set_ylabel(ylabel, fontsize=14, fontfamily="Arial", fontweight="normal")
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    style_axes(ax)
    plt.tight_layout()
    plt.savefig(join(figure_dir, filename), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved KDE plot: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Autocorrelation analysis with scatter or KDE plots")
    parser.add_argument("--scatter", action="store_true", help="Create scatter plots instead of KDE plots")
    args = parser.parse_args()

    plot_type = "scatter" if args.scatter else "kde"
    print(f"Creating {plot_type} plots...")

    _paths = config_utils.get_paths()
    _paper = config_utils.get_paper_config()
    data_dir = str(_paths["data_dir"])
    results_dir = str(_paths["results_dir"])
    K = _paper.get("glmhmm_K", 3)
    rotate_PCs = True
    z_score = True

    figure_dir = str(_paths["figures_dir"])
    os.makedirs(figure_dir, exist_ok=True)
    glmhmm_datadir = join(data_dir, "glmhmm", "bysubject")
    glmhmm_weightsdir = join(results_dir, "glmhmm_logs")

    glmweights = pd.read_parquet(join(glmhmm_weightsdir, f"indivfit_{K}statesglmweights.pqt"))
    data_df = pd.read_parquet(join(results_dir, "vae_logs", "latents", "datalatents_3PC.pqt"))
    if rotate_PCs:
        pc_rotation(data_df, 30)

    emptywheelpos_dir = join(data_dir, "processed_for_vae")
    with open(join(emptywheelpos_dir, "session_emptytrials_dict.json"), "r") as f:
        emptytrials_dict = json.load(f)

    pupildata_directory = join(results_dir, "pupil_logs")
    with open(join(pupildata_directory, "high_quality_pupil_sessions.txt"), "r") as f:
        pupil_sessions = [line.strip() for line in f if line.strip()]
    print(f"Number of high-quality pupil sessions: {len(pupil_sessions)}")

    pupil_df = pd.read_parquet(join(pupildata_directory, "prestim_pupil_w_PCs.pqt"))
    pupil_trials = pupil_df[pupil_df["eid"].isin(pupil_sessions)][
        ["subject", "eid", "starting_diameters", "PC1", "PC2"]
    ]
    if rotate_PCs:
        pc_rotation(pupil_trials, 30)

    data_cache_dir = join(results_dir, "fig5_cache")
    os.makedirs(data_cache_dir, exist_ok=True)
    all_subjects_file = join(data_cache_dir, "ei_stim_df.pqt")
    pupil_ei_stim_file = join(data_cache_dir, "pupil_ei_stim_df.pqt")

    if os.path.exists(all_subjects_file) and os.path.exists(pupil_ei_stim_file):
        print("Loading existing processed data...")
        try:
            all_subjects_df = pd.read_parquet(all_subjects_file)
            pupil_ei_stim_df = pd.read_parquet(pupil_ei_stim_file)
            print(f"Successfully loaded existing data:")
            print(f"  - All subjects: {len(all_subjects_df)} trials from {all_subjects_df['eid'].nunique()} sessions")
            print(f"  - Pupil sessions: {len(pupil_ei_stim_df)} trials from {pupil_ei_stim_df['eid'].nunique()} sessions")
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Will reprocess data...")
            all_subjects_df = None
            pupil_ei_stim_df = None
    else:
        print("No existing processed data found. Will process data...")
        all_subjects_df = None
        pupil_ei_stim_df = None

    if all_subjects_df is None:
        print("Processing all subjects for EI vs Stim autocorrelation...")
        subjects = data_df.subject.unique()
        PC1_all, stim_weights_all, eid_all = np.array([]), np.array([]), np.array([])
        for sidx, subject in enumerate(subjects):
            print(f"{sidx + 1:03d}/{len(subjects)} --> {subject}")
            glmweights_subject = glmweights.loc[glmweights["subject"] == subject].reset_index(drop=True)
            try:
                glhmm_df = pd.read_parquet(join(glmhmm_datadir, f"{subject}_trials_table.pqt"))
                glhmm_df["contrast"] = glhmm_df["contrastRight"].fillna(-glhmm_df["contrastLeft"])
                sessions = glhmm_df.session.unique()
                for eidx, session in enumerate(sessions):
                    print(f"    {eidx + 1:03d}/{len(sessions)} --> {session}")
                    try:
                        glhmm_session_df = glhmm_df[glhmm_df["session"] == session].reset_index(drop=True)
                        glhmm_session_df = glhmm_session_df.drop(emptytrials_dict[session], axis=0)
                        posterior_probs = np.stack(glhmm_session_df[f"glm-hmm_{K}"])
                        stim_weights = get_stim_weight(posterior_probs, glmweights_subject, weighted_stim=True)
                        PC1 = data_df.loc[data_df["eid"] == session, "PC1"].values
                        assert PC1.shape[0] > 0, "The session was not part of VAE dataset."
                        assert PC1.shape[0] == posterior_probs.shape[0], "Mismatch in number of trials."
                        PC1_all = np.concatenate([PC1_all, PC1])
                        stim_weights_all = np.concatenate([stim_weights_all, stim_weights])
                        eid_all = np.concatenate([eid_all, np.array([session] * len(PC1))])
                    except Exception as e:
                        print(f"Error in session {session}: {e}")
            except Exception as e:
                print(f"Error during subject processing for {subject}: {e}")
        all_subjects_df = pd.DataFrame({"stim_weights": stim_weights_all, "PC1": PC1_all, "eid": eid_all})
        print(f"Saving all subjects data to {all_subjects_file}")
        all_subjects_df.to_parquet(all_subjects_file)

    if pupil_ei_stim_df is None:
        print("Processing pupil sessions for pupil autocorrelation...")
        pupil_ei_stim_df = []
        for session in set(pupil_trials.eid.unique()):
            subject = pupil_trials[pupil_trials["eid"] == session]["subject"].unique()[0]
            print(f"Processing {subject}: {session}")
            trials_path = join(glmhmm_datadir, f"{subject}_trials_table.pqt")
            glmhmm_df = pd.read_parquet(trials_path)
            session_df = glmhmm_df[glmhmm_df["session"] == session].reset_index(drop=True)
            empty_trials = emptytrials_dict.get(session, [])
            session_df.drop(empty_trials, axis=0, inplace=True, errors="ignore")
            trials_df = pupil_trials[pupil_trials["eid"] == session].reset_index(drop=True)
            if session_df.shape[0] != trials_df.shape[0]:
                raise ValueError(
                    f"Mismatch in trial count: GLM-HMM={session_df.shape[0]}, Trials={trials_df.shape[0]}"
                )
            posterior_probs = np.stack(session_df[f"glm-hmm_{K}"])
            glmweights_subject = glmweights[glmweights["subject"] == subject].reset_index(drop=True)
            stim_weights = get_stim_weight(posterior_probs, glmweights_subject, weighted_stim=True)
            selected_df = trials_df[["eid", "subject", "starting_diameters", "PC1"]].copy()
            selected_df["weightedStim"] = stim_weights
            pupil_ei_stim_df.append(selected_df)
        pupil_ei_stim_df = pd.concat(pupil_ei_stim_df, ignore_index=True)
        print(f"Saving pupil data to {pupil_ei_stim_file}")
        pupil_ei_stim_df.to_parquet(pupil_ei_stim_file)

    print("Computing autocorrelations...")
    all_subjects_autocorr = (
        all_subjects_df.groupby("eid")
        .agg({"PC1": lag1_autocorr, "stim_weights": lag1_autocorr})
        .reset_index()
        .rename(columns={"PC1": "ei_lag1_autocorr", "stim_weights": "stim_lag1_autocorr"})
    )
    pupil_autocorr = (
        pupil_ei_stim_df.groupby("eid")
        .agg({"PC1": lag1_autocorr, "weightedStim": lag1_autocorr, "starting_diameters": lag1_autocorr})
        .reset_index()
        .rename(columns={
            "PC1": "ei_lag1_autocorr",
            "weightedStim": "stim_lag1_autocorr",
            "starting_diameters": "pupil_lag1_autocorr",
        })
    )

    print("Creating plots...")
    plt.style.use("default")
    sns.set_style("whitegrid", {"axes.grid": False})

    if plot_type == "scatter":
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.scatterplot(
            data=all_subjects_autocorr,
            x="ei_lag1_autocorr",
            y="stim_lag1_autocorr",
            s=80,
            alpha=0.7,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_xlabel("Engagement Index $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.set_ylabel("GLM Stimulus Coeff. $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(join(figure_dir, "fig5d.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.scatterplot(
            data=pupil_autocorr,
            x="ei_lag1_autocorr",
            y="pupil_lag1_autocorr",
            s=80,
            alpha=0.7,
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_xlabel("Engagement Index $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.set_ylabel("Pupil Diameter $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(join(figure_dir, "fig5e.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sns.scatterplot(
            data=pupil_autocorr,
            x="stim_lag1_autocorr",
            y="pupil_lag1_autocorr",
            s=80,
            alpha=0.7,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_xlabel("GLM Stimulus Coeff. $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.set_ylabel("Pupil Diameter $\\rho_1$", fontsize=18, fontfamily="Arial", fontweight="normal")
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        style_axes(ax)
        plt.tight_layout()
        plt.savefig(join(figure_dir, "fig5f.pdf"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        cmap = "Greys"
        create_kde_plot(
            all_subjects_autocorr,
            "ei_lag1_autocorr",
            "stim_lag1_autocorr",
            "Engagement Index $\\rho_1$",
            "GLM Stimulus Coeff. $\\rho_1$",
            "EI vs Stim Autocorrelation",
            "fig5d.pdf",
            figure_dir,
            cmap,
        )
        create_kde_plot(
            pupil_autocorr,
            "ei_lag1_autocorr",
            "pupil_lag1_autocorr",
            "Engagement Index $\\rho_1$",
            "Pupil Diameter $\\rho_1$",
            "EI vs Pupil Autocorrelation",
            "fig5e.pdf",
            figure_dir,
            cmap,
        )
        create_kde_plot(
            pupil_autocorr,
            "stim_lag1_autocorr",
            "pupil_lag1_autocorr",
            "GLM Stimulus Coeff. $\\rho_1$",
            "Pupil Diameter $\\rho_1$",
            "Stim vs Pupil Autocorrelation",
            "fig5f.pdf",
            figure_dir,
            cmap,
        )

    print("Creating KDE plots for raw data...")
    if z_score:
        scaler = StandardScaler()
        all_subjects_df["PC1"] = scaler.fit_transform(all_subjects_df[["PC1"]]).flatten()
        all_subjects_df["stim_weights"] = scaler.fit_transform(all_subjects_df[["stim_weights"]]).flatten()
        pupil_ei_stim_df["PC1"] = scaler.fit_transform(pupil_ei_stim_df[["PC1"]]).flatten()
        pupil_ei_stim_df["weightedStim"] = scaler.fit_transform(pupil_ei_stim_df[["weightedStim"]]).flatten()
        pupil_ei_stim_df["starting_diameters"] = scaler.fit_transform(pupil_ei_stim_df[["starting_diameters"]]).flatten()

    if plot_type == "kde":
        if len(all_subjects_df) > 50000:
            all_subjects_sample = all_subjects_df.sample(n=50000, random_state=42)
        else:
            all_subjects_sample = all_subjects_df
        create_kde_plot(
            all_subjects_sample,
            "PC1",
            "stim_weights",
            "Engagement Index",
            "GLM Stimulus Coeff.",
            "EI vs Stim Raw Data",
            "fig5a.pdf",
            figure_dir,
            cmap,
        )
    if plot_type == "kde":
        if len(pupil_ei_stim_df) > 50000:
            pupil_sample = pupil_ei_stim_df.sample(n=50000, random_state=42)
        else:
            pupil_sample = pupil_ei_stim_df
        create_kde_plot(
            pupil_sample,
            "PC1",
            "starting_diameters",
            "Engagement Index",
            "Pupil Diameter",
            "EI vs Pupil Raw Data",
            "fig5b.pdf",
            figure_dir,
            cmap,
        )
        create_kde_plot(
            pupil_sample,
            "weightedStim",
            "starting_diameters",
            "GLM Stimulus Coeff.",
            "Pupil Diameter",
            "Stim vs Pupil Raw Data",
            "fig5c.pdf",
            figure_dir,
            cmap,
        )

    print("\n=== Summary Statistics ===")
    print(f"All subjects (EI vs Stim): {len(all_subjects_autocorr)} sessions")
    print(f"Pupil sessions: {len(pupil_autocorr)} sessions")
    ei_stim_corr, ei_stim_p = pearsonr(
        all_subjects_autocorr["ei_lag1_autocorr"],
        all_subjects_autocorr["stim_lag1_autocorr"],
    )
    ei_pupil_corr, ei_pupil_p = pearsonr(
        pupil_autocorr["ei_lag1_autocorr"],
        pupil_autocorr["pupil_lag1_autocorr"],
    )
    stim_pupil_corr, stim_pupil_p = pearsonr(
        pupil_autocorr["stim_lag1_autocorr"],
        pupil_autocorr["pupil_lag1_autocorr"],
    )
    print(f"\nAutocorrelation Correlations:")
    print(f"EI vs Stim autocorrelation: r = {ei_stim_corr:.3f}, p = {ei_stim_p:.3g}")
    print(f"EI vs Pupil autocorrelation: r = {ei_pupil_corr:.3f}, p = {ei_pupil_p:.3g}")
    print(f"Stim vs Pupil autocorrelation: r = {stim_pupil_corr:.3f}, p = {stim_pupil_p:.3g}")
    if plot_type == "kde":
        ei_stim_raw_corr, ei_stim_raw_p = pearsonr(all_subjects_df["PC1"], all_subjects_df["stim_weights"])
        print(f"\nRaw Data Correlations:")
        print(f"EI vs Stim raw data: r = {ei_stim_raw_corr:.3f}, p = {ei_stim_raw_p:.3g}")
        ei_pupil_raw_corr, _ = pearsonr(pupil_ei_stim_df["PC1"], pupil_ei_stim_df["starting_diameters"])
        stim_pupil_raw_corr, _ = pearsonr(
            pupil_ei_stim_df["weightedStim"],
            pupil_ei_stim_df["starting_diameters"],
        )
        print(f"EI vs Pupil raw data: r = {ei_pupil_raw_corr:.3f}")
        print(f"Stim vs Pupil raw data: r = {stim_pupil_raw_corr:.3f}")
    print(f"\nFigures saved to {figure_dir}/ (fig5a–fig5f.pdf)")


if __name__ == "__main__":
    main()
