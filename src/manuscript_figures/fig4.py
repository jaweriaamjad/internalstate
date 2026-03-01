# Figure 4: GLM-HMM — weights (a), transition matrix (b), psychometric curves (c), accuracy by state (d)
import json
import os
import sys
from pathlib import Path
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))
import config_utils
from glmhmm.plotting_utils import (
    load_glmhmm_data,
    load_cv_arr,
    load_data,
    get_file_name_for_best_model_fold,
    partition_data_by_session,
    create_violation_mask,
    get_marginal_posterior,
    get_prob_right,
)

_paths = config_utils.get_paths()
_paper = config_utils.get_paper_config()
data_dir = str(_paths["data_dir"])
results_dir = str(_paths["results_dir"])
figure_dir = str(_paths["figures_dir"])
tag = _paper.get("tag", "2023_12_bwm_release")
K = _paper.get("glmhmm_K", 3)

os.makedirs(figure_dir, exist_ok=True)
glmhmm_datadir = join(data_dir, "glmhmm", "bysubject")
glmhmm_resultsdir = join(results_dir, "glmhmm_logs")

cols = [
    "#ff7f00", "#4daf4a", "#377eb8", "#f781bf", "#a65628", "#984ea3",
    "#999999", "#e41a1c", "#dede00",
]

if __name__ == "__main__":
    animal = "ibl_witten_32"

    inpt, old_y, session = load_data(join(glmhmm_datadir, animal + "_processed.npz"))
    unnormalized_inpt, _, _ = load_data(join(glmhmm_datadir, animal + "_unnormalized.npz"))
    y = np.copy(old_y)

    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    old_y[np.where(old_y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), old_y, mask, session
    )
    inpt = inpt[nonviolation_idx, :]
    y = y[nonviolation_idx, :]
    unnormalized_inpt = unnormalized_inpt[nonviolation_idx, :]

    # Overall accuracy
    accuracies_to_plot = []
    not_zero_loc = np.where(unnormalized_inpt[:, 0] != 0)[0]
    correct_ans = (np.sign(unnormalized_inpt[not_zero_loc, 0]) + 1) / 2
    acc = np.sum(y[not_zero_loc, 0] == correct_ans) / len(correct_ans)
    accuracies_to_plot.append(acc)

    cv_file = join(glmhmm_resultsdir, "cvbt_folds_model.npz")
    cvbt_folds_model = load_cv_arr(cv_file)
    with open(join(glmhmm_resultsdir, "best_init_cvbt_dict.json")) as f:
        best_init_cvbt_dict = json.load(f)

    raw_file = get_file_name_for_best_model_fold(
        cvbt_folds_model, K, glmhmm_resultsdir, best_init_cvbt_dict
    )
    hmm_params, lls = load_glmhmm_data(raw_file)
    posterior_probs = get_marginal_posterior(inputs, datas, masks, hmm_params, K, range(K))
    posterior_probs = posterior_probs[nonviolation_idx, :]

    for k in range(K):
        idx_of_interest = np.where(posterior_probs[:, k] >= 0.9)[0]
        inpt_this = inpt[idx_of_interest, :]
        unnorm_this = unnormalized_inpt[idx_of_interest, :]
        y_this = y[idx_of_interest, :]
        not_zero_loc = np.where(unnorm_this[:, 0] != 0)[0]
        correct_ans = (np.sign(unnorm_this[not_zero_loc, 0]) + 1) / 2
        acc = np.sum(y_this[not_zero_loc, 0] == correct_ans) / len(correct_ans)
        accuracies_to_plot.append(acc)

    weight_vectors = -hmm_params[2]  # negate to match paper (fig2a)
    transition_matrix = np.exp(hmm_params[1][0])

    # ---------- Fig 4a: GLM weights ----------
    fig = plt.figure(figsize=(2.7, 2.5))
    plt.subplots_adjust(left=0.3, bottom=0.4, right=0.8, top=0.9)
    M = weight_vectors.shape[2] - 1
    for k in range(K):
        plt.plot(
            range(M + 1),
            weight_vectors[k][0][[0, 3, 1, 2]],
            marker="o",
            label="state " + str(k + 1),
            color=cols[k],
            lw=1,
            alpha=0.7,
        )
    plt.yticks([-2.5, 0, 2.5, 5], fontsize=10)
    plt.xticks(
        [0, 1, 2, 3],
        ["contrast", "bias", "prev. \nchoice", "win-stay-\nlose-switch"],
        fontsize=10,
        rotation=45,
    )
    plt.ylabel(r"$\mathbf{w}_i$", fontsize=12)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--", lw=0.5)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    fig.savefig(join(figure_dir, "fig4a.pdf"))
    plt.close(fig)

    # ---------- Fig 4b: Transition matrix ----------
    fig = plt.figure(figsize=(1.6, 1.6))
    plt.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.95)
    plt.imshow(transition_matrix, vmin=-0.8, vmax=1, cmap="bone")
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            plt.text(
                j, i,
                str(np.around(transition_matrix[i, j], decimals=2)),
                ha="center", va="center", color="k", fontsize=10,
            )
    plt.xlim(-0.5, K - 0.5)
    plt.xticks(range(0, K), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")[:K], fontsize=10)
    plt.yticks(range(0, K), ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")[:K], fontsize=10)
    plt.ylim(K - 0.5, -0.5)
    plt.ylabel("state t-1", fontsize=10)
    plt.xlabel("state t", fontsize=10)
    fig.savefig(join(figure_dir, "fig4b.pdf"))
    plt.close(fig)

    # ---------- Fig 4c: Psychometric curves ----------
    inpt, y, session = load_data(join(glmhmm_datadir, animal + "_processed.npz"))
    unnormalized_inpt, _, _ = load_data(join(glmhmm_datadir, animal + "_unnormalized.npz"))
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session
    )
    raw_file = get_file_name_for_best_model_fold(
        cvbt_folds_model, K, glmhmm_resultsdir, best_init_cvbt_dict
    )
    hmm_params, _ = load_glmhmm_data(raw_file)
    weight_vectors = hmm_params[2]
    posterior_probs = get_marginal_posterior(inputs, datas, masks, hmm_params, K, range(K))

    fig = plt.figure(figsize=(4.6, 2), dpi=80, facecolor="w", edgecolor="k")
    plt.subplots_adjust(left=0.13, bottom=0.23, right=0.9, top=0.8)
    for k in range(K):
        plt.subplot(1, 3, k + 1)
        stim_vals, prob_right_max = get_prob_right(-weight_vectors, inpt, k, 1, 1)
        _, prob_right_min = get_prob_right(-weight_vectors, inpt, k, -1, -1)
        plt.plot(stim_vals, prob_right_max, "-", color=cols[k], alpha=1, lw=1, zorder=5)
        plt.plot(stim_vals, get_prob_right(-weight_vectors, inpt, k, -1, 1)[1], "--", color=cols[k], alpha=0.5, lw=1)
        plt.plot(stim_vals, get_prob_right(-weight_vectors, inpt, k, 1, -1)[1], "-", color=cols[k], alpha=0.5, lw=1)
        plt.plot(stim_vals, prob_right_min, "--", color=cols[k], alpha=1, lw=1)
        plt.xticks([min(stim_vals), 0, max(stim_vals)], labels=["", "", ""], fontsize=10)
        plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
        plt.ylabel("")
        plt.xlabel("")
        if k == 0:
            plt.xticks([min(stim_vals), 0, max(stim_vals)], labels=["-1", "0", "1"], fontsize=10)
            plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
            plt.ylabel('p("R")', fontsize=12)
            plt.xlabel("contrast", fontsize=12)
        plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.ylim((-0.01, 1.01))
    fig.savefig(join(figure_dir, "fig4c.pdf"))
    plt.close(fig)

    # ---------- Fig 4d: Accuracy by state ----------
    fig = plt.figure(figsize=(1.3, 1.7))
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.95)
    for z, acc in enumerate(accuracies_to_plot):
        col = "grey" if z == 0 else cols[z - 1]
        plt.bar(z, acc * 100, width=0.8, color=col)
    plt.ylim((50, 100))
    plt.xticks([0, 1, 2, 3], ["All", "1", "2", "3"], fontsize=10)
    plt.yticks([50, 75, 100], fontsize=10)
    plt.xlabel("state", fontsize=10)
    plt.ylabel("fraction correct", fontsize=10, labelpad=-0.5)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    fig.savefig(join(figure_dir, "fig4d.pdf"))
    plt.close(fig)

    print("Saved fig4a.pdf, fig4b.pdf, fig4c.pdf, fig4d.pdf to", figure_dir)
