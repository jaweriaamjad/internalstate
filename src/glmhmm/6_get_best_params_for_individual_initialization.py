# Save best parameters from IBL global fits (for K = 2 to 5) to initialize
# each subject's model
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from post_processing_utils import load_glmhmm_data, load_cv_arr, \
    create_cv_frame_for_plotting, get_file_name_for_best_model_fold, \
    permute_transition_matrix, calculate_state_permutation
from plotting_utils import plot_GLMHMM_results
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils

K_min = 2
K_max = 5


# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
data_dir = join(save_path, 'processed_for_glmhmm', tag)
results_dir = join(results_path, 'glmhmmlogs', tag, 'results')
fig_dir = join(results_path, 'glmhmmlogs', tag, 'figures', 'global')
save_directory = join(results_path, 'glmhmmlogs', tag, 'best_params')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

labels_for_plot = ['stim', 'pc', 'wsls', 'bias']

cv_file = join(results_dir, "cvbt_folds_model.npz")
cvbt_folds_model = load_cv_arr(cv_file)

for K in range(K_max, K_max + 1):
    print("K = " + str(K))
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for
    # given K value
    raw_file = get_file_name_for_best_model_fold(
        cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Calculate permutation
    permutation = calculate_state_permutation(hmm_params)
    print(permutation)

    # Save parameters for initializing individual fits
    weight_vectors = -hmm_params[2][permutation]
    log_transition_matrix = permute_transition_matrix(
        hmm_params[1][0], permutation)
    transition_matrix = np.exp(log_transition_matrix)
    init_state_dist = hmm_params[0][0][permutation]
    params_for_individual_initialization = [[init_state_dist],
                                            [log_transition_matrix],
                                            weight_vectors]

    np.savez(join(save_directory, 'best_params_K_' + str(K) + '.npz'),
             params_for_individual_initialization)

    plot_GLMHMM_results(weight_vectors, transition_matrix, None, \
            None, None, fig_dir, "All mice", fig_name=f"{K}_states.svg")