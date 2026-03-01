# Fit GLM to each IBL subject separately
import autograd.numpy as np
import autograd.numpy.random as npr
import os
from os.path import join, isfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
from glm_hmm_utils import (load_session_fold_lookup, load_data, load_subject_list,
                           fit_glm, append_zeros)

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

npr.seed(42)

C = 2  # number of output types/categories
N_initializations = 10

_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
data_dir = join(save_path, 'processed_for_glmhmm', tag, 'bysubject')
results_dir = join(results_path, 'glmhmmlogs', tag, 'results', 'individual_fit')
subject_list = load_subject_list(join(data_dir, 'subject_list.npz'))
num_folds = 5

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for subject in subject_list:
    # Fit GLM to data from single subject:
    subject_file = join(data_dir, subject + '_processed.npz')
    session_fold_lookup_table = load_session_fold_lookup(
        join(data_dir, subject + '_session_fold_lookup.npz'))

    for fold in range(num_folds):
        this_results_dir = join(results_dir, subject)

        # Load data
        inpt, y, session = load_data(subject_file)
        labels_for_plot = ['stim', 'pc', 'wsls', 'bias']
        y = y.astype('int')

        figure_directory = join(this_results_dir, 'GLM', 'fold_' + str(fold))
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        # Subset to sessions of interest for fold
        sessions_to_keep = session_fold_lookup_table[np.where(
            session_fold_lookup_table[:, 1] != fold), 0]
        idx_this_fold = [
            str(sess) in sessions_to_keep and y[id, 0] != -1
            for id, sess in enumerate(session)
        ]
        this_inpt, this_y, this_session = inpt[idx_this_fold, :], \
                                          y[idx_this_fold, :], \
                                          session[idx_this_fold]
        assert len(
            np.unique(this_y)
        ) == 2, "choice vector should only include 2 possible values"
        train_size = this_inpt.shape[0]

        M = this_inpt.shape[1]
        loglikelihood_train_vector = []

        for iter in range(N_initializations):
            if isfile(join(figure_directory, 'variables_of_interest_iter_' + str(iter) + '.npz')):
                continue
            loglikelihood_train, recovered_weights = fit_glm([this_inpt],
                                                             [this_y], M,
                                                             C)
            weights_for_plotting = append_zeros(recovered_weights)
            loglikelihood_train_vector.append(loglikelihood_train)
            np.savez(join(figure_directory, 'variables_of_interest_iter_' + str(iter) + '.npz'),
                     loglikelihood_train, recovered_weights)