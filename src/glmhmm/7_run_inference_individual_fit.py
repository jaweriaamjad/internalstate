#  Fit GLM-HMM to data from all IBL subjects together.  These fits will be
#  used to initialize the models for individual subjects
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
import autograd.numpy as np
from os.path import join
from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
        load_subject_list, load_data, create_violation_mask, \
        launch_glm_hmm_job

# Settings
K = int(sys.argv[1])
fold = int(sys.argv[2])
iter = int(sys.argv[3])

prior_sigma = 2
transition_alpha = 2
D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations
z = 0
global_fit = False

# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
global_data_dir = join(save_path, 'processed_for_glmhmm', tag)
data_dir = join(global_data_dir, 'bysubject')
global_results_dir = join(results_path, 'glmhmmlogs', tag, 'results')
results_dir = join(global_results_dir, 'individual_fit')


subject_list = load_subject_list(join(data_dir, 'subject_list.npz'))
for i, subject in enumerate(subject_list):
    print(subject)
    subject_file = join(data_dir, subject + '_processed.npz')
    session_fold_lookup_table = load_session_fold_lookup(join(
        data_dir, subject + '_session_fold_lookup.npz'))

    inpt, y, session = load_data(subject_file)
    #  append a column of ones to inpt to represent the bias covariate:
    inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
    y = y.astype('int')

    overall_dir = join(results_dir, subject)

    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])

    init_param_file = join(global_results_dir, 'best_params', 'best_params_K_' + str(K) + '.npz')
    save_directory = join(overall_dir, 'GLM_HMM_K_' + str(K), 'fold_' + str(fold), 'iter_' + str(iter))

    if os.path.exists(save_directory):
        continue
    else:
        os.makedirs(save_directory)
        launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table,
                           K, D, C, N_em_iters, transition_alpha, prior_sigma,
                           fold, iter, global_fit, init_param_file,
                           save_directory)