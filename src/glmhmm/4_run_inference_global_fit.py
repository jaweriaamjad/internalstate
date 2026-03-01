import sys
import os
from os.path import join
from pathlib import Path
import autograd.numpy as np
from glob import glob
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
from glm_hmm_utils import (load_cluster_arr, load_session_fold_lookup,
                           load_data, create_violation_mask, launch_glm_hmm_job)

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

# Settings
K = int(sys.argv[1])
fold = int(sys.argv[2])
iter = int(sys.argv[3])

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations
global_fit = True
# perform mle => set transition_alpha to 1
transition_alpha = 1
prior_sigma = 100

# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
data_dir = join(save_path, 'processed_for_glmhmm', tag)
results_dir = join(results_path, 'glmhmmlogs', tag, 'results')

#  read in data and train/test split
subject_file = join(data_dir, 'all_subjects_concat.npz')
session_fold_lookup_table = load_session_fold_lookup(join(
    data_dir, 'all_subjects_concat_session_fold_lookup.npz'))

inpt, y, session = load_data(subject_file)
#  append a column of ones to inpt to represent the bias covariate:
inpt = np.hstack((inpt, np.ones((len(inpt),1))))
y = y.astype('int')
# Identify violations for exclusion:
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])

# Get directories with data for each fold
fold_dirs = glob(join(results_dir, 'GLM', 'fold_*'))
fold_dir = fold_dirs[fold]

print(f'Starting fold {fold+1} of {len(fold_dirs)}')

print(f'Initialization {iter+1}')
#  GLM weights to use to initialize GLM-HMM
init_param_file = join(fold_dir, 'variables_of_interest_iter_0.npz')

# create save directory for this initialization/fold combination:
save_directory = join(results_dir, 'GLM_HMM_K_' + str(K), 'fold_' + str(fold), 'iter_' + str(iter))
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

launch_glm_hmm_job(inpt,
                   y,
                   session,
                   mask,
                   session_fold_lookup_table,
                   K,
                   D,
                   C,
                   N_em_iters,
                   transition_alpha,
                   prior_sigma,
                   fold,
                   iter,
                   global_fit,
                   init_param_file,
                   save_directory)