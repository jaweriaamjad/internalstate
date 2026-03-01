# Calculate predictive accuracy for each individual subject (for Figure 2c
# and 4b).  Note: this is the same code for both figures, and only needs to
# be run once to calculate the quantities required to generate both figures
import json
import sys
import numpy as np
from os.path import join
import numpy.random as npr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
from plotting_utils import load_glmhmm_data, load_subject_list, load_cv_arr, \
    load_data, load_glm_vectors, load_lapse_params, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, create_train_test_trials_for_pred_acc, \
    calculate_predictive_accuracy, calculate_predictive_acc_glm, \
    calculate_predictive_acc_lapse_model

N_FOLDS = 5
MAX_K = 5
cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]


# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
dict_dir = join(results_path, 'glmhmmlogs')
data_dir = join(save_path, 'processed_for_glmhmm', tag)
results_dir = join(results_path, 'glmhmmlogs', 'results')

sigma_val = 2
alpha_val = 2
npr.seed(41)

allsubject_params_dict = dict()

cv_file = join(results_dir, "cvbt_folds_model.npz")
cvbt_folds_model = load_cv_arr(cv_file)

predictive_acc_mat = []
num_trials_mat = []

# Also get data for subject:
inpt, y, session = load_data(join(data_dir, 'all_subjects_concat.npz'))

# create train test idx
trial_fold_lookup_table = create_train_test_trials_for_pred_acc(y, num_folds=N_FOLDS)

# GLM fit:
# Load params:
_, glm_weights = load_glm_vectors(join(results_dir, 'GLM', 'fold_0',
                                       'variables_of_interest_iter_0.npz'))
predictive_acc_glm = []
for fold in range(N_FOLDS):
    # identify the idx for exclusion:
    idx_to_exclude = trial_fold_lookup_table[np.where(
        trial_fold_lookup_table[:, 1] == fold)[0], 0].astype('int')
    predictive_acc = calculate_predictive_acc_glm(
        glm_weights, inpt, y, idx_to_exclude)
    predictive_acc_glm.append(predictive_acc)
    num_trials_mat.append(len(idx_to_exclude))
predictive_acc_mat.append(predictive_acc_glm)

for K in range(2, MAX_K+1):
    print(f"K---------")
    with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
        best_init_cvbt_dict = json.load(f)

        # Get the file name corresponding to the best initialization
        # for given K value
        raw_file = get_file_name_for_best_model_fold(
            cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        # Save parameters for initializing individual fits
        weight_vectors = hmm_params[2].reshape(K,-1).T
        log_transition_matrix = hmm_params[1][0]
        init_state_dist = hmm_params[0][0]

        allsubject_params_dict[K] = dict(
            weight_vectors = dict(
                    [(key, val) for key, val in zip(["stim", "pc", "wsls", "bias"], weight_vectors.tolist())]
                ),
            log_transition_matrix = log_transition_matrix.tolist(),
        )

        predictive_acc_this_K = []
        for fold in range(N_FOLDS):
            print(f"fold {fold}")
            # identify the idx for exclusion:
            idx_to_exclude = trial_fold_lookup_table[np.where(
                trial_fold_lookup_table[:, 1] == fold)[0],0].astype('int')
            # Make a copy of y and modify idx
            y_modified = np.copy(y)
            y_modified[idx_to_exclude] = -1
            # Create mask:
            # Identify violations for exclusion:
            violation_idx = np.where(y_modified == -1)[0]
            nonviolation_idx, mask = create_violation_mask(
                violation_idx, inpt.shape[0])
            y_modified[np.where(y_modified == -1), :] = 1
            inputs, datas, train_masks = partition_data_by_session(
                np.hstack((inpt, np.ones((len(inpt), 1)))), y_modified,
                mask, session)
            predictive_acc = calculate_predictive_accuracy(
                inputs, datas, train_masks, hmm_params, K, range(K),
                alpha_val, sigma_val, y, idx_to_exclude)
            predictive_acc_this_K.append(predictive_acc)
        allsubject_params_dict[K]['predictive_acc'] = \
                predictive_acc_this_K
        predictive_acc_mat.append(predictive_acc_this_K)
predictive_acc_mat = np.array(predictive_acc_mat)
np.savez(join(results_dir, "predictive_accuracy_mat.npz"),
         np.array(predictive_acc_mat))
np.savez(join(results_dir, "correct_incorrect_mat.npz"),
         np.array(predictive_acc_mat * num_trials_mat), num_trials_mat)

with open(join(dict_dir, 'allsubject_results_dict.json'), 'w') as f:
    f.write(json.dumps(allsubject_params_dict))