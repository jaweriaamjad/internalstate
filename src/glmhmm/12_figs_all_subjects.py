# Create panels a-c of Figure 3 of Ashwood et al. (2020)
import json
import pandas as pd
import os
import sys
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
from data_utils import subjectdict_to_dataframe
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct, \
    plot_example_sessions, get_state_given_bias, plot_GLMHMM_results, \
    load_bias
from pprint import pprint
from one.api import ONE
from one.alf.io import AlfBunch

one = ONE()

# Settings
try:
    K = int(sys.argv[1])
except:
    K = 3
print("K =", K)
print("-------")

# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
results_path = str(_paths["results_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
dict_dir = join(results_path, 'glmhmmlogs', tag)
data_dir = join(save_path, 'processed_for_glmhmm', tag)
overall_dir = join(results_path, 'glmhmmlogs', tag, 'results')
figure_dir = join(results_path, 'glmhmmlogs', tag, 'figures',  'global')

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# 
try:
    sess_df = pd.read_parquet(join(data_dir, "sessions.pqt"))
except FileNotFoundError as e:
    print("pqt file doesn't exist --", e)
    # with open(join(dict_dir, 'final_subject_eid_dict.json'), 'r') as f:
    #     subject_dict = json.load(f)
    # sess_df = subjectdict_to_dataframe(subject_dict)
    # sess_df.to_parquet(join(dict_dir, "sessions.pqt"))



results_dir = join(dict_dir, 'results')

np.random.seed(41)

cv_file = join(results_dir, "cvbt_folds_model.npz")
cvbt_folds_model = load_cv_arr(cv_file)

with open(join(results_dir, "best_init_cvbt_dict.json"), 'r') as f:
    best_init_cvbt_dict = json.load(f)

# Get the file name corresponding to the best initialization
# for given K value
raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                             results_dir,
                                             best_init_cvbt_dict)
hmm_params, lls = load_glmhmm_data(raw_file)

# Also get data for subject:
inpt, y, session = load_data(join(data_dir, 'all_subjects_concat.npz'))

all_sessions, sess_loc = np.unique(session, return_index=True)
try:
    left_prob = load_bias(join(data_dir, 'all_subjects_bias_probs.npz'))
    all_trials = pd.read_parquet( join(data_dir, 'all_subjects_trials_table.pqt') )
    all_trials = AlfBunch().from_df(all_trials)
except FileNotFoundError as e:
    print("left prob file not found:", e)
    eids = [session[loc] for loc in sorted(sess_loc)]
    print(f'number of total sessions: {len(eids)}')
    for n, eid in enumerate(eids):
        trials = one.load_object(eid, "trials")
        __trials = trials.to_df()
        __trials['session'] = eid
        trials = trials.from_df(__trials)
        # assert np.unique(trials['probabilityLeft']) == [0.5], "only select unbiased trials"
        if n == 0:
            all_trials = trials
            left_prob = np.reshape(trials.probabilityLeft, (-1,1)) # [90:]
        else:
            try:
                all_trials.append(trials, inplace=True)
            except NotImplementedError as e:
                target_keys = all_trials.to_df().columns.to_list()
                source_keys = trials.to_df().columns.to_list()
                diff = list(set(target_keys) - set(source_keys))
                # if the source (dataset that needs to be appended) misses somethign..
                for key in diff:
                    # add a column to the source dataset
                    # ... first convert to a DataFrame
                    __trials = trials.to_df()
                    # ... first convert to a DataFrame
                    __trials[key] = np.nan
                    # ... then convert it back to the AlfBunch type
                    trials = trials.from_df(__trials)

                diff = list(set(source_keys) - set(target_keys))
                # if the target (dataset that needs to be appended TO) misses somethign...
                for key in diff:
                    # add a column to the target dataset
                    # ... first convert to a DataFrame
                    __trials = all_trials.to_df()
                    # ... then add the column
                    __trials[key] = np.nan
                    # ... then convert it back to the AlfBunch type
                    all_trials = all_trials.from_df(__trials)

                all_trials.append(trials, inplace=True)

            left_prob = np.vstack((
                    left_prob,
                    np.reshape(trials.probabilityLeft, (-1,1))# [90:]
                ))
        print(f"\t{n+1}/{len(eids)}", eid, \
            # "\t", 90, \
            "\t", len(np.where(session == eid)[0]))

    np.savez(join(data_dir, 'all_subjects_bias_probs.npz'), left_prob)
    all_trials.to_df().to_parquet( join(data_dir, 'all_subjects_trials_table.pqt') )

# Create mask:
# Identify violations for exclusion:
violation_idx = np.where(y == -1)[0]
nonviolation_idx, mask = create_violation_mask(violation_idx,
                                               inpt.shape[0])
y[np.where(y == -1), :] = 1
inputs, datas, train_masks = partition_data_by_session(
    np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask,
    session)

posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                         hmm_params, K, range(K))

#
# add column for glm-hmm posterior 
#
__all_trials = all_trials.to_df()
__all_trials[f'glm-hmm_{K}'] = pd.Series(list(posterior_probs))
all_trials = all_trials.from_df(__all_trials)
all_trials.to_df().to_parquet( join(data_dir, 'all_subjects_trials_table.pqt') )

#
# Import predictive accuracy
#
pred_acc_arr = load_cv_arr(join(results_dir, "predictive_accuracy_mat.npz"))
pred_acc_arr_for_plotting = pred_acc_arr.copy()

#
# Plot inferred probabilities of discrete latent variables (marginal posterior)
# and the choice / reward data for 3 selected sessions
#
sess_to_plot = sess_df.index.to_list()[-3:]
plot_example_sessions(sess_to_plot, left_prob, posterior_probs, 
        nonviolation_idx, inpt, y, session, figure_dir, "all_subjects",
        fig_name=f'example_sessions_{K}states.png'
    )

#
# Plot summary of the results of the GLM-HMM individual fit
# 
plot_GLMHMM_results(-hmm_params[2], np.exp(hmm_params[1][0]), pred_acc_arr_for_plotting, \
    None, None, figure_dir, subject='all_subjects', fig_name=f"all_subjects_{K}states.png")