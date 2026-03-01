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
from collections import defaultdict
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
all_data_dir = join(save_path, 'processed_for_glmhmm', tag)
data_dir = join(save_path, 'processed_for_glmhmm', tag, 'bysubject')
overall_dir = join(results_path, 'glmhmmlogs', tag, 'results','individual_fit')
# 
try:
    sess_df = pd.read_parquet(join(all_data_dir, "sessions.pqt"))
except FileNotFoundError as e:
    print("pqt file doesn't exist --", e)
    with open(join(dict_dir, 'final_subject_eid_dict.json'), 'r') as f:
        subject_dict = json.load(f)
    sess_df = subjectdict_to_dataframe(subject_dict)
    sess_df.to_parquet(join(dict_dir, "sessions.pqt"))
try:
    df = pd.read_parquet(join(dict_dir,'results', f'indivfit_{K}statesglmweights.pqt'))
    print(df.head())
except:
    subjects = list(sess_df.subject.unique())
    state_weights = defaultdict(list)
    for i, subject in enumerate(subjects):

        print(f"{i+1:03d}/{len(subjects)} --> {subject}")

        results_dir = join(dict_dir, 'results', 'individual_fit', subject)

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
        weight_vectors = -hmm_params[2].squeeze()
        state_weights['subject'].extend([subject] * K)
        state_weights['stimulus'].extend(weight_vectors[:, 0].tolist())
        state_weights['bias'].extend(weight_vectors[:, 3].tolist())
        state_weights['prev_choice'].extend(weight_vectors[:, 1].tolist())
        state_weights['prev_reward'].extend(weight_vectors[:, 2].tolist())
    df=pd.DataFrame(state_weights)
    df.to_parquet(join(dict_dir,'results', f'indivfit_{K}statesglmweights.pqt'))
    print(df.head())
    print(df.shape)