#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 5 2024
adapted from code by By: Guido Meijer and Alberto Pezzotta
"""

import numpy as np
import numpy.random as npr
import pandas as pd
import os
from datetime import date
from collections import defaultdict
from os.path import join, isdir
from sklearn import preprocessing

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # src/glmhmm -> repo
import config_utils
from data_utils import filter_sessions, get_training_criteria, eids_by_tag
from glm_hmm_utils import get_all_unnormalized_data_this_session, \
    fit_glm, create_train_test_sessions


from one.api import ONE
one = ONE()
print("One in use:", one)
from pprint import pprint

"""
Adapted from on Zoe Ashwood's code (https://github.com/zashwood/glm-hmm)
"""

# Settings
N_FOLDS = 5
MIN_SESSIONS = 15

npr.seed(42)

# Paths and tag from repo config
_paths = config_utils.get_paths()
save_path = str(_paths["data_dir"])
_paper = config_utils.get_paper_config()
tag = _paper.get("tag", "2023_12_bwm_release")
data_path = join(save_path, 'partially_processed')
save_path = join(save_path, 'processed_for_glmhmm', tag)

# Create folders
os.makedirs(save_path, exist_ok=True)
os.makedirs(join(save_path, 'bysubject'), exist_ok=True)
# os.makedirs(join(save_path, "allsubjects"), exist_ok=True)

# load filtered sessions (filtering is done using the code in vae folder)
sess_df = pd.read_parquet(join(data_path, f'4-sessions_behav_filtered_{tag}.pqt'))
print(f'Total number of subjects are {sess_df.subject.nunique()}')
# %%
# Identify idx in master array where each subject's data starts and ends:
subject_start_idx = {}
subject_end_idx = {}

final_subject_eid_dict = defaultdict(list)
# WORKHORSE: iterate through each subject and each subject's set of eids;
# obtain unnormalized data.  Write out each subject's data and then also
# write to master array
subject_count = 0
subjects = sess_df.subject.unique()
for z, subject in enumerate(subjects):
    print(f"{z+1}/{len(subjects)}    {subject}")
    sess_counter = 0
    eids = []
    for eid in sess_df.loc[sess_df.subject == subject].index:
        print("\t",eid)
        try:
            subject, unnormalized_inpt, y, session, num_viols, rewarded, bias_probs, to_save = \
                get_all_unnormalized_data_this_session(
                    eid, one)
            if to_save:
                if sess_counter == 0:
                    subject_unnormalized_inpt = np.copy(unnormalized_inpt)
                    subject_y = np.copy(y)
                    subject_session = session
                    subject_rewarded = np.copy(rewarded)
                    subject_bias_probs = np.copy(bias_probs)
                else:
                    subject_unnormalized_inpt = np.vstack(
                        (subject_unnormalized_inpt, unnormalized_inpt))
                    subject_y = np.vstack((subject_y, y))
                    subject_session = np.concatenate((subject_session, session))
                    subject_rewarded = np.vstack((subject_rewarded, rewarded))
                    subject_bias_probs = np.vstack((subject_bias_probs, bias_probs))
                sess_counter += 1
            eids.append(eid)
        except Exception as err:
            print("This is the error:" ,err)
    if len(eids) < MIN_SESSIONS:
        print("\tSkip subject")
        sess_df = sess_df.drop(index=sess_df.loc[sess_df.subject == subject].index)
        continue

    # Write out subject's unnormalized data matrix:
    np.savez(join(save_path, 'bysubject', subject + '_unnormalized.npz'),
             subject_unnormalized_inpt, subject_y, subject_session)
    subject_session_fold_lookup = create_train_test_sessions(subject_session, N_FOLDS)
    np.savez(join(save_path, 'bysubject', subject + "_session_fold_lookup.npz"),
             subject_session_fold_lookup)
    np.savez(join(save_path, 'bysubject', subject + '_rewarded.npz'),
             subject_rewarded)
    np.savez(join(save_path, 'bysubject', subject + '_bias_probs.npz'),
             subject_bias_probs)
    assert subject_rewarded.shape[0] == subject_y.shape[0]
    # Now create or append data to master array across all subjects:
    if subject_count == 0:
        master_inpt = np.copy(subject_unnormalized_inpt)
        subject_start_idx[subject] = 0
        subject_end_idx[subject] = master_inpt.shape[0] - 1
        master_y = np.copy(subject_y)
        master_session = subject_session
        master_session_fold_lookup_table = subject_session_fold_lookup
        master_rewarded = np.copy(subject_rewarded)
    else:
        subject_start_idx[subject] = master_inpt.shape[0]
        master_inpt = np.vstack((master_inpt, subject_unnormalized_inpt))
        subject_end_idx[subject] = master_inpt.shape[0] - 1
        master_y = np.vstack((master_y, subject_y))
        master_session = np.concatenate((master_session, subject_session))
        master_session_fold_lookup_table = np.vstack(
            (master_session_fold_lookup_table, subject_session_fold_lookup))
        master_rewarded = np.vstack((master_rewarded, subject_rewarded))
    subject_count += 1

sess_df.to_parquet(join(save_path, "sessions.pqt"))

sess_df = pd.read_parquet(join(data_path, f'3-sessions_intrainingorlater_{tag}.pqt'))
init_subjects = sess_df.subject.unique()
init_sessions = sess_df.index.to_numpy()
print(f"Initial dataset\n\t{len(init_subjects)} subjects, {len(init_sessions)} sessions")

sess_df = pd.read_parquet(join(save_path, 'sessions.pqt'))
final_subjects = sess_df.subject.unique()
final_sessions = sess_df.index.to_numpy()
print(f"Final dataset\n\t{len(final_subjects)} subjects, {len(final_sessions)} sessions")

# Write out data from across subjects
assert np.shape(master_inpt)[0] == np.shape(master_y)[
    0], "inpt and y not same length"
assert np.shape(master_rewarded)[0] == np.shape(master_y)[
    0], "rewarded and y not same length"
assert len(np.unique(master_session)) == \
       np.shape(master_session_fold_lookup_table)[
           0], "number of unique sessions and session fold lookup don't " \
               "match"
normalized_inpt = np.copy(master_inpt)
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])
np.savez(join(save_path, 'all_subjects_concat.npz'),
         normalized_inpt, master_y, master_session)
np.savez(join(save_path, 'all_subjects_concat_unnormalized.npz'),
              master_inpt, master_y, master_session)
np.savez(join(save_path, 'all_subjects_concat_session_fold_lookup.npz'),
         master_session_fold_lookup_table)
np.savez(join(save_path, 'all_subjects_concat_rewarded.npz'),
         master_rewarded)
np.savez(join(save_path, 'bysubject', 'subject_list.npz'),
         final_subjects)

# Now write out normalized data (when normalized across all subjects) for
# each subject:
counter = 0
for subject in subject_start_idx.keys():
    start_idx = subject_start_idx[subject]
    end_idx = subject_end_idx[subject]
    inpt = normalized_inpt[range(start_idx, end_idx + 1)]
    y = master_y[range(start_idx, end_idx + 1)]
    session = master_session[range(start_idx, end_idx + 1)]
    counter += inpt.shape[0]
    np.savez(join(save_path, 'bysubject', subject + '_processed.npz'),
             inpt, y, session)

assert counter == master_inpt.shape[0]