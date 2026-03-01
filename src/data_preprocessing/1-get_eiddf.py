"""
Step 1: Get session list by tag and training status.
Outputs: 1-eids_{tag}.npy, 2-training_criteria_{tag}.pqt, 3-sessions_intrainingorlater_{tag}.pqt
Adapted from Zoe Ashwood, Guido Meijer, Alberto Pezzotta (glm-hmm).
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent.parent  # src/data_preprocessing -> src -> repo
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import numpy.random as npr
import pandas as pd
import os
from os.path import join
from pprint import pprint

import config_utils
from preprocess_utils import get_training_criteria, eids_by_tag

from one.api import ONE

one = ONE()
print("One in use:", one)
npr.seed(42)

MIN_SESSIONS = 15

# Paths from config
paths = config_utils.get_paths()
paper = config_utils.get_paper_config()
tag = paper["tag"]
save_path = paths["data_dir"]

os.makedirs(save_path, exist_ok=True)

# Get sessions by tag
eids = eids_by_tag(tag, str(save_path), one=one)
print(len(eids))

subjects = list(set([one.get_details(eid)['subject'] for eid in eids]))
pprint(len(subjects))

# Training progress per subject
print("Get training information about subjects...", end=" ")
train_df, none_subj = get_training_criteria(subjects, one=one)
train_stat = ['in_training', 'trained_1a', 'trained_1b',
              'ready4ephysrig', 'ready4delay', 'ready4recording']
train_df = train_df.loc[:, [f"date__{col}" for col in train_stat]]
train_df.to_parquet(join(save_path, f'2-training_criteria_{tag}.pqt'))
print("done.")
print("Subjects with no training information:", none_subj)
print(train_df.shape)

# Session table with training status
columns = ['subject', 'date', 'lab']
eids, details = one.search(subject=subjects, details=True)
sess_df = (pd.DataFrame({key: [info[key] for info in details] for key in columns},
                        columns=columns, index=eids)
           .sort_values(by=['subject', 'date']))
sess_df['training_status'] = None
for subj in subjects:
    for ts in train_stat:
        try:
            sess_df.loc[((sess_df.subject == subj) &
                         (sess_df.date >= train_df.loc[subj, f'date__{ts}'])),
                        'training_status'] = ts
        except KeyError:
            pass
sess_df = sess_df.drop(index=sess_df.loc[sess_df.training_status.isnull()].index)
print(sess_df.head())
print(sess_df.shape)
sess_df.to_parquet(join(save_path, f'3-sessions_intrainingorlater_{tag}.pqt'))
