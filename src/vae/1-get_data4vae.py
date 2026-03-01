"""
Step 1 (VAE): Build VAE dataset from filtered sessions.
Reads: data_dir/4-sessions_behav_filtered_{tag}.pqt
Writes: data_dir/processed_for_vae/{tag}/ (bysubject, allsubjects, session_emptytrials_dict, etc.)
"""

import sys
import time
import json
import os
import warnings
from pathlib import Path
from os.path import join
from collections import defaultdict

import numpy as np
import pandas as pd

_VAE_DIR = Path(__file__).resolve().parent
_REPO = _VAE_DIR.parent.parent  # src/vae -> src -> repo
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_VAE_DIR))

import config_utils
from vae_utils import get_all_data_this_session

from one.api import ONE

warnings.filterwarnings("ignore")
one = ONE()
print("One in use:", one)
np.random.seed(42)

MIN_SESSIONS = 15

paths = config_utils.get_paths()
paper = config_utils.get_paper_config()
tag = paper["tag"]
data_dir = paths["data_dir"]
data_path = data_dir  # 4-sessions_behav_filtered lives here
save_path = join(data_dir, "processed_for_vae", tag)

os.makedirs(save_path, exist_ok=True)
os.makedirs(join(save_path, "bysubject"), exist_ok=True)
os.makedirs(join(save_path, "allsubjects"), exist_ok=True)

sess_df = pd.read_parquet(join(data_path, f'4-sessions_behav_filtered_{tag}.pqt'))
allsubjects_df = pd.DataFrame()
session_emptytrials_dict = defaultdict(list)
final_subject_eiddict = defaultdict(list)

subjects = list(sess_df.subject.unique())
st = time.time()

for z, subject in enumerate(subjects):
    print(f"{z+1}/{len(subjects)}    {subject}")
    subject_df = pd.DataFrame()
    sess_counter = 0
    eids = []
    session_et = defaultdict(list)
    for eid in sess_df.loc[sess_df.subject == subject].index:
        print("\t", eid)
        try:
            session_data, number_of_trials, rej_trials, to_save = get_all_data_this_session(eid, one=one)
            if to_save:
                df = pd.DataFrame(session_data).transpose()
                subject_df = pd.concat([subject_df, df], ignore_index=True)
                session_et[eid] = rej_trials
                sess_counter += 1
            eids.append(eid)
        except Exception as err:
            print("This is the error:", err)
    subject_df.columns = ["subject", "eid", "position", "bias_probs", "contrast", "choice", "feedback",
                          "stimOn_time", "goCue_time", "FMOT_time", "feedback_time", "max_pos", "duration"]
    if len(eids) < MIN_SESSIONS:
        print("\tSkip subject")
        sess_df = sess_df.drop(index=sess_df.loc[sess_df.subject == subject].index)
        continue
    final_subject_eiddict[subject] = eids
    session_emptytrials_dict.update(session_et)
    allsubjects_df = pd.concat([allsubjects_df, subject_df], ignore_index=True)
    subject_df.to_parquet(join(save_path, "bysubject", f"{subject}.pqt"))

with open(join(save_path, "session_emptytrials_dict.json"), "w") as f:
    f.write(json.dumps(session_emptytrials_dict))

allsubjects_df.to_parquet(join(save_path, "allsubjects", "allsubjects.pqt"))
with open(join(save_path, "final_subject_eiddict.json"), "w") as f:
    f.write(json.dumps(final_subject_eiddict))

print(f"{len(allsubjects_df.subject.unique())} subjects, {len(allsubjects_df.eid.unique())} sessions.")
print("Execution time:", time.time() - st, "seconds")
