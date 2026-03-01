"""
Step 2: Filter sessions by behavioral QC (lapse, bias, min_trials, wheel data).
Reads: 3-sessions_intrainingorlater_{tag}.pqt
Outputs: 4-sessions_behav_filtered_{tag}.pqt (used by VAE and GLM-HMM).
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
from os.path import join

import config_utils
from preprocess_utils import load_trials, calc_trialwise_wheel, get_qc_trials

from one.api import ONE

one = ONE()
print("One in use:", one)
npr.seed(42)

# Paths from config
paths = config_utils.get_paths()
paper = config_utils.get_paper_config()
tag = paper["tag"]
save_path = paths["data_dir"]

sess_df = pd.read_parquet(join(save_path, f'3-sessions_intrainingorlater_{tag}.pqt'))
eids = sess_df.index.to_numpy()

max_lapse = 0.5
max_bias = 0.5
min_trials = 200
use_eids = []

for n, eid in enumerate(eids):
    print(f'{n:04d}/{len(eids)}')
    try:
        info = one.eid2ref(eid)
        subject = info['subject']
        print(n, subject + ":" + eid)
        trials = load_trials(eid, one=one)
        contrast = trials.signed_contrast.unique()
        probabilityLeft = trials.probabilityLeft.unique()
        position, timestamps = calc_trialwise_wheel(eid, one=one)
        n_trials = len(position)
    except Exception as e:
        print(e)
        print('Could not load session data %s' % eid)
        continue

    number_of_good_trials, _ = get_qc_trials(position)
    if (len(number_of_good_trials) > (n_trials - 10)):
        print(f"position criteria satisfied! total trials: {n_trials}, empty position trials: {n_trials - len(number_of_good_trials)}")
        num_trials = len(contrast)
        comparison_probL = np.array_equal(np.sort(probabilityLeft), np.sort(np.array([0.2, 0.5, 0.8])))
        comparison_contrast = np.array_equal(np.sort(contrast), np.sort(np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])))
        if isinstance(comparison_probL, np.ndarray):
            comparison_probL = comparison_probL.all()
        if isinstance(comparison_contrast, np.ndarray):
            comparison_contrast = comparison_contrast.all()
        if comparison_probL and comparison_contrast:
            print("contrast and bias probability satisfied")
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias) & comparison_probL):
                use_eids.append(eid)
                print("behavioural criteria satisfied: appending session to eid list!")
            else:
                print('lapse rate not satisfied')
    else:
        print("wheel data missing for more than 10 trials in the session")

print(len(use_eids), "out of", len(eids))

sess_df = sess_df.loc[use_eids, :]
sess_df.to_parquet(join(save_path, f'4-sessions_behav_filtered_{tag}.pqt'))
