"""
Helpers for data_preprocessing scripts.
Adapted from behavior/vae/utils/datapreprocess_utils (originally Alberto Pezzotta et al.).
Requires: one.api.ONE, pandas, numpy.
"""

import numpy as np
import pandas as pd
from os.path import join


def eids_by_tag(tag, path, one=None):
    """
    Load or compute session eids for the given tag.
    Caches to path/1-eids_{tag}.npy.
    """
    if one is None:
        from one.api import ONE
        one = ONE()
    f_name = join(path, f"1-eids_{tag}.npy")
    try:
        eids = np.load(f_name, allow_pickle=True)
    except Exception:
        one.load_cache(tag=tag)
        eids = one.search()
        np.save(f_name, eids)
    return eids


def get_training_criteria(subjects, one=None):
    """Get training status dates per subject from Alyx."""
    if one is None:
        from one.api import ONE
        one = ONE()
    list_tr = []
    none_subj = []
    for subj in subjects:
        tr_cr = one.alyx.rest('subjects', 'read', id=subj)['json'].get('trained_criteria')
        if tr_cr is None:
            none_subj.append(subj)
        else:
            tr_cr['subject'] = subj
            list_tr.append(tr_cr)
    df = pd.DataFrame.from_dict(list_tr)
    for col in df.columns:
        if col == 'subject':
            continue
        df[col].loc[df[col].isnull()] = df[col].loc[df[col].isnull()].apply(lambda x: list())
        date_col = f'date__{col}'
        status_col = f'eid__{col}'
        df[[date_col, status_col]] = pd.DataFrame(df[col].tolist())
        df = df.drop(columns=[col, status_col])
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df.set_index('subject'), none_subj


def load_trials(eid, one=None):
    """Load trials table for one session."""
    if one is None:
        from one.api import ONE
        one = ONE()
    attributes = [
        'stimOn_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice',
    ]
    data = one.load_object(eid, 'trials')
    data = {k: data[k] for k in attributes}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return trials
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    return trials


def get_qc_trials(position):
    """Return indices of trials for which wheel position arrays have more than one entry."""
    qc_trials = [i for i in range(len(position)) if (len(position[i]) > 1)]
    rej_trials = [i for i in range(len(position)) if (len(position[i]) <= 1)]
    return qc_trials, rej_trials


def calc_trialwise_wheel(eid, one=None):
    """Split wheel data into per-trial arrays (stimOn - 0.3s to feedback)."""
    if one is None:
        from one.api import ONE
        one = ONE()
    wheel = one.load_object(eid, 'wheel', collection='alf')
    trial = one.load_object(eid, 'trials', collection='alf')
    stimOn_times = trial.stimOn_times
    feedback_times = trial.feedback_times
    stimOn_pre_duration = 0.3
    total_trial_count = len(stimOn_times)
    trial_position = [[] for _ in range(total_trial_count)]
    trial_timestamps = [[] for _ in range(total_trial_count)]
    tridx = 0
    for tsidx in range(len(wheel.timestamps)):
        timestamp = wheel.timestamps[tsidx]
        while tridx < total_trial_count - 1 and timestamp > stimOn_times[tridx + 1] - stimOn_pre_duration:
            tridx += 1
        if stimOn_times[tridx] - stimOn_pre_duration <= timestamp and timestamp < feedback_times[tridx]:
            trial_position[tridx].append(wheel.position[tsidx])
            trial_timestamps[tridx].append(wheel.timestamps[tsidx])
    return trial_position, trial_timestamps
