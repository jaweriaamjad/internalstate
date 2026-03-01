import json
import numpy as np
import pandas as pd
from collections import defaultdict
from os.path import join, realpath, dirname, isfile
from one.api import ONE
from pprint import pprint

def paths(dropbox=False):
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        paths = dict()
        paths['fig_path'] = input('Path folder to save figures: ')
        paths['save_path'] = input('Path folder to save data: ')
        paths['dropbox_path'] = input('Path to Dropbox folder (can be empty):')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(paths, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        paths = json.load(json_file)
    if dropbox:
        fig_path = paths['dropbox_path']
    else:
        fig_path = paths['fig_path']
    save_path = paths['save_path']
    #save_path = join(dirname(realpath(__file__)), 'Data')
    return fig_path, save_path

def eids_by_tag(tag, path="."):

    f_name = join( path, f"eids_{tag}.npy" )

    try:
        eids = np.load(f_name)
    except:
        one = ONE()
        one.load_cache(tag=tag)
        eids = one.search()
        np.save(f_name, eids)

    return eids


def get_session_ids (eids, one=None):
    '''
    get a list of eids, and return the corresponding list
    of session_id in the format used by this code to 
    identify sessions (saved in '<subject>_processed.npz' files)
    '''
    if one is None:
        one = ONE()
    session_ids = []
    for eid in eids:
        assert isinstance(eid, str), "'eid' must be in string format!"
        info = one.get_details(eid)
        session_id = f"{info['subject']}-{info['date']}-{info['number']:03d}"
        session_ids.append(session_id)
    return session_ids
    

def load_trials(eid, one=None):
    if one is None:
        one = ONE()

    attributes = [
        'stimOn_times', 'goCue_times', 'probabilityLeft', 'contrastLeft',
        'contrastRight', 'feedbackType', 'choice',
        # 'firstMovement_times',
        ]

    data = one.load_object(eid, 'trials') #, attribute=attributes, collection='alf')
    data = {your_key: data[your_key] for your_key in attributes}
    trials = pd.DataFrame(data=data)
    if trials.shape[0] == 0:
        return
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0

    return trials

# def load_subject_trials (subject, one=None, **search_kwargs):
#     if one is None:
#         one = ONE()

#     trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')
#     # Load training status and join to trials table
#     training = one.load_aggregate('subjects', subject, '_ibl_subjectTraining.table')
#     trials = (trials
#               .set_index('session')
#               .join(training.set_index('session'))
#               .sort_values(by=['session_start_time', 'intervals_0']))
#     trials['training_status'] = trials.training_status.fillna(method='ffill')

#     # add "signed_contrast" column
#     trials.contrastLeft.fillna(0, inplace=True)
#     trials.contrastRight.fillna(0, inplace=True)
#     trials["signed_contrast"] = trials.contrastRight - trials.contrastLeft

#     return trials.loc[(trials.task_protocol.str.contains('biasedChoiceWorld')) \
#             | (trials.task_protocol.str.contains('ephysChoiceWorld'))]

# By: Gaelle Chapuis
def get_training_criteria (subjects, one=None):
    if one is None:
        one = ONE()

    list_tr = list()
    none_subj = []
    for subj in subjects:
        tr_cr = one.alyx.rest('subjects', 'read', id=subj)['json'].get('trained_criteria')
        if tr_cr is None:
            none_subj.append(subj)
        else:
            tr_cr['subject'] = subj
            list_tr.append(tr_cr)

    df = pd.DataFrame.from_dict(list_tr)

    # Split columns into date / EID
    for col in df.columns:
        if col == 'subject':
            continue
        # Reformat nan so they are in list for later split
        df[col].loc[df[col].isnull()] = df[col].loc[df[col].isnull()].apply(lambda x: list())
        # New column names
        date_col = f'date__{col}'
        status_col = f'eid__{col}'
        # # Split the list into two columns ; the first element is the date, the second the eid
        df[[date_col, status_col]] = pd.DataFrame(df[col].tolist())
        # Drop the original column
        df = df.drop(columns=[col, status_col])
        # Convert str date to datetime
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

    return df.set_index('subject'), none_subj

def search_sessions (one=None, **rest_kwargs):
    if one is None:
        one = ONE()

    sessions = one.alyx.rest('sessions', 'list', **rest_kwargs)

    subject_eid_dict = dict()
    for i, sess in enumerate(sessions):
        subject = sess['subject']
        eid = sess['id']
        try:
            if eid not in subject_eid_dict[subject]:
                subject_eid_dict[subject].append(eid)
        except KeyError:
            subject_eid_dict[subject] = [eid]
    return subject_eid_dict


def filter_sessions(eids, max_lapse=0.5, max_bias=0.5, \
        min_trials=200, return_excluded=False, p_left=[0.2,0.5,0.8],
        one=None):

    if one is None:
        one = ONE()
    use_eids, excl_eids = [], []
    for j, eid in enumerate(eids):
        print(f'{j:04d}/{len(eids)}')
        try:
            # load the trials table for this session
            trials = load_trials(eid, one=one)

            left_probs = trials.probabilityLeft.unique()
            p_left_check = True # np.array_equal(left_probs, np.array(p_left))

            # check laps rate and bias over the trials of interest
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                 / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))

            # check the relative frequency of violations in the trials of interest
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias) & p_left_check):
                use_eids.append(eid)
            else:
                details = one.get_details(eid)
                print(f"{details['subject']} {details['start_time'][:10]} excluded "+\
                      f"(n_trials: {trials.shape[0]}, lapse_l: {lapse_l:%.2f}, "+\
                      f"lapse_r: {lapse_r:%.2f}, bias: {bias:%.2f}, p_left: {left_probs})")
                excl_eids.append(eid)
        except Exception as e:
            print(e)
            print('Could not load session %s' % eid)
    if return_excluded:
        return use_eids, excl_eids
    else:
        return use_eids


def subjectdict_to_dataframe (subject_dict, one=None, verbose=False):
    if one is None:
        one = ONE()

    session_dict = {}
    subjects = subject_dict.items()
    for i, (subject, d) in enumerate(subjects):
        if verbose:
            print(f"{i+1:03d}/{len(subjects)}, {subject}")
        _d = {}
        eids = d.keys()
        for j, eid in enumerate(eids):
            if verbose:
                print(f"\t{j+1:03d}/{len(eids)}, {eid}")
            info = one.get_details(eid)
            _d = {**_d, eid: info}
        session_dict = {**session_dict, **_d}

    columns = ['subject', 'date', 'lab']
    eids = list(session_dict.keys())
    details = list(session_dict.values())
    sess_df = (pd.DataFrame({key:[info[key] for info in details] for key in columns}, \
                        columns=columns, \
                        index=eids)
                .sort_values(by=['subject', 'date']))

    return sess_df
