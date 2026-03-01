"""
VAE utilities: dataset building (for 1-get_data4vae), training, and latent extraction.
Combines logic from behavior/vae/utils/vaefit_util and vaedataset_utils.
"""

import os
import random
import json
import numpy as np
import pandas as pd
from os.path import join
from collections import defaultdict

import scipy.interpolate as interpolate
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import pad_sequences

# Optional TensorFlow (needed for fit and get_latents only)
try:
    import tensorflow as tf
    import tensorflow.keras as tfk
    from tensorflow.keras import backend as K
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ----- Dataset building (for 1-get_data4vae). Uses ONE. -----

def _get_qc_trials_wheel(position):
    """Indices of trials with wheel position arrays of length > 1."""
    qc_trials = [i for i in range(len(position)) if (len(position[i]) > 1)]
    rej_trials = [i for i in range(len(position)) if (len(position[i]) <= 1)]
    return qc_trials, rej_trials


def _interpolate_position(ts, pos, freq=1000, kind='linear', fill_gaps=None):
    t = np.arange(ts[0], ts[-1], 1 / freq)
    pos_interp = interpolate.interp1d(ts, pos, kind=kind)(t)
    if fill_gaps:
        gaps, = np.where(np.diff(ts) >= fill_gaps)
        for i in gaps:
            pos_interp[(t >= ts[i]) & (t < ts[i + 1])] = pos[i]
    return pos_interp, t


def _get_clip_idx(pos, ts, stop_position=0.3, stop_time=60.0):
    overshoot_p = np.argwhere(np.abs(pos) > stop_position)
    overshoot_t = np.argwhere(ts > stop_time)
    overshoot_p = overshoot_p[0] if len(overshoot_p) > 0 else np.inf
    overshoot_t = overshoot_t[0] if len(overshoot_t) > 0 else np.inf
    return min(overshoot_p, overshoot_t)


def _curate_wheel_data(raw_position, raw_timestamps, freq=50):
    n_trials = len(raw_position)
    position, timestamps, duration, max_position = [], [], [], []
    for ii in range(n_trials):
        timestamps_0 = raw_timestamps[ii] - raw_timestamps[ii][0]
        position_0 = raw_position[ii] - raw_position[ii][0]
        position_interp, timestamps_interp = _interpolate_position(timestamps_0, position_0, freq=50)
        overshoot = _get_clip_idx(position_interp, timestamps_interp)
        if overshoot != np.inf:
            stop_ind = int(overshoot) if np.isscalar(overshoot) else int(np.squeeze(overshoot).flat[0])
            timestamps_final = timestamps_interp[0:stop_ind]
            position_final = position_interp[0:stop_ind]
        else:
            timestamps_final = timestamps_interp
            position_final = position_interp
        duration.append(timestamps_final[-1])
        max_position.append(position_final[-1])
        position.append(position_final)
        timestamps.append(timestamps_final)
    return np.array(position), np.array(timestamps), np.array(max_position), np.array(duration)


def _create_stim_vector(stim_left, stim_right):
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    return stim_right - stim_left


class _WheelData:
    """Trialwise wheel data and movement onset times. Uses ONE."""

    def __init__(self, eid, one=None):
        if one is None:
            from one.api import ONE
            one = ONE()
        self.one = one
        wheel_data = one.load_object(eid, 'wheel', collection='alf')
        self.position = wheel_data.position
        self.timestamps = wheel_data.timestamps
        self.data_error = False
        if str(type(self.position)) == "<class 'pathlib.PosixPath'>" or str(type(self.timestamps)) == "<class 'pathlib.PosixPath'>":
            self.data_error = True
        else:
            self.velocity = self._calc_wheel_velocity()

    def _calc_wheel_velocity(self):
        wv = [0.0]
        for i in range(len(self.position) - 1):
            wv.append((self.position[i + 1] - self.position[i]) / (self.timestamps[i + 1] - self.timestamps[i]))
        return wv

    def calc_trialwise_wheel(self, stimOn_times, feedback_times):
        self.stimOn_pre_duration = 0.3
        self.total_trial_count = len(stimOn_times)
        self.trial_position = [[] for _ in range(self.total_trial_count)]
        self.trial_timestamps = [[] for _ in range(self.total_trial_count)]
        self.trial_velocity = [[] for _ in range(self.total_trial_count)]
        tridx = 0
        for tsidx in range(len(self.timestamps)):
            t = self.timestamps[tsidx]
            while tridx < len(stimOn_times) - 1 and t > stimOn_times[tridx + 1] - self.stimOn_pre_duration:
                tridx += 1
            if stimOn_times[tridx] - self.stimOn_pre_duration <= t and t < feedback_times[tridx]:
                self.trial_position[tridx].append(self.position[tsidx])
                self.trial_timestamps[tridx].append(self.timestamps[tsidx])
                self.trial_velocity[tridx].append(self.velocity[tsidx])

    def calc_movement_onset_times(self, stimOn_times):
        speed_threshold = 0.5
        duration_threshold = 0.05
        self.movement_onset_times = []
        self.first_movement_onset_times = np.zeros(self.total_trial_count)
        self.last_movement_onset_times = np.zeros(self.total_trial_count)
        self.movement_onset_counts = np.zeros(self.total_trial_count)
        for tridx in range(len(self.trial_timestamps)):
            self.movement_onset_times.append([])
            cm_dur = 0.0
            for tpidx in range(len(self.trial_timestamps[tridx])):
                t = self.trial_timestamps[tridx][tpidx]
                tprev = stimOn_times[tridx] - self.stimOn_pre_duration if tpidx == 0 else self.trial_timestamps[tridx][tpidx - 1]
                cm_dur += (t - tprev)
                if np.abs(self.trial_velocity[tridx][tpidx]) > speed_threshold:
                    if cm_dur > duration_threshold:
                        self.movement_onset_times[tridx].append(t)
                    cm_dur = 0.0
            self.movement_onset_counts[tridx] = len(self.movement_onset_times[tridx])
            if len(self.movement_onset_times[tridx]) == 0:
                self.first_movement_onset_times[tridx] = np.nan
                self.last_movement_onset_times[tridx] = np.nan
            else:
                self.first_movement_onset_times[tridx] = self.movement_onset_times[tridx][0]
                self.last_movement_onset_times[tridx] = self.movement_onset_times[tridx][-1]


def _get_raw_data(eid, one=None):
    if one is None:
        from one.api import ONE
        one = ONE()
    info = one.eid2ref(eid)
    subject = info['subject']
    trials = one.load_object(eid, 'trials', collection='alf')
    stim_right = trials.contrastRight
    stim_left = trials.contrastLeft
    choice = trials.choice
    bias_probs = trials.probabilityLeft
    feedback_times = trials.feedback_times
    stimOn_times = trials.stimOn_times
    goCue_times = trials.goCue_times
    rewarded = trials.feedbackType
    wheel = _WheelData(eid, one=one)
    wheel.calc_trialwise_wheel(stimOn_times, feedback_times)
    wheel.calc_movement_onset_times(stimOn_times)
    raw_position = wheel.trial_position
    raw_timestamps = wheel.trial_timestamps
    FMOT_times = wheel.first_movement_onset_times
    qc_idx, rej_idx = _get_qc_trials_wheel(raw_position)
    qc_idx.sort()
    qc_stim_right = np.array(stim_right)[qc_idx]
    qc_stim_left = np.array(stim_left)[qc_idx]
    qc_FMOT_times = FMOT_times[qc_idx]
    qc_choice = np.array(choice)[qc_idx]
    qc_bias_probs = np.array(bias_probs)[qc_idx]
    qc_rewarded = np.array(rewarded)[qc_idx]
    qc_feedback_times = np.array(feedback_times)[qc_idx]
    qc_stimOn_times = np.array(stimOn_times)[qc_idx]
    qc_goCue_times = np.array(goCue_times)[qc_idx]
    qc_raw_position = [raw_position[i] for i in qc_idx]
    qc_raw_timestamps = [raw_timestamps[i] for i in qc_idx]
    assert qc_raw_position.count([]) == 0, f"some trials have empty wheel position arrays"
    return (subject, qc_stim_left, qc_stim_right, qc_rewarded, qc_choice, qc_stimOn_times,
            qc_goCue_times, qc_FMOT_times, qc_feedback_times, qc_bias_probs, qc_raw_position, qc_raw_timestamps, rej_idx)


def get_all_data_this_session(eid, one=None):
    """Load and curate wheel/trial data for one session. Returns (session_data_tuple, number_of_trials, rej_trials, to_save)."""
    to_save = False
    out = _get_raw_data(eid, one=one)
    subject, stim_left, stim_right, rewarded, choice, so_times, gc_times, fmot_times, fb_times, bias_probs, raw_position, raw_timestamps, rej_idx = out
    num_viols = len(np.where(choice == 0)[0])
    number_of_trials = len(bias_probs)
    if num_viols < (0.1 * number_of_trials):
        to_save = True
        position, timestamps, max_pos, duration = _curate_wheel_data(raw_position, raw_timestamps)
        contrast = _create_stim_vector(stim_left, stim_right)
        session = np.array([eid for _ in range(contrast.shape[0])])
        subject_arr = np.array([subject for _ in range(contrast.shape[0])])
        rej_trials = rej_idx
        return (subject_arr, session, position, bias_probs, contrast, choice, rewarded, so_times, gc_times, fmot_times, fb_times, max_pos, duration), number_of_trials, rej_trials, to_save
    return (None,) * 13, number_of_trials, rej_idx, to_save


# ----- VAE fit and latent helpers (require TensorFlow) -----

def set_seed(seed):
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for set_seed")
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def sampling(args):
    z_mean, z_log_sigma, latent_dim = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def masked_vae_loss(original, out, ts, z_log_sigma, z_mean, padding_value):
    mask = tf.cast(tf.not_equal(original, padding_value), dtype=tf.float32)
    reconstruction = K.mean(K.square(tf.math.multiply((original - out), mask)) * ts)
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    return reconstruction + kl


def get_lr(lr_schedule):
    if lr_schedule == 'ed':
        return tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100000, decay_rate=0.96, staircase=True)
    return 0.001


def dataset_for_vae(df, bias_probs, padding_value, train_size, high_freq_th=None, duration=None):
    if duration:
        df_for_training = df[(df['bias_probs'].isin(bias_probs)) & (df['duration'].between(0, duration))]
    else:
        df_for_training = df[(df['bias_probs'].isin(bias_probs))]
    pos_shuffled, idx_shuffled = shuffle(df_for_training['position'].values, df_for_training.index, random_state=0)
    x_train, x_test, idx_train, idx_test = train_test_split(pos_shuffled, idx_shuffled, test_size=1 - train_size, random_state=42)
    return (x_train, x_test, idx_train, idx_test)


def data_generator(X, batch_size, max_pad_length=3000, padding_value=-2.0, scale=5.0):
    num_samples = len(X)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = [X[idx] for idx in batch_indices]
            batch_X_padded = add_padding_and_scaling(batch_X, max_pad_length, padding_value=padding_value, scale=scale)
            yield batch_X_padded


def add_padding_and_scaling(position, max_pad_length=3000, padding_value=-2, scale=5.0):
    pos_padded = pad_sequences(position, value=padding_value, dtype='float32',
                               maxlen=max_pad_length, padding='post', truncating='post')
    return pos_padded * scale


def loss_curves(history, name):
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history.history)
    epochs = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, frame[['loss', 'val_loss']], 'o--', linewidth=2)
    ax.set_xlabel('epochs', fontsize=16)
    ax.set_ylabel('loss', fontsize=16)
    ax.legend(['Train', 'Validation'])
    plt.savefig(f"{name}.png")
    plt.close()


def logs(results_dir, history, enc, dec, vae, x_test, max_timesteps, idx_test, save_path, padding_value=-10.0, scale=1.0):
    """padding_value should be the final masked value (e.g. -10); scale for test generator: use 1.0 so pad value stays padding_value."""
    base = join(results_dir, 'vaelogs')
    path2traininghistory = join(base, 'training_history', save_path)
    os.makedirs(path2traininghistory, exist_ok=True)
    pd.DataFrame(history.history).to_parquet(join(path2traininghistory, "training_history.pqt"))
    path2losscurves = join(base, 'loss_curves')
    os.makedirs(path2losscurves, exist_ok=True)
    loss_curves(history, join(path2losscurves, save_path))
    path2models = join(base, 'models', save_path)
    os.makedirs(path2models, exist_ok=True)
    vae.save(join(path2models, 'vaelstm'))
    enc.save(join(path2models, 'vaelstm_enc'))
    dec.save(join(path2models, 'vaelstm_dec'))
    path2latents = join(base, 'test_latents', save_path)
    os.makedirs(path2latents, exist_ok=True)
    test_gen = data_generator(x_test, 256, max_pad_length=max_timesteps, padding_value=padding_value, scale=scale)
    test_steps = (len(x_test) + 255) // 256
    z_mean, z_log_sigma = enc.predict(test_gen, steps=test_steps)
    z_dict = {'x_test': x_test, 'idx_test': idx_test, 'z_mean': z_mean, 'z_sigma': z_log_sigma}
    np.save(join(path2latents, 'conditionalvaelstm_latents_dict.npy'), z_dict)


def get_vae_latents(df, probLeft, folder, results_dir, encoder_dir=None, padding_value=-2.0, scale=5.0, max_pad_length=3000):
    """Load trained encoder from encoder_dir/vaelstm_enc (if encoder_dir set) else results_dir/vaelogs/models/{folder},
    run on df, save to results_dir/vaelogs/all_latents/{folder}."""
    if not _TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for get_vae_latents")
    df_probL = df.loc[df['bias_probs'].isin(probLeft)]
    # Same padding/scaling as training: pad with -2 then *5
    position = add_padding_and_scaling(
        list(df_probL['position'].values),
        max_pad_length=max_pad_length,
        padding_value=padding_value,
        scale=scale,
    )
    if encoder_dir is not None:
        encoder_path = join(encoder_dir, 'vaelstm_enc')
    else:
        encoder_path = join(results_dir, 'vaelogs', 'models', folder, 'vaelstm_enc')
    vaeenc_loaded = tf.keras.models.load_model(encoder_path, compile=False)
    dataset_size = position.shape[0]
    z_mean_list, z_log_sigma_list = [], []
    for start_idx in range(0, dataset_size, 40000):
        end_idx = min(start_idx + 40000, dataset_size)
        zm, zs = vaeenc_loaded.predict(position[start_idx:end_idx])
        z_mean_list.append(zm)
        z_log_sigma_list.append(zs)
    df_probL = df_probL.copy()
    df_probL['z_mean'] = list(np.concatenate(z_mean_list))
    df_probL['z_log_sigma'] = list(np.concatenate(z_log_sigma_list))
    out_dir = join(results_dir, 'vaelogs', 'all_latents', folder)
    os.makedirs(out_dir, exist_ok=True)
    df_probL.to_parquet(join(out_dir, "data_with_vaelatents.pqt"))
