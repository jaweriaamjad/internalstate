"""
Step 2 (VAE): Train LSTM VAE on wheel trajectories.
Usage: python vae/2-fit_vae.py <probL> <latent_dim> <training_epochs> <seed_value>
  e.g. python vae/2-fit_vae.py all 8 400 5312
Reads: data_dir/processed_for_vae/{tag}/allsubjects/allsubjects.pqt
Writes: results_dir/vaelogs/ (models, training_history, loss_curves, test_latents).
"""

import sys
import os
import logging
from pathlib import Path
from os.path import join
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.callbacks import Callback, EarlyStopping

_VAE_DIR = Path(__file__).resolve().parent
_REPO = _VAE_DIR.parent.parent  # src/vae -> src -> repo
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_VAE_DIR))

import config_utils
import vae_utils

# GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# argv: probL, latent_dim, training_epochs, seed_value
probL = sys.argv[1]
latent_dim = int(sys.argv[2])
training_epochs = int(sys.argv[3])
seed_value = int(sys.argv[4])

vae_utils.set_seed(seed_value)

lr_schedule = 'fixed'
fit = 'allsubjects'
batch_size = 128
padding_value = -2.0
scale = 5.0
padding_value = padding_value * scale
train_size = 0.7
max_timesteps = 3000

if probL == 'unbiased':
    probLeft = [0.5]
elif probL == 'biased':
    probLeft = [0.2, 0.8]
else:
    probLeft = [0.5, 0.2, 0.8]

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

paths = config_utils.get_paths()
paper = config_utils.get_paper_config()
tag = paper["tag"]
results_dir = paths["results_dir"]
data_dir = paths["data_dir"]
data_path = join(data_dir, "processed_for_vae", tag)

os.makedirs(results_dir, exist_ok=True)

allsubjects_df = pd.read_parquet(join(data_path, "allsubjects", "allsubjects.pqt"))
x_train, x_test, idx_train, idx_test = vae_utils.dataset_for_vae(allsubjects_df, probLeft, padding_value, train_size)
print(f"total trials: {allsubjects_df.shape[0]}, samples used in the fit: {x_train.shape[0] + x_test.shape[0]}")
print(f"training samples: {x_train.shape}, test samples: {x_test.shape}")

# add_padding_and_scaling pads with padding_value then * scale; we want final mask = -10 so pass -2 and 5
train_generator = vae_utils.data_generator(x_train, batch_size, max_pad_length=max_timesteps, padding_value=-2.0, scale=5.0)
val_generator = vae_utils.data_generator(x_test, batch_size, max_pad_length=max_timesteps, padding_value=-2.0, scale=5.0)
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_test) // batch_size


def get_model(max_ts, latent_dim, padding_value):
    inp_x = tfk.layers.Input(shape=(max_ts, 1))
    masked_x = tfk.layers.Masking(mask_value=padding_value)(inp_x)
    x = tfk.layers.LSTM(256, return_sequences=True)(masked_x)
    x = tfk.layers.LSTM(128)(x)
    x = tfk.layers.Dense(64, activation="relu")(x)
    z_mean = tfk.layers.Dense(latent_dim)(x)
    z_log_sigma = tfk.layers.Dense(latent_dim)(x)
    encoder = tfk.Model(inp_x, [z_mean, z_log_sigma])
    inp_z = tfk.layers.Input(shape=(latent_dim,))
    y = tfk.layers.RepeatVector(max_ts)(inp_z)
    y = tfk.layers.LSTM(128, return_sequences=True)(y)
    y = tfk.layers.LSTM(256, return_sequences=True)(y)
    out = tfk.layers.TimeDistributed(tfk.layers.Dense(1))(y)
    decoder = tfk.Model(inp_z, out)
    z_mean, z_log_sigma = encoder(inp_x)
    z = tfk.layers.Lambda(vae_utils.sampling)([z_mean, z_log_sigma, latent_dim])
    out = decoder(z)
    vae = tfk.Model(inp_x, out)
    vae.add_loss(vae_utils.masked_vae_loss(inp_x, out, max_ts, z_log_sigma, z_mean, padding_value))
    return vae, encoder, decoder


class TerminateOnNaN(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs.get('loss', 0)) or np.isnan(logs.get('val_loss', 0)):
            print(f"Epoch {epoch + 1}: NaN loss. Stopping.")
            self.model.stop_training = True


path2logfile = join(results_dir, "vaelogs", "logfile")
os.makedirs(path2logfile, exist_ok=True)
logging.basicConfig(filename=join(path2logfile, "training.log"), level=logging.INFO, format='%(asctime)s - %(message)s')


class TrainingLogger(Callback):
    def on_train_begin(self, logs=None):
        logging.info("Training begins...")
        logging.info(f"Timestamp: {timestamp}, Seed: {seed_value}, probL: {probL}, latent_dim: {latent_dim}, epochs: {training_epochs}, batch_size: {batch_size}")

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1}: Loss - {logs['loss']}, Val Loss - {logs['val_loss']}")

    def on_train_end(self, logs=None):
        logging.info("Training ends...")


es = EarlyStopping(patience=20, verbose=1, min_delta=0.0001, mode='auto', restore_best_weights=True)
path2tb = join(results_dir, f"vaelogs/tb/{latent_dim}_{fit}_{probL}_{seed_value}_{timestamp}")
os.makedirs(path2tb, exist_ok=True)
tb = tfk.callbacks.TensorBoard(log_dir=path2tb, histogram_freq=0, write_graph=True, update_freq='epoch')

callbacks = [TerminateOnNaN(), TrainingLogger(), es, tb]

vae, enc, dec = get_model(max_timesteps, latent_dim, padding_value)
lr = vae_utils.get_lr(lr_schedule)
vae.compile(loss=None, optimizer=tfk.optimizers.Adam(learning_rate=lr))

history = vae.fit(train_generator, epochs=training_epochs, validation_data=val_generator,
                  validation_steps=validation_steps, steps_per_epoch=steps_per_epoch,
                  verbose=2, callbacks=callbacks)

save_path = f'{latent_dim}_{fit}_{probL}_{seed_value}_{timestamp}'
vae_utils.logs(results_dir, history, enc, dec, vae, x_test, max_timesteps, idx_test, save_path,
              padding_value=padding_value, scale=1.0)
