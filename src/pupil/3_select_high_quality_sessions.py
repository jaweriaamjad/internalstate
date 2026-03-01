import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys
from os.path import join
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent.parent  # src/pupil -> src -> repo
sys.path.insert(0, str(_repo))
import config_utils

# Paths from repo config
_paths = config_utils.get_paths()
data_dir = str(_paths["data_dir"])
pupil_data = join(data_dir, "pupil")
processed_dir = join(pupil_data, "processed")
trial_table_path = join(pupil_data, "prestim_pupil_w_PCs.pqt")
output_file = join(pupil_data, "high_quality_pupil_sessions.txt")

# Load master trial table
trial_table = pd.read_parquet(trial_table_path)

# Store high quality session IDs
high_quality_sessions = []

for fname in os.listdir(processed_dir):
    if not fname.endswith('_pupil_data.pqt'):
        continue
    session_id = fname.split('_pupil_data.pqt')[0]
    pupil_df = pd.read_parquet(join(processed_dir, fname))
    # Get trial info for this session
    session_trials = trial_table[trial_table['eid'] == session_id]
    mean_pupil = []
    abs_contrast = []
    for _, trial_row in session_trials.iterrows():
        stim_on = trial_row['stimOn_time']
        contrast = trial_row['contrast']
        # If trial column exists in pupil_df, filter by trial as well
        if 'trial' in pupil_df.columns and 'trial' in trial_row:
            trial_num = trial_row['trial']
            mask = (
                (pupil_df['trial'] == trial_num) &
                (pupil_df['timestamps'] >= stim_on + 2) &
                (pupil_df['timestamps'] <= stim_on + 2.5)
            )
        else:
            mask = (
                (pupil_df['timestamps'] >= stim_on + 1) &
                (pupil_df['timestamps'] <= stim_on + 2)
            )
        pupil_vals = pupil_df.loc[mask, 'horizontal_distance']
        if len(pupil_vals) > 0 and not np.isnan(contrast):
            mean_pupil.append(pupil_vals.mean())
            abs_contrast.append(np.abs(contrast))
    if len(mean_pupil) > 5:
        X = sm.add_constant(abs_contrast)
        model = sm.OLS(mean_pupil, X)
        results = model.fit()
        pval = results.pvalues[1]  # p-value for abs_contrast
        print(f"Session {session_id}: coef={results.params[1]:.3f}, p={pval:.3g}")
        if pval < 0.05:
            high_quality_sessions.append(session_id)

# Save high quality session IDs
with open(output_file, 'w') as f:
    for eid in high_quality_sessions:
        f.write(f"{eid}\n")

print("High quality sessions saved to", output_file) 