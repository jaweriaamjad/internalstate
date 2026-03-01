import pandas as pd
from os.path import join
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from pprint import pprint
from one.api import ONE
from one.alf.io import AlfBunch
import seaborn as sns

_repo = Path(__file__).resolve().parent.parent.parent  # src/pupil -> src -> repo
sys.path.insert(0, str(_repo))
import config_utils

one = ONE()
from brainbox.io.one import SessionLoader
from one.api import ONE
from pupil_utils import calculate_pupil_dia

# connect to server
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# Paths from repo config
_paths = config_utils.get_paths()
_paper = config_utils.get_paper_config()
data_dir = str(_paths["data_dir"])
results_dir = str(_paths["results_dir"])
tag = _paper.get("tag", "2023_12_bwm_release")
pupil_data = join(data_dir, "pupil")
pupildata_directory = join(pupil_data, "pupil_001")
pupilresults_directory = join(results_dir, "pupil", "figures")
os.makedirs(pupilresults_directory, exist_ok=True)
behaviordata_directory = join(data_dir, "processed_for_vae", tag, "bysubject")

plotting = False

ctr=0
# Loop over all files in the directory
for filename in os.listdir(pupildata_directory):
	# Check if the file is a regular file
	if os.path.isfile(os.path.join(pupildata_directory, filename)):
		# Split the filename using '.' as a delimiter and extract the ID from the second part
		session = filename.split('.')[1]
		subject = one.get_details(session)['subject']
		print(f"{subject}: {session}")
		pupil_df = calculate_pupil_dia(join(pupildata_directory, filename), session, subject)
		
		behavior_df = pd.read_parquet(join(behaviordata_directory, f'{subject}.pqt'))
		behavior_eid_df = behavior_df.loc[behavior_df['eid']==session]
		lastfbtime = behavior_eid_df['feedback_time'].iloc[-1]
		pupil_df_lastfbtime = pupil_df[pupil_df['timestamps']<=lastfbtime]
		pupil_df_lastfbtime.to_parquet(join(pupil_data, 'processed', f'{session}_pupil_data.pqt'), index=False)

		# Find peaks in the horizontal and vertical diameter
		peaks_horizontal, _ = find_peaks(pupil_df_lastfbtime['horizontal_distance'], prominence=0.4)  # Adjust prominence threshold as needed
		peaks_vertical, _ = find_peaks(pupil_df_lastfbtime['vertical_distance'], prominence=0.4)  # Adjust prominence threshold as needed

		# Extract timestamps corresponding to the peak indices
		timestamps_horizontalpeaks = pupil_df_lastfbtime['timestamps'].iloc[peaks_horizontal]
		ctr=ctr+1
		if plotting:
			# Plotting the pupil diameter throughout the session
			fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
			# Plot horizontal distance vs timestamps
			ax1.plot(pupil_df_lastfbtime['timestamps'], pupil_df_lastfbtime['horizontal_distance'], label='Horizontal Distance')
			ax1.set_ylabel('Horizontal Distance')
			ax1.legend()
			ax1.grid(True)
			# Plot vertical distance vs timestamps
			ax2.plot(pupil_df_lastfbtime['timestamps'], pupil_df_lastfbtime['vertical_distance'], label='Vertical Distance', color='green')
			ax2.set_ylabel('Vertical Distance')
			ax2.legend()
			# Set common x-axis label
			ax2.set_xlabel('Timestamps')
			ax2.grid(True)
			plt.tight_layout()
			# Add dashed vertical lines
			for time in timestamps_horizontalpeaks:
				ax1.axvline(time, color='red', linestyle='--')
			# 	ax2.axvline(time, color='red', linestyle='--')
			plt.savefig(join(pupilresults_directory,f'{subject}_{session}_.pdf'), dpi=300)

			# Plotting the wheel traces around the peaks
			# Compute absolute differences between each element peak diameter and the goCue times for each session
			absolute_diff = np.abs(timestamps_horizontalpeaks[:, np.newaxis] - behavior_eid_df['goCue_time'].values)
			closest_goCue = np.argmin(absolute_diff, axis=1)
			closest_position = behavior_eid_df.iloc[closest_goCue]['position']

			plt.figure(figsize=(10, 6))  # Adjust figure size as needed
			for pos in closest_position:
				plt.plot(np.arange(pos.shape[0])/50., pos,  color= 'black', linestyle = "--", lw=1.5)
			
			# plt.spines[['right', 'top']].set_visible(False)
			plt.ylabel('wheel position (rad)', fontsize=10)
			plt.xlabel('time (s)', fontsize=10)
			plt.grid(True)
			plt.savefig(join(pupilresults_directory,f'{subject}_{session}_positionatpeaks.pdf'), dpi=300)
print(f'total number of sessions processed are {ctr}')