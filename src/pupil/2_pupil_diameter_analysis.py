import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from pprint import pprint
import seaborn as sns
from scipy.stats import pearsonr
# import pingouin as pg

import sys
from pathlib import Path
_repo = Path(__file__).resolve().parent.parent.parent  # src/pupil -> src -> repo
sys.path.insert(0, str(_repo))
import config_utils

from pupil_utils import *

# Paths from repo config
_paths = config_utils.get_paths()
data_dir = str(_paths["data_dir"])
results_dir = str(_paths["results_dir"])
pupil_data = join(data_dir, "pupil")
pupildata_directory = join(pupil_data, "processed")
pupilresults_directory = join(results_dir, "pupil", "figures")
os.makedirs(pupilresults_directory, exist_ok=True)
figure_directory = join(results_dir, "pupil", "pre_stim_diameters")
os.makedirs(figure_directory, exist_ok=True)

make_df = False
if make_df:
	print('extracting prestim diameters ...')
	vae_latents_df = pd.read_parquet(join(pupil_data, 'prestim_pupil_w_PCs.pqt'))

	# Initialize an empty list to hold the DataFrames
	dfs = []

	for ii, filename in enumerate(os.listdir(pupildata_directory)):
		pupil_df = pd.read_parquet(join(pupildata_directory, filename))
		session = pupil_df['eid'].unique()
		
		filtered_vae_latents = vae_latents_df[vae_latents_df['eid'].isin(session)]
		subject = filtered_vae_latents['subject'].unique()
		print(f"\t{subject}:{session}")

		# Plotting
		plt.figure(figsize=(12, 8))

		# Lists to hold diameters
		starting_diameters = []
		diameters_0_6 = []
		diameters_0_4 = []
		diameters_0_2 = []
		average_diameters = []
		average_adjusted_diameters = []

		rows_to_drop = []
		# Iterate through each row in the filtered vae_latents_df
		for idx, row in filtered_vae_latents.iterrows():
			stim_on = row['stimOn_time']			
			# Get the range of timestamps for the current stim_On
			time_mask = (pupil_df['timestamps'] >= stim_on - 0.6) & (pupil_df['timestamps'] <= stim_on - 0.18)
			if not all([not value for value in time_mask]):
				# Extract the corresponding horizontal_diameter values
				times = pupil_df.loc[time_mask, 'timestamps'] - stim_on
				diameters_ = pupil_df.loc[time_mask, 'horizontal_distance']
				diameters = diameters_ - diameters_.iloc[0]

				# Extract specific diameters at given times
				diam_0_6 = diameters_[pupil_df['timestamps'] >= stim_on - 0.6].iloc[0]
				diam_0_4 = diameters_[pupil_df['timestamps'] >= stim_on - 0.4].iloc[0]
				diam_0_2 = diameters_[pupil_df['timestamps'] >= stim_on - 0.2].iloc[0]

				# Append the diameters to the lists
				starting_diameters.append(diameters_.iloc[0])
				diameters_0_6.append(diam_0_6)
				diameters_0_4.append(diam_0_4)
				diameters_0_2.append(diam_0_2)
				average_diameters.append(diameters_.mean())
				average_adjusted_diameters.append(diameters.mean())
				
				# Plot each stim_On segment
				plt.plot(times, diameters)
			else:
				print(session, idx)
				# Collect the index of the row to drop
				rows_to_drop.append(idx)

		# Drop the collected rows from the DataFrame
		filtered_vae_latents = filtered_vae_latents.drop(rows_to_drop)

		filtered_vae_latents['prestimdiameters-600ms'] = diameters_0_6
		filtered_vae_latents['prestimdiameters-400ms'] = diameters_0_4
		filtered_vae_latents['prestimdiameters-200ms'] = diameters_0_2
		
		# Plot aesthetics
		plt.xlabel('Timestamps')
		plt.ylabel('Horizontal Diameter')
		plt.title(f'{subject[0]}: {session[0]}')
		plt.legend()
		plt.grid(True)
		plt.savefig(join(figure_directory, f'stim_on_plot_{ii}.png'))
		plt.close()

		dfs.append(filtered_vae_latents)

	# Concatenate all the DataFrames in the list
	final_df = pd.concat(dfs, ignore_index=True)
	# Save the final DataFrame to a CSV file
	final_df.to_parquet(join(pupil_data, 'prestim_pupil_w_PCs_.pqt'))
	print(final_df.columns)
	print(f"total number of subjects: {final_df['subject'].nunique()}")
	print(f"total number of sessions: {final_df['eid'].nunique()}")
	print(final_df.shape)

else:
	prestim_pupil_df = pd.read_parquet(join(pupil_data, 'prestim_pupil_w_PCs.pqt'))
	print(prestim_pupil_df.columns)
	print(f"total number of subjects: {prestim_pupil_df['subject'].nunique()}")
	print(f"total number of sessions: {prestim_pupil_df['eid'].nunique()}")

	# Perform additional analyses or plots with prestim_pupil_df as needed
	# plot_correlation_heatmap(prestim_pupil_df, 'subject', 'starting_diameters', 
	#     ['PC1', 'PC2', 'PC3'], join('figures', 'individual_mice_correlations_plot'))

