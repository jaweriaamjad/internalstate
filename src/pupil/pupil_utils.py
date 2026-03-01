import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pprint import pprint
from one.api import ONE
from one.alf.io import AlfBunch
from scipy.stats import pearsonr
# import pingouin as pg

one = ONE()
from brainbox.io.one import SessionLoader
from one.api import ONE

def calculate_pupil_dia(file_path, session, subject):
	sl = SessionLoader(one, session)
	sl.load_pose(views=['left'])
	# access times
	times = sl.pose['leftCamera']['times']
	# Read the CSV file into a DataFrame
	df = pd.read_csv(file_path, header=None)
	df = df.drop([0, 1, 2])
	df.dropna(axis=1, inplace=True, how='all')
	df = df.astype(float)
	df.columns = ['index', 'pupil_top_x', 'pupil_top_y', 'pupil_right_x', 'pupil_right_y', 'pupil_bottom_x', 'pupil_bottom_y', 'pupil_left_x', 'pupil_left_y']
	df.set_index('index', inplace=True)
	horiz_dst = np.sqrt((df['pupil_right_x'] - df['pupil_left_x'])**2 + (df['pupil_right_y'] - df['pupil_left_y'])**2)
	df['horizontal_distance'] = normalise_dfcolumn(horiz_dst)
	vrtcl_dst = np.sqrt((df['pupil_top_x'] - df['pupil_bottom_x'])**2 + (df['pupil_top_y'] - df['pupil_bottom_y'])**2)
	df['vertical_distance'] = normalise_dfcolumn(vrtcl_dst)

	df_diameter = pd.DataFrame({'eid': [session]* df.shape[0], 'subject': [subject]*df.shape[0], 
		'vertical_distance': df['vertical_distance'], 'horizontal_distance': df['horizontal_distance'],
		'timestamps': times})
	# df['timestamps'] = times
	# df['eid'] = session
	# df['subject'] = subject
	return df_diameter

def pc_rotation(df, rotation_angle=25):
    # Define the angle of rotation in radians (e.g., 45 degrees)
    theta = np.radians(rotation_angle)
    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    PC1 = df['PC1'].values
    PC2 = df['PC2'].values
    # Your existing PC1 and PC2 data
    data = np.column_stack([PC1, PC2])
    data_standardized = data / np.std(data, axis=0)
    # Apply the rotation
    rotated_data = data.dot(rotation_matrix)
     # If necessary, rescale the rotated data back to original variance
    rotated_data_rescaled = rotated_data * np.std(data, axis=0)
    # Store the rotated components
    df['PC1'] = rotated_data_rescaled[:, 0]*(-1)
    df['PC2'] = rotated_data_rescaled[:, 1]
    return df

def normalise_dfcolumn(values):
	# Min-max normalization
	min_val = values.min()
	max_val = values.max()
	normalized_values = (values - min_val) / (max_val - min_val)
	return normalized_values

def calc_trianwise_pupil(pupil_diameter, timestamps, begin_time, end_time):
	trialwise_pupil = []
	for ii in range(begin_time.shape[0]):
		indx_0 = np.argmin(np.abs(begin_time[ii]-timestamps))
		indx_1 = np.argmin(np.abs(end_time[ii]-timestamps))
		pupil_ii = pupil_diameter[indx_0:indx_1]
		trialwise_pupil.append(pupil_ii)
	return trialwise_pupil

def calc_trianwise_pupil(pupil_diameter, timestamps, stomOn_time, window=[-0.4, -0.2]):
	pupil_dia = []
	for ii in range(begin_time.shape[0]):
		indx_0 = np.argmin(np.abs(begin_time[ii]-timestamps))
		indx_1 = np.argmin(np.abs(end_time[ii]-timestamps))
		pupil_ii = pupil_diameter[indx_0:indx_1]
		trialwise_pupil.append(pupil_ii)
	return trialwise_pupil


def calculate_correlations(prestim_df, target_columns):
	embedding_columns = [col for col in prestim_df.columns if col.startswith('PC')]
	# Calculate correlations
	correlations = prestim_df.corr()
	correlation_with_targets = correlations[target_columns].loc[embedding_columns]
	return correlation_with_targets

def create_correlation_plots(correlation_data, target_cols, plot_titles, filename):
	num_plots = len(target_cols)
	fig, axes = plt.subplots(1, num_plots, figsize=(20, 5), sharey=True)

	if num_plots == 1:
		axes = [axes]

	for ax, target_col, title in zip(axes, target_cols, plot_titles):
		sns.barplot(y=correlation_data.index, x=correlation_data[target_col], ax=ax, palette="viridis")
		ax.set_title(title, fontsize=16)
		ax.set_xlabel('Correlation', fontsize=16)
		ax.set_ylabel('Features', fontsize=16)
		ax.axvline(0, color='grey', linestyle='--')
		ax.tick_params(axis='y', labelsize=12)

	plt.tight_layout()
	plt.savefig(filename)



def plot_correlations_with_ci(df, source_col, target_cols, filename):
	correlations = {}
	starting_diameters = df[source_col]
	
	for col in target_cols:
		# Compute Pearson correlation coefficient and p-value
		res = pearsonr(starting_diameters, df[col])
		corr_coef, p_value = res
		
		# Compute 95% confidence interval
		ci = res.confidence_interval(confidence_level=0.9)
		
		correlations[col] = {
			'correlation': corr_coef,
			'p_value': p_value,
			'ci_lower': ci[0],
			'ci_upper': ci[1]
		}

	# Create DataFrame from correlations dictionary
	correlations_df = pd.DataFrame(correlations).T
	correlations_df['ci_error'] = correlations_df['ci_upper'] - correlations_df['correlation']
	
	# Plotting
	plt.figure(figsize=(8, 7))
	sns.set(style="whitegrid")
	
	barplot = sns.barplot(x=correlations_df.index, y=correlations_df['correlation'],
		errorbar=None, width=0.4)
	
	# Adding error bars
	for i, (index, row) in enumerate(correlations_df.iterrows()):
		plt.errorbar(i, row['correlation'], 
					 yerr=[[row['correlation'] - row['ci_lower']], [row['ci_upper'] - row['correlation']]],
					 fmt='none', c='black', capsize=5, capthick=2)
	
	# Adding asterisks for significance
	for i, (index, row) in enumerate(correlations_df.iterrows()):
		p_value = row['p_value']
		if p_value < 0.001:
			significance = '***'
		elif p_value < 0.01:
			significance = '**'
		elif p_value < 0.05:
			significance = '*'
		else:
			significance = ''
		
		# Annotate the plot
		y = row['correlation']
		offset = 10 if y > 0 else -30
		plt.annotate(significance, xy=(i, y), xytext=(0, offset),
					 textcoords='offset points', ha='center', va='bottom', fontsize=20, color='red')
	
	# Enhancing plot aesthetics
	plt.title('Correlations with Pupil Diameters at [stimOn-0.6s] (90% Confidence Intervals)', fontsize=12)
	plt.ylabel('Correlation Coefficient', fontsize=14)
	plt.xlabel('Principal Components', fontsize=14)
	# plt.ylim(-0.1, 0.12)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	sns.despine(left=True)
	
	plt.tight_layout()
	plt.savefig(filename)




# def plot_correlation_heatmap(df, subject_col, source_col, target_cols, filename):
# 	# Initialize a dictionary to store the results
# 	correlations = {col: [] for col in target_cols}
# 	p_values = {col: [] for col in target_cols}

# 	# Group the data by subject
# 	grouped = df.groupby(subject_col)

# 	for subject, group in grouped:
# 		starting_diameters = group[source_col]
		
# 		for col in target_cols:
# 			# Compute Pearson correlation coefficient and p-value
# 			corr_coef, p_value = pearsonr(starting_diameters, group[col])
			
# 			# Append the results to the dictionary
# 			correlations[col].append(corr_coef)
# 			p_values[col].append(p_value)
	
# 	# Convert the results to DataFrames
# 	corr_df = pd.DataFrame(correlations, index=grouped.groups.keys())
# 	pval_df = pd.DataFrame(p_values, index=grouped.groups.keys())

# 	# Create a heatmap for each PC
# 	for col in target_cols:
# 		plt.figure(figsize=(18, 6))
# 		sns.set(style="whitegrid")
		
# 		# Create the heatmap
# 		ax = sns.heatmap(corr_df[[col]].T, annot=False, linewidth=.5, cmap="coolwarm", cbar=True, center=0, vmin=-0.5, vmax=0.5)
		
# 		# Annotate the heatmap with asterisks for significance
# 		for i in range(corr_df.shape[0]):
# 			p_value = pval_df[col].iloc[i]
# 			if p_value < 0.001:
# 				significance = '***'
# 			elif p_value < 0.01:
# 				significance = '**'
# 			elif p_value < 0.05:
# 				significance = '*'
# 			else:
# 				significance = 'ns'
			
# 			# Get the correlation value for the annotation
# 			corr_value = corr_df[col].iloc[i]
			
# 			# Annotate the plot
# 			ax.text(i + 0.5, 0.5, significance, ha='center', va='center', color='black', fontsize=14)

# 		plt.title(f'Correlation of {col} with Pupil Diameters at [stimOn-0.6s]')
# 		plt.xlabel('Subjects')
# 		plt.ylabel('')

# 		# Save the plot
# 		plt.tight_layout()
# 		plt.savefig(f"{filename}_{col}.png")


def plot_correlation_heatmap(df, subject_col, source_col, target_cols, filename):
	# Initialize a dictionary to store the results
	correlations = {col: [] for col in target_cols}
	p_values = {col: [] for col in target_cols}

	# Group the data by subject
	grouped = df.groupby(subject_col)

	for subject, group in grouped:
		starting_diameters = group[source_col]
		
		for col in target_cols:
			# Compute Pearson correlation coefficient and p-value
			corr_coef, p_value = pearsonr(starting_diameters, group[col])
			
			# Append the results to the dictionary
			correlations[col].append(corr_coef)
			p_values[col].append(p_value)
	
	# Convert the results to DataFrames
	corr_df = pd.DataFrame(correlations, index=grouped.groups.keys())
	pval_df = pd.DataFrame(p_values, index=grouped.groups.keys())

	# Sort subjects by the correlation with PC1
	sorted_indices = corr_df['PC1'].sort_values(ascending=False).index

	# Reorder the DataFrame based on the sorted indices
	corr_df = corr_df.loc[sorted_indices]
	pval_df = pval_df.loc[sorted_indices]

	# Plot heatmap for PC1
	plt.figure(figsize=(18, 6))
	sns.set(style="whitegrid")
	ax = sns.heatmap(corr_df[['PC1']].T, annot=False, linewidths=.5, cmap="coolwarm", cbar=True, center=0, vmin=-0.5, vmax=0.5)

	# Annotate the heatmap with asterisks for significance
	for i in range(corr_df.shape[0]):
		p_value = pval_df['PC1'].iloc[i]
		if p_value < 0.001:
			significance = '***'
		elif p_value < 0.01:
			significance = '**'
		elif p_value < 0.05:
			significance = '*'
		else:
			significance = 'ns'
		
		# Annotate the plot
		ax.text(i + 0.5, 0.5, significance, ha='center', va='center', color='black', fontsize=14)

	plt.title('Correlation of PC1 with Pupil Diameters at [stimOn-0.6s]')
	plt.xlabel('Subjects')
	plt.ylabel('')
	plt.tight_layout()
	plt.savefig(f"{filename}_PC1_sorted.png")
	plt.show()

	# Plot heatmap for PC1, PC2, and PC3
	plt.figure(figsize=(18, 6))
	sns.set(style="whitegrid")
	ax = sns.heatmap(corr_df[target_cols].T, annot=False, linewidths=.5, cmap="coolwarm", cbar=True, center=0, vmin=-0.5, vmax=0.5)

	# Annotate the heatmap with asterisks for significance
	for j, col in enumerate(target_cols):
		for i in range(corr_df.shape[0]):
			p_value = pval_df[col].iloc[i]
			if p_value < 0.001:
				significance = '***'
			elif p_value < 0.01:
				significance = '**'
			elif p_value < 0.05:
				significance = '*'
			else:
				significance = 'ns'
			
			# Annotate the plot
			ax.text(i + 0.5, j + 0.5, significance, ha='center', va='center', color='black', fontsize=14)

	plt.title('Correlation of PCs with Pupil Diameters at [stimOn-0.6s]')
	plt.xlabel('Subjects')
	plt.ylabel('')
	plt.tight_layout()
	plt.savefig(f"{filename}_PCs_sorted.pdf")
	plt.show()



def plot_pvalue_hist(df, condition_col, source_col, target_cols, filename):
	# Initialize dictionaries to store results
	correlations = {col: [] for col in target_cols}
	p_values = {col: [] for col in target_cols}

	# Group the data by condition_col
	grouped = df.groupby(condition_col)

	for condition, group in grouped:
		starting_diameters = group[source_col]
		
		for col in target_cols:
			# Compute Pearson correlation coefficient and p-value
			corr_coef, p_value = pearsonr(starting_diameters, group[col])
			
			# Append results to dictionaries
			correlations[col].append(corr_coef)
			p_values[col].append(p_value)

	# Convert dictionaries to DataFrames
	corr_df = pd.DataFrame(correlations, index=grouped.groups.keys())
	pval_df = pd.DataFrame(p_values, index=grouped.groups.keys())

	# Plotting histograms for corr_df and pval_df in subplots
	fig, axes = plt.subplots(nrows=2, ncols=len(target_cols), figsize=(15, 10))

	# Plot histograms for corr_df in the top row
	for i, col in enumerate(target_cols):
		ax = axes[0, i] if len(target_cols) > 1 else axes[0]  # Handle single column case
		ax.hist(corr_df[col], bins=50, color='lightgreen', edgecolor='white', linewidth=1.2)
		ax.set_title(f'Correlation Histogram for {col}')
		ax.set_xlabel('Correlation Coefficients')
		ax.set_ylabel('Frequency')

	# Plot histograms for pval_df in the bottom row
	for i, col in enumerate(target_cols):
		ax = axes[1, i] if len(target_cols) > 1 else axes[1]  # Handle single column case
		ax.hist(pval_df[col], bins=50, color='skyblue', edgecolor='white', linewidth=1.2)
		ax.set_title(f'P-value Histogram for {col}')
		ax.set_xlabel('P-values')
		ax.set_ylabel('Frequency')


	# Adjust layout and save the plot
	plt.tight_layout()
	plt.savefig(f'{filename}.png')

def compute_pearson_and_pvalue(df, col1, col2):
	"""
	Computes the Pearson correlation coefficient and p-value between two columns of a DataFrame.

	Parameters:
	df (pd.DataFrame): The DataFrame containing the data.
	col1 (str): The name of the first column.
	col2 (str): The name of the second column.

	Returns:
	tuple: Pearson correlation coefficient and p-value.
	"""
	# Extract the two columns
	x = df[col1].values
	y = df[col2].values
	
	# Compute Pearson correlation and p-value
	pearson_corr, p_value = pearsonr(x, y)
	
	return pearson_corr, p_value

def bootstrap_pearsonr(x, y, n_bootstraps=1000):

	x, y = np.array(x), np.array(y)
	
	# Remove NaNs from the data
	valid_indices = ~np.isnan(x) & ~np.isnan(y)
	x_clean, y_clean = x[valid_indices], y[valid_indices]

	# Initialize bootstrap results
	bootstrap_r = []
	n = len(x_clean)
	
	# Perform bootstrap resampling
	for _ in range(n_bootstraps):
		idx = np.random.choice(range(n), size=n, replace=True)
		r, _ = pearsonr(x_clean[idx], y_clean[idx])
		bootstrap_r.append(r)
	
	# Compute 95% confidence intervals
	r_lower = np.percentile(bootstrap_r, 2.5)
	r_upper = np.percentile(bootstrap_r, 97.5)
	
	return np.mean(bootstrap_r), r_lower, r_upper


# Function to plot Pearson r as a bar plot with error bars
def plot_pearson_r_with_errorbars(mean_r, lower_r, upper_r, labels, figure_path):
	# Ensure labels are provided or generate generic labels
	if labels is None:
		labels = [f"Pearson r {i+1}" for i in range(len(mean_r))]

	# Convert lists to numpy arrays for element-wise operations
	mean_r = np.array(mean_r)
	lower_r = np.array(lower_r)
	upper_r = np.array(upper_r)

	# Calculate the error for plotting
	error = [mean_r - lower_r, upper_r - mean_r]

	# Set plot style
	sns.set(style="whitegrid")

	# Create the bar plot
	plt.figure(figsize=(4,3))

	# Set positions of bars closer together (positions for 2 bars)
	bar_positions = np.arange(len(labels))  # [0, 1]

	# Plot bars with error bars
	plt.bar(bar_positions, mean_r, yerr=error, capsize=5, color='#808080', 
			width=0.2, edgecolor='black', ecolor='r', alpha=0.8)

	# Aesthetics: labels, title, grid, and despine
	plt.ylabel("Pearson Correlation", fontsize=14)
	
	# Adjust the y-axis limit to focus on small r values
	# plt.ylim(-0.05, 0.2)

	# Add horizontal line at y=0 for reference
	plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
	plt.grid(False)

	# Set x-ticks and move labels closer together
	plt.xticks(bar_positions, labels)

	# Remove top and right spines for a clean look
	sns.despine()

	# Save the plot
	plt.savefig(join(figure_path, 'pupil_corr_PC1_glm.pdf'), bbox_inches='tight')