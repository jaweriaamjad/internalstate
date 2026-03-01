"""
Plotting helpers for figure scripts (fig3, fig6, etc.).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

try:
    import psychofit as psy
except ImportError:
    psy = None


def pc_rotation(df, rotation_angle=30):
    """Rotate PC1/PC2 in place by rotation_angle degrees."""
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    PC1 = df['PC1'].values
    PC2 = df['PC2'].values
    data = np.column_stack([PC1, PC2])
    data_standardized = data / np.std(data, axis=0)
    rotated_data = data.dot(rotation_matrix)
    rotated_data_rescaled = rotated_data * np.std(data, axis=0)
    df['PC1'] = rotated_data_rescaled[:, 0] * (-1)
    df['PC2'] = rotated_data_rescaled[:, 1]
    return df


def plot_wheelInPCspace(df_z, n_binsx, n_binsy, n_traj, embed_type='PC', out_path=None, z_score=False):
    """
    Plot wheel trajectories in PC space. Returns selected_indices for use in scatterplot_latents.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    if z_score:
        scaler = StandardScaler()
        df_z[f'{embed_type}1'] = scaler.fit_transform(df_z[[f'{embed_type}1']]).flatten()

    selected_indices = []
    fig_width = 14
    fig_height = 8
    fig, axs = plt.subplots(n_binsy, n_binsx, figsize=(fig_width, fig_height), sharey=True)

    PC1_values = np.linspace(df_z[f'{embed_type}1'].min(), df_z[f'{embed_type}1'].max(), n_binsx + 1)
    PC2_values = np.linspace(df_z[f'{embed_type}2'].min(), df_z[f'{embed_type}2'].max(), n_binsy + 1)

    for ii in range(n_binsx):
        counter = 0
        for jj in range(n_binsy - 1, -1, -1):
            indices = df_z.index[
                (df_z[f"{embed_type}1"].between(PC1_values[ii], PC1_values[ii + 1])) &
                (df_z[f"{embed_type}2"].between(PC2_values[counter], PC2_values[counter + 1]))
            ].tolist()
            if len(indices) > 0:
                for kk in range(n_traj):
                    selected_index = np.random.choice(indices)
                    selected_indices.append(selected_index)
                    traj = df_z.loc[selected_index, "position"]
                    axs[jj][ii].plot(np.arange(traj.shape[0]) / 50.0, traj, color="black", linestyle="--", lw=1.5)
                axs[jj][ii].spines[['right', 'top']].set_visible(False)
                axs[jj][ii].grid(False)
                axs[jj][ii].tick_params(axis='both', which='major', labelsize=8)
            else:
                axs[jj][ii].axis('off')
            counter += 1

    axs[n_binsy - 1][0].set_ylabel('wheel position (rad)', fontsize=12)
    axs[n_binsy - 1][0].set_xlabel('time (s)', fontsize=12)

    ax2 = plt.axes(facecolor=(1, 1, 1, 0))
    ax2.set(xlim=(PC1_values[0], PC1_values[-1]), xticks=PC1_values,
            ylim=(PC2_values[0], PC2_values[-1]), yticks=PC2_values)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color('black')
    ax2.spines[['left', 'bottom']].set_linewidth(2.0)
    ax2.tick_params(colors='black')
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.spines.left.set_position(('outward', 55))
    ax2.spines.bottom.set_position(('outward', 45))
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(False)
    ax2.set_ylabel('Wheel Direction', fontsize=18, color='black')
    ax2.set_xlabel('Engagement Index', fontsize=18, color='black')

    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return selected_indices


def scatterplot_latents(data_df, selected_indices, out_path=None):
    """Scatter plot of latent space (PC1 vs PC2) with selected trajectories highlighted."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, ax_main = plt.subplots(figsize=(14, 6))
    PC1 = data_df["PC1"].values
    PC2 = data_df["PC2"].values
    selected_PC1 = data_df.loc[selected_indices, "PC1"].values
    selected_PC2 = data_df.loc[selected_indices, "PC2"].values

    ax_main.scatter(PC1, PC2, alpha=0.7, s=0.3, color='#808080', rasterized=True)
    ax_main.scatter(selected_PC1, selected_PC2, alpha=0.7, s=10, color='black', marker='^')
    ax_main.set_xlabel('Engagement Index', fontsize=18, color='black')
    ax_main.set_ylabel('Wheel Direction', fontsize=18, color='black')
    ax_main.spines[['right', 'top']].set_visible(False)
    ax_main.spines[['left', 'bottom']].set_color('black')
    ax_main.spines[['left', 'bottom']].set_linewidth(2.0)
    ax_main.tick_params(axis='both', which='major', labelsize=12)
    ax_main.tick_params(colors='black')
    ax_main.yaxis.set_ticks_position('left')
    ax_main.xaxis.set_ticks_position('bottom')
    ax_main.spines.left.set_position(('outward', 10))
    ax_main.spines.bottom.set_position(('outward', 10))

    x_min, x_max = PC1.min(), PC1.max()
    y_min, y_max = PC2.min(), PC2.max()
    x_ticks = np.linspace(x_min, x_max, 8 + 1)
    y_ticks = np.linspace(y_min, y_max, 5 + 1)
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xticks(x_ticks)
    ax_main.set_yticks(y_ticks)
    ax_main.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)
    ax_main.set_axisbelow(True)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def fit_psychfunc(stim_levels, n_trials, proportion):
    """Fit psychometric function (erf_psycho_2gammas). Returns [bias, threshold, lapselow, lapsehigh]."""
    if psy is None:
        raise ImportError("psychofit is required for plot_psychometric")
    stim_levels = np.asarray(stim_levels)
    n_trials = np.asarray(n_trials).flatten()
    proportion = np.asarray(proportion).flatten()
    assert stim_levels.shape[0] == n_trials.shape[0] == proportion.shape[0]
    if stim_levels.max() <= 1:
        stim_levels = stim_levels * 100
    pars, _ = psy.mle_fit_psycho(
        np.vstack((stim_levels, n_trials, proportion)),
        P_model="erf_psycho_2gammas",
        parstart=np.array([0, 20, 0.05, 0.05]),
        parmin=np.array([-100, 5, 0, 0]),
        parmax=np.array([100, 100, 1, 1]),
    )
    return pars


def plot_psychometric(trials, ax, label=None, **kwargs):
    """Plot psychometric curve on ax; returns slope (1 / (threshold * sqrt(2*pi)))."""
    if psy is None:
        raise ImportError("psychofit is required for plot_psychometric")
    from matplotlib import rcParams
    rcParams["legend.fontsize"] = 20
    trials = trials.copy()
    if trials["contrast"].max() <= 1:
        trials.loc[:, "contrast"] = (trials["contrast"] * 100).astype(int)
    stim_levels = np.sort(trials["contrast"].unique())
    grp = trials.groupby("contrast")
    n_trials = grp.size().reindex(stim_levels).fillna(0).values.astype(np.int64)
    proportion = grp["right_choice"].mean().reindex(stim_levels).values
    pars = fit_psychfunc(stim_levels, n_trials, proportion)
    sns.lineplot(
        x=np.arange(-27, 27),
        y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
        ax=ax, label=label, linewidth=3, **kwargs
    )
    sns.lineplot(
        x=np.arange(-36, -31),
        y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
        ax=ax, label=None, linewidth=3, **kwargs
    )
    sns.lineplot(
        x=np.arange(31, 36),
        y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
        ax=ax, label=None, linewidth=3, **kwargs
    )
    kwargs_pop = kwargs.copy()
    kwargs_pop.pop("linestyle", None)
    sns.lineplot(
        x=trials["contrast"],
        y=trials["right_choice"],
        ax=ax, label=None,
        err_style="bars", linewidth=0, linestyle="None", mew=0.5,
        marker="o", markersize=7, errorbar="se", err_kws={"elinewidth": 2},
        **kwargs_pop
    )
    ax.set(xticks=[-35, 0, 35], xlim=[-40, 40])
    ax.set_xticklabels(["-100", "0", "100"])
    ax.set(ylim=[0, 1.02], yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=15)
    ax.set_ylabel("Right choices", fontsize=18)
    ax.set_xlabel("Contrast (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=0)
    sns.despine(ax=ax)
    if label is not None:
        ax.legend(fontsize=18, frameon=False)
    slope = 1 / (pars[1] * np.sqrt(2 * np.pi))
    return slope
