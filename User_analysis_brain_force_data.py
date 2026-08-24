import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import lib
import Lib_grip as lb


############################
#### Directory settings ####
############################
PARTICIPANTS_DIRECTORY = Path(
    r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals"
)

PARTICIPANTS_FILE = PARTICIPANTS_DIRECTORY / "Participants.xlsx"

DATA_DIRECTORY = Path(
    r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen"
)

RESULTS_DIRECTORY = Path(
    r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"
)


###########################
#### Analysis settings ####
###########################
FORCE_FS = 100
LOW_PASS_FILTER_FREQUENCY = 15
TIME_THRESHOLD = 2
M = 2
R = 0.2

BRAIN_FILTER_LOW = 0.01
BRAIN_FILTER_HIGH = 0.30
BRAIN_FILTER_ORDER = 4
BRAIN_FILTER_PADDING_SECONDS = 60
BRAIN_ANALYSIS_PADDING_SECONDS = 30


################################
#### Brain-channel settings ####
################################
SHORT_CHANNEL_COLUMNS = [
    "[9322] Rx2 - Tx1  O2Hb",
    "[9322] Rx2 - Tx2  O2Hb",
    "[9322] Rx2 - Tx3  O2Hb",
    "[9323] Rx4 - Tx4  O2Hb",
    "[9323] Rx4 - Tx5  O2Hb",
    "[9323] Rx4 - Tx6  O2Hb",
    "[9322] Rx2 - Tx1  HHb",
    "[9322] Rx2 - Tx2  HHb",
    "[9322] Rx2 - Tx3  HHb",
    "[9323] Rx4 - Tx4  HHb",
    "[9323] Rx4 - Tx5  HHb",
    "[9323] Rx4 - Tx6  HHb"
]

LONG_CHANNELS = {
    "Left Tx1 O2Hb": "[9322] Rx1 - Tx1  O2Hb",
    "Left Tx2 O2Hb": "[9322] Rx1 - Tx2  O2Hb",
    "Left Tx3 O2Hb": "[9322] Rx1 - Tx3  O2Hb",
    "Left Tx1 HHb": "[9322] Rx1 - Tx1  HHb",
    "Left Tx2 HHb": "[9322] Rx1 - Tx2  HHb",
    "Left Tx3 HHb": "[9322] Rx1 - Tx3  HHb",
    "Right Tx4 O2Hb": "[9323] Rx3 - Tx4  O2Hb",
    "Right Tx5 O2Hb": "[9323] Rx3 - Tx5  O2Hb",
    "Right Tx6 O2Hb": "[9323] Rx3 - Tx6  O2Hb",
    "Right Tx4 HHb": "[9323] Rx3 - Tx4  HHb",
    "Right Tx5 HHb": "[9323] Rx3 - Tx5  HHb",
    "Right Tx6 HHb": "[9323] Rx3 - Tx6  HHb"
}

ALL_HEMOGLOBIN_COLUMNS = list(LONG_CHANNELS.values()) + SHORT_CHANNEL_COLUMNS


########################################
#### Brain-processing core functions ####
########################################
def remove_superficial_signal_from_long_channel(long_signal, short_channel_pca_components, fs, signal_name, plot=True):
    X = np.column_stack([np.ones(len(long_signal)), short_channel_pca_components])
    betas = np.linalg.lstsq(X, long_signal, rcond=None)[0]
    component_contributions = short_channel_pca_components * betas[1:]
    estimated_superficial_signal = np.sum(component_contributions, axis=1)
    cleaned_long_signal = long_signal - estimated_superficial_signal

    component_contribution_variance = np.var(component_contributions, axis=0)
    total_contribution_variance = np.sum(component_contribution_variance)

    if total_contribution_variance > 0:
        component_contribution_percentage = component_contribution_variance / total_contribution_variance * 100
    else:
        component_contribution_percentage = np.zeros(short_channel_pca_components.shape[1])

    if plot:
        analysis_time = np.arange(len(long_signal)) / fs
        component_labels = [f"PC{component_number}" for component_number in range(1, short_channel_pca_components.shape[1] + 1)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [2, 1.5]})

        ax1.plot(analysis_time, long_signal, color="black", linewidth=2, label="Original long signal")
        ax1.plot(analysis_time, estimated_superficial_signal, color="red", linewidth=2, label="Estimated superficial signal")
        ax1.plot(analysis_time, cleaned_long_signal, color="blue", linewidth=2, label="Cleaned long signal")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Hemoglobin signal")
        ax1.set_title(f"Short-channel regression – {signal_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        bars = ax2.barh(component_labels, component_contribution_percentage, color="steelblue")
        ax2.bar_label(bars, labels=[f"{percentage:.2f}%" for percentage in component_contribution_percentage], padding=3, fontsize=8)
        ax2.set_xlabel("Contribution to estimated superficial variance (%)")
        ax2.set_ylabel("PCA component")
        ax2.set_title("Contribution of each PCA component")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)

        maximum_percentage = np.max(component_contribution_percentage)

        if maximum_percentage > 0:
            ax2.set_xlim(0, maximum_percentage * 1.15)

        plt.tight_layout()
        plt.show()

    return cleaned_long_signal, estimated_superficial_signal, betas, component_contribution_percentage


def extract_cleaned_sets(cleaned_signal, baseline_start_indices, training_end_indices):
    cleaned_sets = []

    for baseline_start, training_end in zip(baseline_start_indices, training_end_indices):
        cleaned_set = cleaned_signal[int(baseline_start):int(training_end) + 1]
        cleaned_sets.append(cleaned_set)

    return np.array(cleaned_sets)


def baseline_correct_cleaned_sets(cleaned_sets, baseline_start_indices, training_start_indices):
    baseline_corrected_sets = []

    for set_index in range(len(cleaned_sets)):
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        baseline_mean = np.mean(cleaned_sets[set_index, :training_start_local])
        baseline_corrected_set = cleaned_sets[set_index] - baseline_mean
        baseline_corrected_sets.append(baseline_corrected_set)

    return np.array(baseline_corrected_sets)


def calculate_mean_training_change(baseline_corrected_sets, baseline_start_indices, training_start_indices, training_end_indices):
    mean_training_changes = []

    for set_index in range(len(baseline_corrected_sets)):
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        training_end_local = int(training_end_indices[set_index] - baseline_start_indices[set_index])
        training_signal = baseline_corrected_sets[set_index, training_start_local:training_end_local + 1]
        mean_training_change = np.mean(training_signal)
        mean_training_changes.append(mean_training_change)

    return np.array(mean_training_changes)


#########################################
#### Brain diagnostic plot functions ####
#########################################
def plot_short_channel_pca_results(pca, short_channel_matrix_standardized, short_channel_pca_components, fs, ID, variance_threshold=90):
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)
    number_of_components = short_channel_pca_components.shape[1]
    component_numbers = np.arange(1, number_of_components + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.bar(component_numbers, explained_variance)
    ax1.set_xticks(component_numbers)
    ax1.set_xlabel("PCA component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(f"Variance explained by each component – {ID}")

    ax2.plot(component_numbers, cumulative_variance, marker="o")
    ax2.axhline(variance_threshold, color="red", linestyle="--", label=f"{variance_threshold}%")
    ax2.set_xticks(component_numbers)
    ax2.set_xlabel("Number of PCA components")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    number_of_important_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    analysis_time = np.arange(short_channel_matrix_standardized.shape[0]) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(analysis_time, short_channel_matrix_standardized[:, :6], color="tab:blue", alpha=0.5)
    ax1.plot(analysis_time, short_channel_matrix_standardized[:, 6:], color="tab:orange", alpha=0.5)
    ax1.plot([], [], color="tab:blue", label="Short HbO signals")
    ax1.plot([], [], color="tab:orange", label="Short HHb signals")
    ax1.set_ylabel("Standardized value")
    ax1.set_title(f"Standardized short-channel signals – {ID}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for component_index in range(number_of_important_components):
        ax2.plot(analysis_time, short_channel_pca_components[:, component_index], label=f"PC{component_index + 1} ({explained_variance[component_index]:.1f}%)")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("PCA score")
    ax2.set_title(f"PCA components explaining {variance_threshold}% of the variance – {ID}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_all_long_channel_regressions(original_signals, superficial_signals, cleaned_signals, signal_names, fs, ID):
    fig, axes = plt.subplots(4, 3, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for signal_index, ax in enumerate(axes):
        time = np.arange(len(original_signals[signal_index])) / fs

        ax.plot(time, original_signals[signal_index], color="black", linewidth=1.5, label="Original")
        ax.plot(time, superficial_signals[signal_index], color="red", linewidth=1.5, label="Superficial")
        ax.plot(time, cleaned_signals[signal_index], color="blue", linewidth=1.5, label="Cleaned")
        ax.set_title(signal_names[signal_index])
        ax.grid(alpha=0.3)

        if signal_index % 3 == 0:
            ax.set_ylabel("Hemoglobin signal")

        if signal_index >= 9:
            ax.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"Short-channel regression for all long channels – {ID}", fontsize=16, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_short_channel_explained_variance(original_signals, cleaned_signals, signal_names, ID):
    explained_variance_percentages = []

    for original_signal, cleaned_signal in zip(original_signals, cleaned_signals):
        original_variance = np.var(original_signal)
        cleaned_variance = np.var(cleaned_signal)
        explained_percentage = (1 - cleaned_variance / original_variance) * 100
        explained_variance_percentages.append(explained_percentage)

    explained_variance_percentages = np.array(explained_variance_percentages)

    for signal_name, explained_percentage in zip(signal_names, explained_variance_percentages):
        print(f"{signal_name}: {explained_percentage:.2f}%")

    colors = ["tab:blue" if "O2Hb" in signal_name else "tab:orange" for signal_name in signal_names]
    plt.figure(figsize=(12, 7))
    bars = plt.barh(signal_names, explained_variance_percentages, color=colors)
    plt.bar_label(bars, labels=[f"{percentage:.1f}%" for percentage in explained_variance_percentages], padding=3)
    plt.xlabel("Long-signal variance explained by short channels (%)")
    plt.ylabel("Long channel")
    plt.title(f"Effect of short-channel regression – {ID}")
    plt.xlim(0, 105)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return explained_variance_percentages


def plot_cleaned_sets_with_events(cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, signal_name):
    global_minimum = np.min(cleaned_sets)
    global_maximum = np.max(cleaned_sets)
    y_range = global_maximum - global_minimum
    y_padding = y_range * 0.05
    y_minimum = global_minimum - y_padding
    y_maximum = global_maximum + y_padding

    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]

    for ax, set_index in zip(axes, plot_order):
        cleaned_set = cleaned_sets[set_index]
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        training_end_local = int(training_end_indices[set_index] - baseline_start_indices[set_index])
        set_time = np.arange(len(cleaned_set)) / fs
        baseline_mean = np.mean(cleaned_set[:training_start_local])

        ax.plot(set_time, cleaned_set, color="black", linewidth=1.5, label="Cleaned signal")
        ax.axvline(0, color="blue", linestyle="--", label="Baseline start")
        ax.axvline(training_start_local / fs, color="green", linestyle="--", label="Training start")
        ax.axvline(training_end_local / fs, color="red", linestyle="--", label="Training end")
        ax.axhline(baseline_mean, color="orange", linestyle="--", linewidth=2, label="Baseline average")
        ax.set_ylim(y_minimum, y_maximum)
        ax.set_xlim(-0.5, set_time[-1] + 0.5)
        ax.set_title(f"Set {set_index + 1}")
        ax.set_ylabel("Hemoglobin signal")
        ax.grid(alpha=0.3)

    axes[-2].set_xlabel("Time from baseline start (s)")
    axes[-1].set_xlabel("Time from baseline start (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(signal_name, fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_baseline_corrected_cleaned_sets(cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, signal_name):
    baseline_corrected_sets = baseline_correct_cleaned_sets(cleaned_sets, baseline_start_indices, training_start_indices)
    global_minimum = np.min(baseline_corrected_sets)
    global_maximum = np.max(baseline_corrected_sets)
    y_range = global_maximum - global_minimum
    y_padding = y_range * 0.05 if y_range > 0 else 1
    y_minimum = global_minimum - y_padding
    y_maximum = global_maximum + y_padding

    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]

    for ax, set_index in zip(axes, plot_order):
        baseline_corrected_set = baseline_corrected_sets[set_index]
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        training_end_local = int(training_end_indices[set_index] - baseline_start_indices[set_index])
        set_time = np.arange(len(baseline_corrected_set)) / fs

        ax.plot(set_time, baseline_corrected_set, color="black", linewidth=1.5, label="Baseline-corrected signal")
        ax.axvline(0, color="blue", linestyle="--", label="Baseline start")
        ax.axvline(training_start_local / fs, color="green", linestyle="--", label="Training start")
        ax.axvline(training_end_local / fs, color="red", linestyle="--", label="Training end")
        ax.axhline(0, color="orange", linestyle="--", linewidth=2, label="Baseline average")
        ax.set_ylim(y_minimum, y_maximum)
        ax.set_xlim(-0.5, set_time[-1] + 0.5)
        ax.set_title(f"Set {set_index + 1}")
        ax.set_ylabel("Change from baseline")
        ax.grid(alpha=0.3)

    axes[-2].set_xlabel("Time from baseline start (s)")
    axes[-1].set_xlabel("Time from baseline start (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(signal_name, fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    return baseline_corrected_sets


def plot_participant_mean_training_changes(mean_training_changes, signal_names, ID):
    set_numbers = np.arange(1, 11)
    O2Hb_indices = [0, 1, 2, 6, 7, 8]
    HHb_indices = [3, 4, 5, 9, 10, 11]
    O2Hb_minimum = np.min(mean_training_changes[O2Hb_indices])
    O2Hb_maximum = np.max(mean_training_changes[O2Hb_indices])
    HHb_minimum = np.min(mean_training_changes[HHb_indices])
    HHb_maximum = np.max(mean_training_changes[HHb_indices])
    O2Hb_range = O2Hb_maximum - O2Hb_minimum
    HHb_range = HHb_maximum - HHb_minimum
    O2Hb_padding = O2Hb_range * 0.05 if O2Hb_range > 0 else 1
    HHb_padding = HHb_range * 0.05 if HHb_range > 0 else 1

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    axes = axes.flatten()

    for signal_index, ax in enumerate(axes):
        color = "tab:blue" if "O2Hb" in signal_names[signal_index] else "tab:orange"
        ax.plot(set_numbers, mean_training_changes[signal_index], color=color, marker="o", linewidth=2)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

        if "O2Hb" in signal_names[signal_index]:
            ax.set_ylim(O2Hb_minimum - O2Hb_padding, O2Hb_maximum + O2Hb_padding)
        else:
            ax.set_ylim(HHb_minimum - HHb_padding, HHb_maximum + HHb_padding)

        ax.set_xticks(set_numbers)
        ax.set_title(signal_names[signal_index])
        ax.set_ylabel("Mean change from baseline")
        ax.grid(alpha=0.3)

        if signal_index >= 9:
            ax.set_xlabel("Training set")

    fig.suptitle(f"Mean hemodynamic change during training – {ID}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


####################################
#### Group-level plot functions ####
####################################
def plot_group_boxplots_by_set(mean_training_changes_df, signal_name):
    signal_data = mean_training_changes_df[mean_training_changes_df["Signal"] == signal_name]
    group_order = ["White", "Sine", "Pink"]
    group_colors = {"White": "lightgray", "Sine": "lightskyblue", "Pink": "hotpink"}
    group_offsets = {"White": -0.25, "Sine": 0, "Pink": 0.25}
    set_numbers = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(16, 7))
    random_generator = np.random.default_rng(10)

    for set_number in set_numbers:
        for group in group_order:
            values = signal_data.loc[(signal_data["Set"] == set_number) & (signal_data["Group"] == group), "Mean Training Change"].to_numpy()
            box_position = set_number + group_offsets[group]

            boxplot = ax.boxplot([values], positions=[box_position], widths=0.20, patch_artist=True, showfliers=False, manage_ticks=False)
            boxplot["boxes"][0].set_facecolor(group_colors[group])
            boxplot["boxes"][0].set_alpha(0.6)
            boxplot["medians"][0].set_color("black")
            boxplot["medians"][0].set_linewidth(2)

            jitter = random_generator.uniform(-0.05, 0.05, size=len(values))
            ax.scatter(np.full(len(values), box_position) + jitter, values, color=group_colors[group], edgecolor="black", s=45, alpha=0.9, zorder=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(set_numbers)
    ax.set_xticklabels([f"Set {set_number}" for set_number in set_numbers])
    ax.set_xlabel("Training set")
    ax.set_ylabel("Mean training change from baseline")
    ax.set_title(signal_name)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=group_colors["White"], alpha=0.6, label="White"),
        plt.Rectangle((0, 0), 1, 1, color=group_colors["Sine"], alpha=0.6, label="Sine"),
        plt.Rectangle((0, 0), 1, 1, color=group_colors["Pink"], alpha=0.6, label="Pink")
    ]

    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.show()

    return signal_data


def plot_group_boxplots_in_subplots(mean_training_changes_df, signal_name, show_data_points=True, connect_participants=False, connect_group_mean=True):
    signal_data = mean_training_changes_df[mean_training_changes_df["Signal"] == signal_name]

    if signal_data.empty:
        raise ValueError(f"No data found for: {signal_name}")

    group_order = ["White", "Sine", "Pink"]
    group_colors = {"White": "#7A7A7A", "Sine": "#2F7FBF", "Pink": "#E75480"}
    set_numbers = np.arange(1, 11)
    global_minimum = signal_data["Mean Training Change"].min()
    global_maximum = signal_data["Mean Training Change"].max()
    y_range = global_maximum - global_minimum
    y_padding = y_range * 0.05 if y_range > 0 else 0.1
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
    random_generator = np.random.default_rng(10)
    legend_handles = []

    for ax, group in zip(axes, group_order):
        group_data = signal_data[signal_data["Group"] == group]
        boxplot_values = []

        for set_number in set_numbers:
            values = group_data.loc[group_data["Set"] == set_number, "Mean Training Change"].to_numpy()
            values = values[np.isfinite(values)]

            if len(values) == 0:
                values = np.array([np.nan])

            boxplot_values.append(values)

        boxplot = ax.boxplot(boxplot_values, positions=set_numbers, widths=0.55, patch_artist=True, showfliers=False, manage_ticks=False)

        for box in boxplot["boxes"]:
            box.set_facecolor(group_colors[group])
            box.set_alpha(0.35)

        for median in boxplot["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        participant_IDs = group_data["ID"].unique()
        participant_jitter = {participant_ID: random_generator.uniform(-0.10, 0.10) for participant_ID in participant_IDs}

        for participant_ID in participant_IDs:
            participant_data = group_data[group_data["ID"] == participant_ID]
            participant_data = participant_data.set_index("Set").reindex(set_numbers)
            participant_values = participant_data["Mean Training Change"].to_numpy()
            x_positions = set_numbers + participant_jitter[participant_ID]
            valid_values = np.isfinite(participant_values)

            if connect_participants:
                ax.plot(x_positions[valid_values], participant_values[valid_values], color=group_colors[group], linewidth=0.7, alpha=0.35, zorder=2)

            if show_data_points:
                ax.scatter(x_positions[valid_values], participant_values[valid_values], color=group_colors[group], edgecolor="black", linewidth=0.5, s=40, alpha=0.85, zorder=3)

        if connect_group_mean:
            group_means = group_data.groupby("Set")["Mean Training Change"].mean().reindex(set_numbers)
            ax.plot(set_numbers, group_means.to_numpy(), color="black", linewidth=3.5, marker="o", markerfacecolor=group_colors[group], markeredgecolor="black", markersize=7, zorder=4)

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(set_numbers)
        ax.set_xlabel("Training set")
        ax.set_title(f"{group} (n = {len(participant_IDs)})")
        ax.set_ylim(global_minimum - y_padding, global_maximum + y_padding)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean training change from baseline")

    if show_data_points:
        legend_handles.append(axes[0].scatter([], [], color="gray", edgecolor="black", linewidth=0.5, s=40, label="Participant"))

    if connect_participants:
        legend_handles.append(axes[0].plot([], [], color="gray", linewidth=0.7, alpha=0.5, label="Participant progression")[0])

    if connect_group_mean:
        legend_handles.append(axes[0].plot([], [], color="black", linewidth=3.5, marker="o", label="Group mean")[0])

    fig.suptitle(signal_name, fontsize=16, y=0.99)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(legend_handles))

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.show()


def plot_hemoglobin_and_force_by_group(mean_training_changes_df, spatial_error_df, hemoglobin_signal, spatial_variable, show_ID_names=True):
    hemoglobin_data = mean_training_changes_df[mean_training_changes_df["Signal"] == hemoglobin_signal].copy()
    spatial_error_data = spatial_error_df.copy()
    hemoglobin_data["ID"] = hemoglobin_data["ID"].astype(str)
    spatial_error_data["ID"] = spatial_error_data["ID"].astype(str)

    spatial_columns = [f"{spatial_variable} Training {set_number}" for set_number in range(1, 11)]
    missing_columns = [column for column in spatial_columns if column not in spatial_error_data.columns]

    if missing_columns:
        raise ValueError(f"These columns were not found: {missing_columns}")

    common_IDs = sorted(set(hemoglobin_data["ID"]).intersection(set(spatial_error_data["ID"])))

    if len(common_IDs) == 0:
        raise ValueError("No matching participant IDs were found between the two dataframes.")

    group_order = ["White", "Sine", "Pink"]
    set_numbers = np.arange(1, 11)
    participant_colors = plt.cm.tab20(np.linspace(0, 1, len(common_IDs)))
    color_by_ID = dict(zip(common_IDs, participant_colors))
    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=True, sharey="row")
    participant_artists = {participant_ID: [] for participant_ID in common_IDs}
    legend_handles = []
    legend_IDs = []

    for column_index, group in enumerate(group_order):
        hemoglobin_axis = axes[0, column_index]
        spatial_axis = axes[1, column_index]
        group_hemoglobin_data = hemoglobin_data[hemoglobin_data["Group"] == group]
        group_IDs = [participant_ID for participant_ID in common_IDs if participant_ID in group_hemoglobin_data["ID"].values]

        for participant_ID in group_IDs:
            participant_color = color_by_ID[participant_ID]
            participant_hemoglobin_data = group_hemoglobin_data[group_hemoglobin_data["ID"] == participant_ID]
            participant_hemoglobin_data = participant_hemoglobin_data.set_index("Set").reindex(set_numbers)
            hemoglobin_values = participant_hemoglobin_data["Mean Training Change"].to_numpy(dtype=float)

            hemoglobin_line = hemoglobin_axis.plot(set_numbers, hemoglobin_values, marker="o", markersize=5, linewidth=1.2, color=participant_color, label=participant_ID)[0]
            participant_artists[participant_ID].append(hemoglobin_line)

            participant_spatial_data = spatial_error_data[spatial_error_data["ID"] == participant_ID]
            spatial_values = participant_spatial_data[spatial_columns].iloc[0].to_numpy(dtype=float)
            spatial_line = spatial_axis.plot(set_numbers, spatial_values, marker="o", markersize=5, linewidth=1.2, color=participant_color, label=participant_ID)[0]
            participant_artists[participant_ID].append(spatial_line)

            if show_ID_names:
                valid_hemoglobin = np.where(np.isfinite(hemoglobin_values))[0]

                if len(valid_hemoglobin) > 0:
                    final_index = valid_hemoglobin[-1]
                    hemoglobin_text = hemoglobin_axis.annotate(participant_ID, (set_numbers[final_index], hemoglobin_values[final_index]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=participant_color, va="center")
                    participant_artists[participant_ID].append(hemoglobin_text)

                valid_spatial = np.where(np.isfinite(spatial_values))[0]

                if len(valid_spatial) > 0:
                    final_index = valid_spatial[-1]
                    spatial_text = spatial_axis.annotate(participant_ID, (set_numbers[final_index], spatial_values[final_index]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=participant_color, va="center")
                    participant_artists[participant_ID].append(spatial_text)

            legend_handles.append(hemoglobin_line)
            legend_IDs.append(participant_ID)

        hemoglobin_axis.axhline(0, color="black", linestyle="--", linewidth=1)
        hemoglobin_axis.set_title(group, fontsize=14)
        spatial_axis.set_xlabel("Training set")
        hemoglobin_axis.set_xlim(0.7, 11)
        spatial_axis.set_xlim(0.7, 11)
        hemoglobin_axis.set_xticks(set_numbers)
        spatial_axis.set_xticks(set_numbers)
        hemoglobin_axis.grid(alpha=0.3)
        spatial_axis.grid(alpha=0.3)

    axes[0, 0].set_ylabel(f"{hemoglobin_signal}\nMean change from baseline")
    axes[1, 0].set_ylabel(spatial_variable)

    unique_legend_handles = []
    unique_legend_IDs = []

    for participant_ID, legend_handle in zip(legend_IDs, legend_handles):
        if participant_ID not in unique_legend_IDs:
            unique_legend_IDs.append(participant_ID)
            unique_legend_handles.append(legend_handle)

    legend = fig.legend(handles=unique_legend_handles, labels=unique_legend_IDs, title="Click ID to show/hide", loc="center left", bbox_to_anchor=(0.84, 0.5), ncol=1)
    legend_artist_to_ID = {}
    legend_entries = {}

    for legend_line, legend_text, participant_ID in zip(legend.get_lines(), legend.get_texts(), unique_legend_IDs):
        legend_line.set_picker(5)
        legend_text.set_picker(True)
        legend_artist_to_ID[legend_line] = participant_ID
        legend_artist_to_ID[legend_text] = participant_ID
        legend_entries[participant_ID] = (legend_line, legend_text)

    def toggle_participant(event):
        selected_artist = event.artist

        if selected_artist not in legend_artist_to_ID:
            return

        participant_ID = legend_artist_to_ID[selected_artist]
        artists = participant_artists[participant_ID]
        new_visibility = not artists[0].get_visible()

        for artist in artists:
            artist.set_visible(new_visibility)

        legend_line, legend_text = legend_entries[participant_ID]
        legend_line.set_alpha(1.0 if new_visibility else 0.2)
        legend_text.set_alpha(1.0 if new_visibility else 0.2)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", toggle_participant)
    fig.suptitle(f"{hemoglobin_signal} and {spatial_variable}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.83, 0.96])
    plt.show()


def process_brain_participant(ID, group, participant_folder):
    brain_folder = participant_folder / "Brain data"
    name = "Artinis_" + ID[0] + ID.split("_")[1]

    os.chdir(brain_folder)
    data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_10_events_plot(str(brain_folder), name, write_manual_events_to_excel=False, plot=False)

    participant_brain_data = {
        "Group": group,
        "Sampling Frequency": fs,
        "Raw Data": data,
        "Event Indices": list_indices,
        "Event Times": list_time_events,
        "Pre Event Indices": pre_event_indices,
        "Derived End Indices": derived_end_indices,
        "Final Event Indices": final_event_indices,
        "Raw Training Sets": list_training_sets
    }

    filter_start_index = max(0, int(list_indices[0] - BRAIN_FILTER_PADDING_SECONDS * fs))
    filter_end_index = min(data.height, int(derived_end_indices[-1] + BRAIN_FILTER_PADDING_SECONDS * fs + 1))
    number_of_rows = filter_end_index - filter_start_index
    data_to_filter = data.slice(filter_start_index, number_of_rows)

    filtered_signals = {}

    for column_name in ALL_HEMOGLOBIN_COLUMNS:
        signal = data_to_filter[column_name].to_numpy()
        filtered_signals[column_name] = lb.butter_bandpass_filtfilt_SOS(signal, fs, low=BRAIN_FILTER_LOW, high=BRAIN_FILTER_HIGH, order=BRAIN_FILTER_ORDER, plot=False, demean=False)

    analysis_start_index = max(0, int(list_indices[0] - BRAIN_ANALYSIS_PADDING_SECONDS * fs))
    number_of_indices_cut = analysis_start_index
    analysis_end_index = min(data.height, int(derived_end_indices[-1] + 1))
    analysis_start_index_filtered = analysis_start_index - filter_start_index
    analysis_end_index_filtered = analysis_end_index - filter_start_index
    analysis_slice = slice(analysis_start_index_filtered, analysis_end_index_filtered)

    short_channel_matrix = np.column_stack([
        filtered_signals[column_name][analysis_slice]
        for column_name in SHORT_CHANNEL_COLUMNS
    ])

    scaler = StandardScaler()
    short_channel_matrix_standardized = scaler.fit_transform(short_channel_matrix)

    pca = PCA(n_components=12)
    short_channel_pca_components = pca.fit_transform(short_channel_matrix_standardized)

    original_long_signals = []
    superficial_long_signals = []
    cleaned_long_signals = []

    for signal_name, column_name in LONG_CHANNELS.items():
        original_long_signal = filtered_signals[column_name][analysis_slice]

        cleaned_long_signal, estimated_superficial_signal, betas, component_contribution_percentage = remove_superficial_signal_from_long_channel(
            original_long_signal,
            short_channel_pca_components,
            fs,
            signal_name,
            plot=False
        )

        original_long_signals.append(original_long_signal)
        superficial_long_signals.append(estimated_superficial_signal)
        cleaned_long_signals.append(cleaned_long_signal)

    baseline_start_indices = np.array(pre_event_indices) - number_of_indices_cut
    training_start_indices = np.array(list_indices) - number_of_indices_cut
    training_end_indices = np.array(derived_end_indices) - number_of_indices_cut

    cleaned_sets_list = []
    baseline_corrected_sets_list = []
    mean_training_changes_list = []

    for cleaned_long_signal in cleaned_long_signals:
        cleaned_sets = extract_cleaned_sets(cleaned_long_signal, baseline_start_indices, training_end_indices)
        baseline_corrected_sets = baseline_correct_cleaned_sets(cleaned_sets, baseline_start_indices, training_start_indices)
        mean_training_changes = calculate_mean_training_change(baseline_corrected_sets, baseline_start_indices, training_start_indices, training_end_indices)

        cleaned_sets_list.append(cleaned_sets)
        baseline_corrected_sets_list.append(baseline_corrected_sets)
        mean_training_changes_list.append(mean_training_changes)

    print(cleaned_sets_list[0].shape)

    long_signal_names = list(LONG_CHANNELS.keys())

    participant_brain_data["Cleaned Training Sets"] = dict(zip(long_signal_names, cleaned_sets_list))
    participant_brain_data["Baseline Corrected Training Sets"] = dict(zip(long_signal_names, baseline_corrected_sets_list))
    participant_brain_data["Mean Training Changes"] = dict(zip(long_signal_names, mean_training_changes_list))

    mean_training_change_rows = []

    for signal_name, mean_training_changes in zip(long_signal_names, mean_training_changes_list):
        for set_number, mean_training_change in enumerate(mean_training_changes, start=1):
            mean_training_change_rows.append({
                "ID": ID,
                "Group": group,
                "Signal": signal_name,
                "Set": set_number,
                "Mean Training Change": mean_training_change
            })

    return participant_brain_data, mean_training_change_rows


########################################
#### Force-processing core functions ####
########################################
def read_force_training_sets(grip_folder):
    training_signals = []

    for training_set in range(1, 11):
        training_file = grip_folder / f"Training_{training_set}.csv"
        training_signal = pd.read_csv(training_file, skiprows=2)
        training_signals.append(training_signal)

    return training_signals


def process_force_participant(ID, group, participant_folder, participants):
    MVC = participants.loc[participants["ID"] == ID, participants.columns[6]].iloc[0]
    print(MVC)

    grip_folder = participant_folder / "Grip data"
    training_signals = read_force_training_sets(grip_folder)

    for training_signal in training_signals:
        training_signal["Performance"] = lib.Butterworth(FORCE_FS, LOW_PASS_FILTER_FREQUENCY, training_signal["Performance"])

    synchronized_training_signals = [
        lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_signal)
        for training_signal in training_signals
    ]

    synchronized_training_signals_for_analysis = [
        synchronized_training_signal[synchronized_training_signal["Time"] > TIME_THRESHOLD].reset_index(drop=True).copy()
        for synchronized_training_signal in synchronized_training_signals
    ]

    participant_results = {
        "ID": ID,
        "Group": group
    }

    for trial, (synchronized_training_signal, training_signal) in enumerate(zip(synchronized_training_signals_for_analysis, training_signals), start=1):
        spatial_error = lb.spatial_error(synchronized_training_signal)
        average_spatial_error = np.mean(spatial_error)
        variable_error = np.std(spatial_error)
        relative_to_MVC_average_spatial_error = (average_spatial_error / MVC) * 100
        relative_to_MVC_variable_error = (variable_error / MVC) * 100

        AMI = lib.AMI_Stergiou(training_signal["Performance"].to_numpy(), 10, FORCE_FS, n_bins=0, plot=False)
        SaEn = lib.SaEn_once_again(training_signal["Performance"].to_numpy(), M, R, AMI[0][0][0], return_probabilities=False)

        lag, max_correlation = lib.Cross_correlation(synchronized_training_signal["Target"], synchronized_training_signal["Performance"], plot=False)

        participant_results[f"Average Spatial Error Training {trial}"] = average_spatial_error
        participant_results[f"Variable Error Training {trial}"] = variable_error
        participant_results[f"Normalized Average Spatial Error Training {trial}"] = relative_to_MVC_average_spatial_error
        participant_results[f"Normalized Variable Error Training {trial}"] = relative_to_MVC_variable_error
        participant_results[f"SaEn Training {trial}"] = SaEn
        participant_results[f"Sample Lag Training {trial}"] = lag
        participant_results[f"Max Correlation Training {trial}"] = max_correlation

    return participant_results


################################
#### Complete-set time plot ####
################################
def plot_complete_training_set(ID, training_set, hemoglobin_signal, hemoglobin_version):
    if ID not in brain_data:
        raise ValueError(f"{ID} was not found in brain_data.")

    if training_set < 1 or training_set > 10:
        raise ValueError("training_set must be between 1 and 10.")

    hemoglobin_version_clean = hemoglobin_version.lower().replace("_", "-").replace(" ", "-")

    if hemoglobin_version_clean in ["cleaned", "clean"]:
        hemoglobin_dictionary_name = "Cleaned Training Sets"
        hemoglobin_version_title = "Cleaned"
    elif hemoglobin_version_clean in ["baseline-corrected", "baseline-correction"]:
        hemoglobin_dictionary_name = "Baseline Corrected Training Sets"
        hemoglobin_version_title = "Baseline-corrected"
    else:
        raise ValueError("hemoglobin_version must be 'cleaned' or 'baseline-corrected'.")

    available_signals = brain_data[ID][hemoglobin_dictionary_name].keys()

    if hemoglobin_signal not in available_signals:
        raise ValueError(f"{hemoglobin_signal} was not found. Available signals: {list(available_signals)}")

    set_index = training_set - 1
    hemoglobin_set = brain_data[ID][hemoglobin_dictionary_name][hemoglobin_signal][set_index]
    baseline_start_index = brain_data[ID]["Pre Event Indices"][set_index]
    training_start_index = brain_data[ID]["Event Indices"][set_index]
    training_start_in_set = int(training_start_index - baseline_start_index)
    hemoglobin_fs = brain_data[ID]["Sampling Frequency"]
    hemoglobin_time = (np.arange(len(hemoglobin_set)) - training_start_in_set) / hemoglobin_fs
    hemoglobin_time_mask = (hemoglobin_time >= -5) & (hemoglobin_time <= 30)

    grip_file = DATA_DIRECTORY / ID / "Grip data" / f"Training_{training_set}.csv"
    training_signal = pd.read_csv(grip_file, skiprows=2)
    training_signal["Performance"] = lib.Butterworth(FORCE_FS, LOW_PASS_FILTER_FREQUENCY, training_signal["Performance"])
    synchronized_training_signal = lb.synchronization_of_Time_and_ClosestSampleTime_Anestis(training_signal)
    spatial_error = lb.spatial_error(synchronized_training_signal)

    force_time = synchronized_training_signal["Time"].to_numpy(dtype=float)
    performance = synchronized_training_signal["Performance"].to_numpy(dtype=float)
    target = synchronized_training_signal["Target"].to_numpy(dtype=float)
    spatial_error = np.asarray(spatial_error, dtype=float)

    number_of_force_samples = min(len(force_time), len(performance), len(target), len(spatial_error))

    force_time = force_time[:number_of_force_samples]
    performance = performance[:number_of_force_samples]
    target = target[:number_of_force_samples]
    spatial_error = spatial_error[:number_of_force_samples]

    force_time = force_time - force_time[0]
    force_time_mask = (force_time >= 0) & (force_time <= 30)

    force_time = force_time[force_time_mask]
    performance = performance[force_time_mask]
    target = target[force_time_mask]
    spatial_error = spatial_error[force_time_mask]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    axes[0].plot(hemoglobin_time[hemoglobin_time_mask], hemoglobin_set[hemoglobin_time_mask], color="tab:red", linewidth=2, label=f"{hemoglobin_signal} — {hemoglobin_version_title}")

    if hemoglobin_version_title == "Baseline-corrected":
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=1, label="Baseline average")

    axes[0].set_ylabel("Hemoglobin change")
    axes[0].set_title("Hemoglobin activity")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(force_time, spatial_error, color="tab:orange", linewidth=1.5)
    axes[1].set_ylabel("Spatial error")
    axes[1].set_title("Spatial error")
    axes[1].grid(alpha=0.3)

    axes[2].plot(force_time, performance, color="tab:blue", linewidth=1.5, label="Force output")
    axes[2].plot(force_time, target, color="black", linewidth=2, linestyle="--", label="Target")
    axes[2].set_xlabel("Time from training start (s)")
    axes[2].set_ylabel("Force")
    axes[2].set_title("Force output and target")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    for axis in axes:
        axis.axvline(0, color="green", linestyle="--", linewidth=1.5)
        axis.set_xlim(-5, 30)
        axis.set_xticks(np.arange(-5, 31, 5))

    fig.suptitle(f"{ID} — Training set {training_set}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_hemoglobin_window_averages(brain_data, training_set, hemoglobin_signal, show_connected_lines=True, show_ID=True):
    if training_set < 1 or training_set > 10:
        raise ValueError("training_set must be between 1 and 10.")

    set_index = training_set - 1
    group_order = ["White", "Sine", "Pink"]
    window_numbers = np.array([1, 2, 3])
    window_labels = ["First window\n0–10 s", "Second window\n10–20 s", "Third window\n20–30 s"]

    window_average_results = []

    for participant_ID, participant_data in brain_data.items():
        if "Baseline Corrected Training Sets" not in participant_data:
            continue

        if hemoglobin_signal not in participant_data["Baseline Corrected Training Sets"]:
            continue

        fs = participant_data["Sampling Frequency"]
        baseline_corrected_sets = participant_data["Baseline Corrected Training Sets"][hemoglobin_signal]
        selected_set = baseline_corrected_sets[set_index]

        baseline_start_index = participant_data["Pre Event Indices"][set_index]
        training_start_index = participant_data["Event Indices"][set_index]
        training_end_index = participant_data["Derived End Indices"][set_index]

        training_start_local = int(training_start_index - baseline_start_index)
        training_end_local = int(training_end_index - baseline_start_index)

        training_signal = selected_set[training_start_local:training_end_local]

        window_length = int(10 * fs)

        first_window = training_signal[0:window_length]
        second_window = training_signal[window_length:2 * window_length]
        third_window = training_signal[2 * window_length:3 * window_length]

        window_averages = [
            np.mean(first_window),
            np.mean(second_window),
            np.mean(third_window)
        ]

        for window_number, window_average in zip(window_numbers, window_averages):
            window_average_results.append({
                "ID": participant_ID,
                "Group": participant_data["Group"],
                "Signal": hemoglobin_signal,
                "Set": training_set,
                "Window": window_number,
                "Window Average": window_average
            })

    window_average_df = pd.DataFrame(window_average_results)

    if window_average_df.empty:
        raise ValueError("No matching hemoglobin data were found.")

    participant_IDs = sorted(window_average_df["ID"].unique())
    participant_colors = plt.cm.tab20(np.linspace(0, 1, len(participant_IDs)))
    color_by_ID = dict(zip(participant_IDs, participant_colors))

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, sharey=True)

    participant_artists = {participant_ID: [] for participant_ID in participant_IDs}
    legend_handles = []
    legend_IDs = []

    for ax, group in zip(axes, group_order):
        group_data = window_average_df[window_average_df["Group"] == group]
        group_IDs = sorted(group_data["ID"].unique())

        for participant_ID in group_IDs:
            participant_data = group_data[group_data["ID"] == participant_ID]
            participant_data = participant_data.set_index("Window").reindex(window_numbers)

            window_averages = participant_data["Window Average"].to_numpy(dtype=float)

            if show_connected_lines:
                line_style = "-"
            else:
                line_style = "None"

            participant_line = ax.plot(window_numbers, window_averages, marker="o", markersize=7, linestyle=line_style, linewidth=1.2, color=color_by_ID[participant_ID], label=participant_ID)[0]

            participant_artists[participant_ID].append(participant_line)

            if show_ID:
                participant_text = ax.annotate(participant_ID, (window_numbers[-1], window_averages[-1]), xytext=(7, 0), textcoords="offset points", fontsize=8, color=color_by_ID[participant_ID], va="center")

                participant_artists[participant_ID].append(participant_text)

            legend_handles.append(participant_line)
            legend_IDs.append(participant_ID)

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Average hemoglobin\nchange from baseline")
        ax.set_title(f"{group} group")
        ax.set_xlim(0.8, 3.25)
        ax.grid(alpha=0.3)

    axes[-1].set_xticks(window_numbers)
    axes[-1].set_xticklabels(window_labels)
    axes[-1].set_xlabel("Training-set window")

    legend = fig.legend(handles=legend_handles, labels=legend_IDs, title="Click ID to show/hide", loc="center left", bbox_to_anchor=(0.83, 0.5), ncol=1)

    legend_artist_to_ID = {}
    legend_entries = {}

    for legend_line, legend_text, participant_ID in zip(legend.get_lines(), legend.get_texts(), legend_IDs):
        legend_line.set_picker(5)
        legend_text.set_picker(True)

        legend_artist_to_ID[legend_line] = participant_ID
        legend_artist_to_ID[legend_text] = participant_ID
        legend_entries[participant_ID] = (legend_line, legend_text)

    def toggle_participant(event):
        selected_artist = event.artist

        if selected_artist not in legend_artist_to_ID:
            return

        participant_ID = legend_artist_to_ID[selected_artist]
        artists = participant_artists[participant_ID]
        new_visibility = not artists[0].get_visible()

        for artist in artists:
            artist.set_visible(new_visibility)

        legend_line, legend_text = legend_entries[participant_ID]

        if new_visibility:
            legend_line.set_alpha(1.0)
            legend_text.set_alpha(1.0)
        else:
            legend_line.set_alpha(0.2)
            legend_text.set_alpha(0.2)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", toggle_participant)

    fig.suptitle(f"{hemoglobin_signal} – Training set {training_set}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.82, 0.97])
    plt.show()

    return window_average_df

def plot_late_minus_early_by_group(hemoglobin_signal, show_data_points=True, show_connected_lines=True, show_ID=True):
    group_order = ["White", "Sine", "Pink"]

    group_colors = {
        "White": "gray",
        "Sine": "tab:blue",
        "Pink": "hotpink"
    }

    set_numbers = np.arange(1, 11)
    late_change_results = []

    for participant_ID, participant_data in brain_data.items():
        if "Baseline Corrected Training Sets" not in participant_data:
            continue

        if hemoglobin_signal not in participant_data["Baseline Corrected Training Sets"]:
            continue

        fs = participant_data["Sampling Frequency"]
        baseline_corrected_sets = participant_data["Baseline Corrected Training Sets"][hemoglobin_signal]

        for set_index in range(10):
            selected_set = baseline_corrected_sets[set_index]

            baseline_start_index = participant_data["Pre Event Indices"][set_index]
            training_start_index = participant_data["Event Indices"][set_index]
            training_end_index = participant_data["Derived End Indices"][set_index]

            training_start_local = int(training_start_index - baseline_start_index)
            training_end_local = int(training_end_index - baseline_start_index)

            training_signal = selected_set[training_start_local:training_end_local]

            window_length = int(10 * fs)

            early_window = training_signal[0:window_length]
            late_window = training_signal[2 * window_length:3 * window_length]

            early_average = np.mean(early_window)
            late_average = np.mean(late_window)
            late_change = late_average - early_average

            late_change_results.append({
                "ID": participant_ID,
                "Group": participant_data["Group"],
                "Signal": hemoglobin_signal,
                "Set": set_index + 1,
                "Early Average": early_average,
                "Late Average": late_average,
                "Late Change": late_change
            })

    late_change_df = pd.DataFrame(late_change_results)

    if late_change_df.empty:
        raise ValueError(f"No data found for {hemoglobin_signal}.")

    global_minimum = late_change_df["Late Change"].min()
    global_maximum = late_change_df["Late Change"].max()

    y_range = global_maximum - global_minimum
    y_padding = y_range * 0.05 if y_range > 0 else 0.1

    participant_IDs = sorted(late_change_df["ID"].unique())
    participant_colors = plt.cm.tab20(np.linspace(0, 1, len(participant_IDs)))
    color_by_ID = dict(zip(participant_IDs, participant_colors))

    random_generator = np.random.default_rng(10)

    participant_jitter = {
        participant_ID: random_generator.uniform(-0.10, 0.10)
        for participant_ID in participant_IDs
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=True, sharey=True)

    for ax, group in zip(axes, group_order):
        group_data = late_change_df[late_change_df["Group"] == group]
        group_IDs = sorted(group_data["ID"].unique())

        boxplot_values = []

        for set_number in set_numbers:
            values = group_data.loc[group_data["Set"] == set_number, "Late Change"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                values = np.array([np.nan])

            boxplot_values.append(values)

        boxplot = ax.boxplot(boxplot_values, positions=set_numbers, widths=0.55, patch_artist=True, showfliers=False, manage_ticks=False)

        for box in boxplot["boxes"]:
            box.set_facecolor(group_colors[group])
            box.set_alpha(0.35)

        for median in boxplot["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        for participant_ID in group_IDs:
            participant_data = group_data[group_data["ID"] == participant_ID]
            participant_data = participant_data.set_index("Set").reindex(set_numbers)

            participant_values = participant_data["Late Change"].to_numpy(dtype=float)
            valid_values = np.isfinite(participant_values)
            x_positions = set_numbers + participant_jitter[participant_ID]

            if show_connected_lines:
                ax.plot(x_positions[valid_values], participant_values[valid_values], color=color_by_ID[participant_ID], linewidth=0.8, alpha=0.7, zorder=2)

            if show_data_points:
                ax.scatter(x_positions[valid_values], participant_values[valid_values], color=color_by_ID[participant_ID], edgecolor="black", linewidth=0.5, s=45, zorder=3)

            if show_ID:
                valid_indices = np.where(valid_values)[0]

                if len(valid_indices) > 0:
                    final_index = valid_indices[-1]

                    ax.annotate(participant_ID, (x_positions[final_index], participant_values[final_index]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=color_by_ID[participant_ID], va="center")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(set_numbers)
        ax.set_xlabel("Training set")
        ax.set_title(f"{group} group (n = {len(group_IDs)})")
        ax.set_xlim(0.5, 11)
        ax.set_ylim(global_minimum - y_padding, global_maximum + y_padding)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Late change: Late average − Early average")

    fig.suptitle(f"Within-set hemoglobin change – {hemoglobin_signal}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return late_change_df

def plot_three_training_windows_by_group(brain_data, hemoglobin_signal, show_data_points=True, show_connected_lines=True, show_ID=True):
    group_order = ["White", "Sine", "Pink"]

    window_names = ["0–10 s", "10–20 s", "20–30 s"]

    window_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green"
    ]

    window_offsets = [
        -0.25,
        0,
        0.25
    ]

    set_numbers = np.arange(1, 11)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True, sharey=True)

    for group_index, group in enumerate(group_order):
        ax = axes[group_index]

        group_participants = [
            participant_ID
            for participant_ID in brain_data
            if brain_data[participant_ID]["Group"] == group
        ]

        participant_colors = plt.cm.tab10(
            np.linspace(0, 1, len(group_participants))
        )

        color_by_ID = dict(
            zip(group_participants, participant_colors)
        )

        participant_window_values = {}

        for participant_ID in group_participants:
            fs = brain_data[participant_ID]["Sampling Frequency"]

            participant_sets = brain_data[participant_ID]["Baseline Corrected Training Sets"][hemoglobin_signal]

            participant_window_values[participant_ID] = {
                "0–10 s": [],
                "10–20 s": [],
                "20–30 s": []
            }

            for set_index in range(10):
                current_set = participant_sets[set_index]

                training_start_sample = int(10 * fs)

                early_start = training_start_sample
                early_end = training_start_sample + int(10 * fs)

                middle_start = early_end
                middle_end = middle_start + int(10 * fs)

                late_start = middle_end
                late_end = late_start + int(10 * fs)

                early_average = np.mean(
                    current_set[early_start:early_end]
                )

                middle_average = np.mean(
                    current_set[middle_start:middle_end]
                )

                late_average = np.mean(
                    current_set[late_start:late_end]
                )

                participant_window_values[participant_ID]["0–10 s"].append(
                    early_average
                )

                participant_window_values[participant_ID]["10–20 s"].append(
                    middle_average
                )

                participant_window_values[participant_ID]["20–30 s"].append(
                    late_average
                )

        for set_index in range(10):
            window_values = []

            for window_name in window_names:
                values_for_current_box = [
                    participant_window_values[participant_ID][window_name][set_index]
                    for participant_ID in group_participants
                ]

                window_values.append(
                    values_for_current_box
                )

            box_positions = [
                set_numbers[set_index] + window_offsets[0],
                set_numbers[set_index] + window_offsets[1],
                set_numbers[set_index] + window_offsets[2]
            ]

            boxplot = ax.boxplot(window_values, positions=box_positions, widths=0.20, patch_artist=True, showfliers=False)

            for box, color in zip(boxplot["boxes"], window_colors):
                box.set_facecolor(color)
                box.set_alpha(0.55)

            for median in boxplot["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

        for participant_ID in group_participants:
            participant_color = color_by_ID[participant_ID]

            for set_index in range(10):
                participant_positions = [
                    set_numbers[set_index] + window_offsets[0],
                    set_numbers[set_index] + window_offsets[1],
                    set_numbers[set_index] + window_offsets[2]
                ]

                participant_values = [
                    participant_window_values[participant_ID]["0–10 s"][set_index],
                    participant_window_values[participant_ID]["10–20 s"][set_index],
                    participant_window_values[participant_ID]["20–30 s"][set_index]
                ]

                if show_connected_lines:
                    ax.plot(
                        participant_positions,
                        participant_values,
                        color=participant_color,
                        linewidth=0.8,
                        alpha=0.6,
                        zorder=2
                    )

                if show_data_points:
                    ax.scatter(
                        participant_positions,
                        participant_values,
                        color=participant_color,
                        edgecolor="black",
                        linewidth=0.3,
                        s=24,
                        alpha=0.8,
                        zorder=3
                    )

            if show_ID:
                final_late_value = participant_window_values[participant_ID]["20–30 s"][-1]

                ax.annotate(
                    participant_ID,
                    (
                        set_numbers[-1] + window_offsets[2],
                        final_late_value
                    ),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=participant_color,
                    va="center"
                )

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        ax.set_ylabel(
            "Average change\nfrom baseline"
        )

        ax.set_title(
            f"{group} group",
            fontsize=14
        )

        ax.set_xlim(
            0.5,
            11
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

    axes[-1].set_xticks(set_numbers)
    axes[-1].set_xticklabels(set_numbers)
    axes[-1].set_xlabel("Training set")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=window_colors[0], alpha=0.55, label=window_names[0]),
        plt.Rectangle((0, 0), 1, 1, color=window_colors[1], alpha=0.55, label=window_names[1]),
        plt.Rectangle((0, 0), 1, 1, color=window_colors[2], alpha=0.55, label=window_names[2])
    ]

    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3)

    fig.suptitle(
        f"Training-window hemoglobin activity – {hemoglobin_signal}",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    plt.show()

################################
#### Main analysis function ####
################################
def run_complete_analysis():
    participants = pd.read_excel(PARTICIPANTS_FILE)

    participant_IDs = []
    participant_groups = []
    mean_training_change_results = []
    force_results = []
    all_brain_data = {}

    for folder in os.listdir(DATA_DIRECTORY):
        ID = str(folder)
        group = folder.split("_")[0]
        participant_folder = DATA_DIRECTORY / folder

        participant_IDs.append(ID)
        participant_groups.append(group)

        print(ID)

        ############################
        #### Participant folders ####
        ############################
        brain_folder = participant_folder / "Brain data"
        grip_folder = participant_folder / "Grip data"

        if not brain_folder.exists():
            raise FileNotFoundError(f"Brain data folder was not found: {brain_folder}")

        if not grip_folder.exists():
            raise FileNotFoundError(f"Grip data folder was not found: {grip_folder}")

        ############################
        #### Brain-data analysis ####
        ############################
        participant_brain_data, participant_mean_rows = process_brain_participant(ID, group, participant_folder)

        all_brain_data[ID] = participant_brain_data
        mean_training_change_results.extend(participant_mean_rows)

        ###########################
        #### Grip-data analysis ####
        ###########################
        participant_force_results = process_force_participant(ID, group, participant_folder, participants)
        force_results.append(participant_force_results)

    mean_training_changes_df = pd.DataFrame(mean_training_change_results)
    force_results_df = pd.DataFrame(force_results)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    force_results_df.to_excel(RESULTS_DIRECTORY / "Training_results.xlsx", index=False)

    return all_brain_data, mean_training_changes_df, force_results_df, participant_IDs, participant_groups


brain_data = {}
mean_training_changes_df = pd.DataFrame()
spatial_error_df = pd.DataFrame()
list_ID = []
list_group_ID = []


if __name__ == "__main__":
    brain_data, mean_training_changes_df, spatial_error_df, list_ID, list_group_ID = run_complete_analysis()

    plot_three_training_windows_by_group(brain_data, "Left Tx1 O2Hb", show_data_points=True, show_connected_lines=True, show_ID=True)
    plot_three_training_windows_by_group(brain_data, "Right Tx4 O2Hb", show_data_points=True, show_connected_lines=True, show_ID=True)
    plot_three_training_windows_by_group(brain_data, "Left Tx1 HHb", show_data_points=True, show_connected_lines=True, show_ID=True)
    plot_three_training_windows_by_group(brain_data, "Right Tx4 HHb", show_data_points=True, show_connected_lines=True, show_ID=True)
    plot_three_training_windows_by_group(brain_data, "Left Tx1 O2Hb", show_data_points=False, show_connected_lines=False, show_ID=False)
    plot_three_training_windows_by_group(brain_data, "Right Tx4 O2Hb", show_data_points=False, show_connected_lines=False, show_ID=False)
    plot_three_training_windows_by_group(brain_data, "Left Tx1 HHb", show_data_points=False, show_connected_lines=False, show_ID=False)
    plot_three_training_windows_by_group(brain_data, "Right Tx4 HHb", show_data_points=False, show_connected_lines=False, show_ID=False)


    late_change_df = plot_late_minus_early_by_group("Left Tx1 O2Hb", show_data_points=False, show_connected_lines=False, show_ID=False)
    late_change_df = plot_late_minus_early_by_group("Right Tx4 O2Hb", show_data_points=False, show_connected_lines=False, show_ID=False)
    late_change_df = plot_late_minus_early_by_group("Left Tx1 HHb", show_data_points=False, show_connected_lines=False, show_ID=False)
    late_change_df = plot_late_minus_early_by_group("Right Tx4 HHb", show_data_points=False, show_connected_lines=False, show_ID=False)


    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=1, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=2, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=3, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=4, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=5, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=6, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=7, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=8, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=9, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    window_average_df = plot_hemoglobin_window_averages(brain_data, training_set=10, hemoglobin_signal="Left Tx1 O2Hb", show_connected_lines=True, show_ID=True)
    # Example group plots:
    # plot_group_boxplots_in_subplots(mean_training_changes_df, "Left Tx1 O2Hb", show_data_points=True, connect_participants=True, connect_group_mean=True)
    # plot_hemoglobin_and_force_by_group(mean_training_changes_df, spatial_error_df, "Left Tx1 O2Hb", "Normalized Average Spatial Error", show_ID_names=True)

    # Example complete-set plot:
    # plot_complete_training_set("Pink_4", 3, "Left Tx1 O2Hb", "baseline-corrected")




    plot_complete_training_set("Pink_4", 1, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_4", 1, "Left Tx1 O2Hb", "baseline-corrected")

    plot_complete_training_set("Pink_4", 1, "Left Tx1 HHb", "cleaned")

    plot_complete_training_set("Sine_2", 1, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 1, "Left Tx1 HHb", "cleaned")

    plot_complete_training_set("White_2", 1, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 2, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 3, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 4, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 5, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 6, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 7, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 8, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 9, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("White_2", 10, "Left Tx1 O2Hb", "baseline-corrected")


    plot_complete_training_set("Sine_2", 1, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 2, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 3, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 4, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 5, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 6, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 7, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 8, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 9, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Sine_2", 10, "Left Tx1 O2Hb", "baseline-corrected")

    plot_complete_training_set("Pink_2", 1, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 2, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 3, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 4, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 5, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 6, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 7, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 8, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 9, "Left Tx1 O2Hb", "baseline-corrected")
    plot_complete_training_set("Pink_2", 10, "Left Tx1 O2Hb", "baseline-corrected")



    plot_complete_training_set("White_2", 1, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 2, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 3, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 4, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 5, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 6, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 7, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 8, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 9, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("White_2", 10, "Left Tx1 O2Hb", "cleaned")


    plot_complete_training_set("Sine_2", 1, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 2, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 3, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 4, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 5, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 6, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 7, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 8, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 9, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Sine_2", 10, "Left Tx1 O2Hb", "cleaned")

    plot_complete_training_set("Pink_2", 1, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 2, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 3, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 4, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 5, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 6, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 7, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 8, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 9, "Left Tx1 O2Hb", "cleaned")
    plot_complete_training_set("Pink_2", 10, "Left Tx1 O2Hb", "cleaned")







