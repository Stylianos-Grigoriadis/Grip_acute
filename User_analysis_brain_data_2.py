import pandas as pd
import numpy as np
import lib
import Lib_grip as lb
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

def extract_cleaned_sets(cleaned_signal, baseline_start_indices, training_end_indices):
    cleaned_sets = []

    for baseline_start, training_end in zip(baseline_start_indices, training_end_indices):
        cleaned_set = cleaned_signal[int(baseline_start):int(training_end) + 1]
        cleaned_sets.append(cleaned_set)

    return np.array(cleaned_sets)

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
    baseline_corrected_sets = []

    for set_index in range(len(cleaned_sets)):
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])

        baseline_mean = np.mean(cleaned_sets[set_index, :training_start_local])

        baseline_corrected_set = cleaned_sets[set_index] - baseline_mean

        baseline_corrected_sets.append(baseline_corrected_set)

    baseline_corrected_sets = np.array(baseline_corrected_sets)

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

def calculate_mean_training_change(baseline_corrected_sets, baseline_start_indices, training_start_indices, training_end_indices):
    mean_training_changes = []

    for set_index in range(len(baseline_corrected_sets)):
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        training_end_local = int(training_end_indices[set_index] - baseline_start_indices[set_index])

        training_signal = baseline_corrected_sets[set_index, training_start_local:training_end_local + 1]

        mean_training_change = np.mean(training_signal)

        mean_training_changes.append(mean_training_change)

    return np.array(mean_training_changes)

def baseline_correct_cleaned_sets(cleaned_sets, baseline_start_indices, training_start_indices):
    baseline_corrected_sets = []

    for set_index in range(len(cleaned_sets)):
        training_start_local = int(training_start_indices[set_index] - baseline_start_indices[set_index])
        baseline_mean = np.mean(cleaned_sets[set_index, :training_start_local])
        baseline_corrected_set = cleaned_sets[set_index] - baseline_mean
        baseline_corrected_sets.append(baseline_corrected_set)

    return np.array(baseline_corrected_sets)

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

def plot_group_boxplots_by_set(mean_training_changes_df, signal_name):
    signal_data = mean_training_changes_df[mean_training_changes_df["Signal"] == signal_name]

    group_order = ["White", "Sine", "Pink"]

    group_colors = {
        "White": "lightgray",
        "Sine": "lightskyblue",
        "Pink": "hotpink"
    }

    group_offsets = {
        "White": -0.25,
        "Sine": 0,
        "Pink": 0.25
    }

    set_numbers = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(16, 7))

    random_generator = np.random.default_rng(10)

    for set_number in set_numbers:
        for group in group_order:
            values = signal_data.loc[
                (signal_data["Set"] == set_number) &
                (signal_data["Group"] == group),
                "Mean Training Change"
            ].to_numpy()

            box_position = set_number + group_offsets[group]

            boxplot = ax.boxplot(
                [values],
                positions=[box_position],
                widths=0.20,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False
            )

            boxplot["boxes"][0].set_facecolor(group_colors[group])
            boxplot["boxes"][0].set_alpha(0.6)

            boxplot["medians"][0].set_color("black")
            boxplot["medians"][0].set_linewidth(2)

            jitter = random_generator.uniform(-0.05, 0.05, size=len(values))

            ax.scatter(
                np.full(len(values), box_position) + jitter,
                values,
                color=group_colors[group],
                edgecolor="black",
                s=45,
                alpha=0.9,
                zorder=3
            )

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

    group_colors = {
        "White": "#7A7A7A",
        "Sine": "#2F7FBF",
        "Pink": "#E75480"
    }

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

        participant_jitter = {
            participant_ID: random_generator.uniform(-0.10, 0.10)
            for participant_ID in participant_IDs
        }

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
            group_means = group_data.groupby("Set")["Mean Training Change"].mean()
            group_means = group_means.reindex(set_numbers)

            ax.plot(set_numbers, group_means.to_numpy(), color="black", linewidth=3.5, marker="o", markerfacecolor=group_colors[group], markeredgecolor="black", markersize=7, zorder=4)

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(set_numbers)
        ax.set_xlabel("Training set")
        ax.set_title(f"{group} (n = {len(participant_IDs)})")
        ax.set_ylim(global_minimum - y_padding, global_maximum + y_padding)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean training change from baseline")

    if show_data_points:
        participant_point = axes[0].scatter([], [], color="gray", edgecolor="black", linewidth=0.5, s=40, label="Participant")
        legend_handles.append(participant_point)

    if connect_participants:
        participant_line = axes[0].plot([], [], color="gray", linewidth=0.7, alpha=0.5, label="Participant progression")[0]
        legend_handles.append(participant_line)

    if connect_group_mean:
        group_mean_line = axes[0].plot([], [], color="black", linewidth=3.5, marker="o", label="Group mean")[0]
        legend_handles.append(group_mean_line)

    fig.suptitle(signal_name, fontsize=16, y=0.99)

    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(legend_handles))

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.show()

def plot_hemoglobin_and_force_by_group(mean_training_changes_df, spatial_error_df, hemoglobin_signal, spatial_variable, show_ID_names=True):
    hemoglobin_data = mean_training_changes_df[
        mean_training_changes_df["Signal"] == hemoglobin_signal
    ].copy()

    spatial_error_data = spatial_error_df.copy()

    hemoglobin_data["ID"] = hemoglobin_data["ID"].astype(str)
    spatial_error_data["ID"] = spatial_error_data["ID"].astype(str)

    spatial_columns = [
        f"{spatial_variable} Training {set_number}"
        for set_number in range(1, 11)
    ]

    missing_columns = [
        column
        for column in spatial_columns
        if column not in spatial_error_data.columns
    ]

    if missing_columns:
        raise ValueError(f"These columns were not found: {missing_columns}")

    common_IDs = sorted(
        set(hemoglobin_data["ID"]).intersection(
            set(spatial_error_data["ID"])
        )
    )

    if len(common_IDs) == 0:
        raise ValueError("No matching participant IDs were found between the two dataframes.")

    group_order = ["White", "Sine", "Pink"]
    set_numbers = np.arange(1, 11)

    participant_colors = plt.cm.tab20(
        np.linspace(0, 1, len(common_IDs))
    )

    color_by_ID = dict(
        zip(common_IDs, participant_colors)
    )

    fig, axes = plt.subplots(2, 3, figsize=(22, 11), sharex=True, sharey="row")

    participant_artists = {
        participant_ID: []
        for participant_ID in common_IDs
    }

    legend_handles = []
    legend_IDs = []

    for column_index, group in enumerate(group_order):
        hemoglobin_axis = axes[0, column_index]
        spatial_axis = axes[1, column_index]

        group_hemoglobin_data = hemoglobin_data[
            hemoglobin_data["Group"] == group
        ]

        group_IDs = [
            participant_ID
            for participant_ID in common_IDs
            if participant_ID in group_hemoglobin_data["ID"].values
        ]

        for participant_ID in group_IDs:
            participant_color = color_by_ID[participant_ID]

            participant_hemoglobin_data = group_hemoglobin_data[
                group_hemoglobin_data["ID"] == participant_ID
            ]

            participant_hemoglobin_data = participant_hemoglobin_data.set_index(
                "Set"
            ).reindex(set_numbers)

            hemoglobin_values = participant_hemoglobin_data[
                "Mean Training Change"
            ].to_numpy(dtype=float)

            hemoglobin_line = hemoglobin_axis.plot(
                set_numbers,
                hemoglobin_values,
                marker="o",
                markersize=5,
                linewidth=1.2,
                color=participant_color,
                label=participant_ID
            )[0]

            participant_artists[participant_ID].append(
                hemoglobin_line
            )

            participant_spatial_data = spatial_error_data[
                spatial_error_data["ID"] == participant_ID
            ]

            spatial_values = participant_spatial_data[
                spatial_columns
            ].iloc[0].to_numpy(dtype=float)

            spatial_line = spatial_axis.plot(
                set_numbers,
                spatial_values,
                marker="o",
                markersize=5,
                linewidth=1.2,
                color=participant_color,
                label=participant_ID
            )[0]

            participant_artists[participant_ID].append(
                spatial_line
            )

            if show_ID_names:
                valid_hemoglobin = np.where(
                    np.isfinite(hemoglobin_values)
                )[0]

                if len(valid_hemoglobin) > 0:
                    final_index = valid_hemoglobin[-1]

                    hemoglobin_text = hemoglobin_axis.annotate(
                        participant_ID,
                        (
                            set_numbers[final_index],
                            hemoglobin_values[final_index]
                        ),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=participant_color,
                        va="center"
                    )

                    participant_artists[participant_ID].append(
                        hemoglobin_text
                    )

                valid_spatial = np.where(
                    np.isfinite(spatial_values)
                )[0]

                if len(valid_spatial) > 0:
                    final_index = valid_spatial[-1]

                    spatial_text = spatial_axis.annotate(
                        participant_ID,
                        (
                            set_numbers[final_index],
                            spatial_values[final_index]
                        ),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=participant_color,
                        va="center"
                    )

                    participant_artists[participant_ID].append(
                        spatial_text
                    )

            legend_handles.append(
                hemoglobin_line
            )

            legend_IDs.append(
                participant_ID
            )

        hemoglobin_axis.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        hemoglobin_axis.set_title(
            group,
            fontsize=14
        )

        spatial_axis.set_xlabel(
            "Training set"
        )

        hemoglobin_axis.set_xlim(
            0.7,
            11
        )

        spatial_axis.set_xlim(
            0.7,
            11
        )

        hemoglobin_axis.set_xticks(
            set_numbers
        )

        spatial_axis.set_xticks(
            set_numbers
        )

        hemoglobin_axis.grid(
            alpha=0.3
        )

        spatial_axis.grid(
            alpha=0.3
        )

    axes[0, 0].set_ylabel(
        f"{hemoglobin_signal}\nMean change from baseline"
    )

    axes[1, 0].set_ylabel(
        spatial_variable
    )

    unique_legend_handles = []
    unique_legend_IDs = []

    for participant_ID, legend_handle in zip(legend_IDs, legend_handles):
        if participant_ID not in unique_legend_IDs:
            unique_legend_IDs.append(
                participant_ID
            )

            unique_legend_handles.append(
                legend_handle
            )

    legend = fig.legend(
        handles=unique_legend_handles,
        labels=unique_legend_IDs,
        title="Click ID to show/hide",
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        ncol=1
    )

    legend_artist_to_ID = {}
    legend_entries = {}

    for legend_line, legend_text, participant_ID in zip(
        legend.get_lines(),
        legend.get_texts(),
        unique_legend_IDs
    ):
        legend_line.set_picker(5)
        legend_text.set_picker(True)

        legend_artist_to_ID[legend_line] = participant_ID
        legend_artist_to_ID[legend_text] = participant_ID

        legend_entries[participant_ID] = (
            legend_line,
            legend_text
        )

    def toggle_participant(event):
        selected_artist = event.artist

        if selected_artist not in legend_artist_to_ID:
            return

        participant_ID = legend_artist_to_ID[selected_artist]
        artists = participant_artists[participant_ID]

        new_visibility = not artists[0].get_visible()

        for artist in artists:
            artist.set_visible(
                new_visibility
            )

        legend_line, legend_text = legend_entries[participant_ID]

        if new_visibility:
            legend_line.set_alpha(1.0)
            legend_text.set_alpha(1.0)
        else:
            legend_line.set_alpha(0.2)
            legend_text.set_alpha(0.2)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect(
        "pick_event",
        toggle_participant
    )

    fig.suptitle(
        f"{hemoglobin_signal} and {spatial_variable}",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 0.83, 0.96]
    )

    plt.show()


participants_directory = r'C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Signals'
os.chdir(participants_directory)
participants = pd.read_excel(r'Participants.xlsx')



directory = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Data\Data to screen"

fs = 100

brain_data = {}
list_ID = []
list_group_ID = []
mean_training_change_results = []
for folder in os.listdir(directory):
    ID = str(folder)
    list_ID.append(ID)
    print(ID)
    group = folder.split("_")[0]
    list_group_ID.append(group)

    brain_folder = os.path.join(directory, folder, "Brain data")
    os.chdir(brain_folder)
    name = "Artinis_" + ID[0] + ID.split("_")[1]

    data, fs, list_indices, list_time_events, pre_event_indices, derived_end_indices, final_event_indices, list_training_sets = lb.artinis_read_file_10_events_plot(brain_folder, name, write_manual_events_to_excel=False, plot=False)
    brain_data[ID] = {
        'Group': group,
        'Sampling Frequency': fs,
        'Raw Data': data,
        'Event Indices': list_indices,
        'Event Times': list_time_events,
        'Pre Event Indices': pre_event_indices,
        'Derived End Indices': derived_end_indices,
        'Final Event Indices': final_event_indices,
        'Raw Training Sets': list_training_sets
    }

    time = data["Time"].to_numpy()

    ###########################
    #### Filtering Process ####
    ###########################
    filter_start_index = max(0, int(list_indices[0] - 60 * fs))
    filter_end_index = min(data.height, int(derived_end_indices[-1] + 60 * fs + 1))
    number_of_rows = filter_end_index - filter_start_index

    data_to_filter = data.slice(filter_start_index, number_of_rows)

    left_Rx1_Tx1_O2Hb = data_to_filter['[9322] Rx1 - Tx1  O2Hb'].to_numpy()
    left_Rx1_Tx2_O2Hb = data_to_filter['[9322] Rx1 - Tx2  O2Hb'].to_numpy()
    left_Rx1_Tx3_O2Hb = data_to_filter['[9322] Rx1 - Tx3  O2Hb'].to_numpy()
    left_Rx2_Tx1_O2Hb = data_to_filter['[9322] Rx2 - Tx1  O2Hb'].to_numpy()
    left_Rx2_Tx2_O2Hb = data_to_filter['[9322] Rx2 - Tx2  O2Hb'].to_numpy()
    left_Rx2_Tx3_O2Hb = data_to_filter['[9322] Rx2 - Tx3  O2Hb'].to_numpy()
    left_Rx1_Tx1_HHb = data_to_filter['[9322] Rx1 - Tx1  HHb'].to_numpy()
    left_Rx1_Tx2_HHb = data_to_filter['[9322] Rx1 - Tx2  HHb'].to_numpy()
    left_Rx1_Tx3_HHb = data_to_filter['[9322] Rx1 - Tx3  HHb'].to_numpy()
    left_Rx2_Tx1_HHb = data_to_filter['[9322] Rx2 - Tx1  HHb'].to_numpy()
    left_Rx2_Tx2_HHb = data_to_filter['[9322] Rx2 - Tx2  HHb'].to_numpy()
    left_Rx2_Tx3_HHb = data_to_filter['[9322] Rx2 - Tx3  HHb'].to_numpy()
    right_Rx3_Tx4_O2Hb = data_to_filter['[9323] Rx3 - Tx4  O2Hb'].to_numpy()
    right_Rx3_Tx5_O2Hb = data_to_filter['[9323] Rx3 - Tx5  O2Hb'].to_numpy()
    right_Rx3_Tx6_O2Hb = data_to_filter['[9323] Rx3 - Tx6  O2Hb'].to_numpy()
    right_Rx4_Tx4_O2Hb = data_to_filter['[9323] Rx4 - Tx4  O2Hb'].to_numpy()
    right_Rx4_Tx5_O2Hb = data_to_filter['[9323] Rx4 - Tx5  O2Hb'].to_numpy()
    right_Rx4_Tx6_O2Hb = data_to_filter['[9323] Rx4 - Tx6  O2Hb'].to_numpy()
    right_Rx3_Tx4_HHb = data_to_filter['[9323] Rx3 - Tx4  HHb'].to_numpy()
    right_Rx3_Tx5_HHb = data_to_filter['[9323] Rx3 - Tx5  HHb'].to_numpy()
    right_Rx3_Tx6_HHb = data_to_filter['[9323] Rx3 - Tx6  HHb'].to_numpy()
    right_Rx4_Tx4_HHb = data_to_filter['[9323] Rx4 - Tx4  HHb'].to_numpy()
    right_Rx4_Tx5_HHb = data_to_filter['[9323] Rx4 - Tx5  HHb'].to_numpy()
    right_Rx4_Tx6_HHb = data_to_filter['[9323] Rx4 - Tx6  HHb'].to_numpy()

    left_Rx1_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_O2Hb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx1_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx1_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx1_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx1_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx2_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx2_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    left_Rx2_Tx3_HHb = lb.butter_bandpass_filtfilt_SOS(left_Rx2_Tx3_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_O2Hb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_O2Hb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx3_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx3_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx4_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx4_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx5_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx5_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)
    right_Rx4_Tx6_HHb = lb.butter_bandpass_filtfilt_SOS(right_Rx4_Tx6_HHb, fs, low=0.01, high=0.30, order=4, plot=False, demean=False)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    #
    # ax1.plot(left_Rx2_Tx1_O2Hb, label="left_Rx2_Tx1_O2Hb")
    # ax1.plot(left_Rx2_Tx2_O2Hb, label="left_Rx2_Tx2_O2Hb")
    # ax1.plot(left_Rx2_Tx3_O2Hb, label="left_Rx2_Tx3_O2Hb")
    # ax1.plot(right_Rx4_Tx4_O2Hb, label="right_Rx4_Tx4_O2Hb")
    # ax1.plot(right_Rx4_Tx5_O2Hb, label="right_Rx4_Tx5_O2Hb")
    # ax1.plot(right_Rx4_Tx6_O2Hb, label="right_Rx4_Tx6_O2Hb")
    # ax1.plot(left_Rx1_Tx1_O2Hb, label="left_Rx1_Tx1_O2Hb", color="black", linewidth=2)
    # ax1.set_ylabel("HbO")
    # ax1.set_title("Filtered HbO signals")
    # ax1.legend()
    # ax1.grid(alpha=0.3)
    #
    # ax2.plot(left_Rx2_Tx1_HHb, label="left_Rx2_Tx1_HHb")
    # ax2.plot(left_Rx2_Tx2_HHb, label="left_Rx2_Tx2_HHb")
    # ax2.plot(left_Rx2_Tx3_HHb, label="left_Rx2_Tx3_HHb")
    # ax2.plot(right_Rx4_Tx4_HHb, label="right_Rx4_Tx4_HHb")
    # ax2.plot(right_Rx4_Tx5_HHb, label="right_Rx4_Tx5_HHb")
    # ax2.plot(right_Rx4_Tx6_HHb, label="right_Rx4_Tx6_HHb")
    # ax2.set_xlabel("Sample")
    # ax2.set_ylabel("HHb")
    # ax2.set_title("Filtered HHb signals")
    # ax2.legend()
    # ax2.grid(alpha=0.3)
    #
    # plt.tight_layout()
    # plt.show()

    #####################
    #### PCA Process ####
    #####################
    analysis_start_index = max(0, int(list_indices[0] - 30 * fs))
    number_of_indices_cut = analysis_start_index
    analysis_end_index = min(data.height, int(derived_end_indices[-1] + 1))
    analysis_start_index_filtered = analysis_start_index - filter_start_index
    analysis_end_index_filtered = analysis_end_index - filter_start_index
    analysis_slice = slice(analysis_start_index_filtered, analysis_end_index_filtered)

    short_channel_matrix = np.column_stack([
        left_Rx2_Tx1_O2Hb[analysis_slice],  # Left short HbO 1
        left_Rx2_Tx2_O2Hb[analysis_slice],  # Left short HbO 2
        left_Rx2_Tx3_O2Hb[analysis_slice],  # Left short HbO 3
        right_Rx4_Tx4_O2Hb[analysis_slice],  # Right short HbO 1
        right_Rx4_Tx5_O2Hb[analysis_slice],  # Right short HbO 2
        right_Rx4_Tx6_O2Hb[analysis_slice],  # Right short HbO 3
        left_Rx2_Tx1_HHb[analysis_slice],  # Left short HHb 1
        left_Rx2_Tx2_HHb[analysis_slice],  # Left short HHb 2
        left_Rx2_Tx3_HHb[analysis_slice],  # Left short HHb 3
        right_Rx4_Tx4_HHb[analysis_slice],  # Right short HHb 1
        right_Rx4_Tx5_HHb[analysis_slice],  # Right short HHb 2
        right_Rx4_Tx6_HHb[analysis_slice]  # Right short HHb 3
    ])
    scaler = StandardScaler()
    short_channel_matrix_standardized = scaler.fit_transform(short_channel_matrix)

    pca = PCA(n_components=12)
    short_channel_pca_components = pca.fit_transform(short_channel_matrix_standardized)

    # plot_short_channel_pca_results(pca, short_channel_matrix_standardized, short_channel_pca_components, fs, ID, variance_threshold=90)

    left_Rx1_Tx1_O2Hb_cut_for_analysis = left_Rx1_Tx1_O2Hb[analysis_slice]
    left_Rx1_Tx2_O2Hb_cut_for_analysis = left_Rx1_Tx2_O2Hb[analysis_slice]
    left_Rx1_Tx3_O2Hb_cut_for_analysis = left_Rx1_Tx3_O2Hb[analysis_slice]
    left_Rx1_Tx1_HHb_cut_for_analysis = left_Rx1_Tx1_HHb[analysis_slice]
    left_Rx1_Tx2_HHb_cut_for_analysis = left_Rx1_Tx2_HHb[analysis_slice]
    left_Rx1_Tx3_HHb_cut_for_analysis = left_Rx1_Tx3_HHb[analysis_slice]
    right_Rx3_Tx4_O2Hb_cut_for_analysis = right_Rx3_Tx4_O2Hb[analysis_slice]
    right_Rx3_Tx5_O2Hb_cut_for_analysis = right_Rx3_Tx5_O2Hb[analysis_slice]
    right_Rx3_Tx6_O2Hb_cut_for_analysis = right_Rx3_Tx6_O2Hb[analysis_slice]
    right_Rx3_Tx4_HHb_cut_for_analysis = right_Rx3_Tx4_HHb[analysis_slice]
    right_Rx3_Tx5_HHb_cut_for_analysis = right_Rx3_Tx5_HHb[analysis_slice]
    right_Rx3_Tx6_HHb_cut_for_analysis = right_Rx3_Tx6_HHb[analysis_slice]

    left_Rx1_Tx1_O2Hb_cleaned, left_Rx1_Tx1_O2Hb_superficial, left_Rx1_Tx1_O2Hb_betas, left_Rx1_Tx1_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx1_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx1_O2Hb", plot=False)
    left_Rx1_Tx2_O2Hb_cleaned, left_Rx1_Tx2_O2Hb_superficial, left_Rx1_Tx2_O2Hb_betas, left_Rx1_Tx2_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx2_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx2_O2Hb", plot=False)
    left_Rx1_Tx3_O2Hb_cleaned, left_Rx1_Tx3_O2Hb_superficial, left_Rx1_Tx3_O2Hb_betas, left_Rx1_Tx3_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx3_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx3_O2Hb", plot=False)
    left_Rx1_Tx1_HHb_cleaned, left_Rx1_Tx1_HHb_superficial, left_Rx1_Tx1_HHb_betas, left_Rx1_Tx1_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx1_HHb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx1_HHb", plot=False)
    left_Rx1_Tx2_HHb_cleaned, left_Rx1_Tx2_HHb_superficial, left_Rx1_Tx2_HHb_betas, left_Rx1_Tx2_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx2_HHb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx2_HHb", plot=False)
    left_Rx1_Tx3_HHb_cleaned, left_Rx1_Tx3_HHb_superficial, left_Rx1_Tx3_HHb_betas, left_Rx1_Tx3_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(left_Rx1_Tx3_HHb_cut_for_analysis, short_channel_pca_components, fs, "left_Rx1_Tx3_HHb", plot=False)
    right_Rx3_Tx4_O2Hb_cleaned, right_Rx3_Tx4_O2Hb_superficial, right_Rx3_Tx4_O2Hb_betas, right_Rx3_Tx4_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx4_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx4_O2Hb", plot=False)
    right_Rx3_Tx5_O2Hb_cleaned, right_Rx3_Tx5_O2Hb_superficial, right_Rx3_Tx5_O2Hb_betas, right_Rx3_Tx5_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx5_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx5_O2Hb", plot=False)
    right_Rx3_Tx6_O2Hb_cleaned, right_Rx3_Tx6_O2Hb_superficial, right_Rx3_Tx6_O2Hb_betas, right_Rx3_Tx6_O2Hb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx6_O2Hb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx6_O2Hb", plot=False)
    right_Rx3_Tx4_HHb_cleaned, right_Rx3_Tx4_HHb_superficial, right_Rx3_Tx4_HHb_betas, right_Rx3_Tx4_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx4_HHb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx4_HHb", plot=False)
    right_Rx3_Tx5_HHb_cleaned, right_Rx3_Tx5_HHb_superficial, right_Rx3_Tx5_HHb_betas, right_Rx3_Tx5_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx5_HHb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx5_HHb", plot=False)
    right_Rx3_Tx6_HHb_cleaned, right_Rx3_Tx6_HHb_superficial, right_Rx3_Tx6_HHb_betas, right_Rx3_Tx6_HHb_contribution_percentage = remove_superficial_signal_from_long_channel(right_Rx3_Tx6_HHb_cut_for_analysis, short_channel_pca_components, fs, "right_Rx3_Tx6_HHb", plot=False)

    original_long_signals = [
        left_Rx1_Tx1_O2Hb_cut_for_analysis,
        left_Rx1_Tx2_O2Hb_cut_for_analysis,
        left_Rx1_Tx3_O2Hb_cut_for_analysis,
        left_Rx1_Tx1_HHb_cut_for_analysis,
        left_Rx1_Tx2_HHb_cut_for_analysis,
        left_Rx1_Tx3_HHb_cut_for_analysis,
        right_Rx3_Tx4_O2Hb_cut_for_analysis,
        right_Rx3_Tx5_O2Hb_cut_for_analysis,
        right_Rx3_Tx6_O2Hb_cut_for_analysis,
        right_Rx3_Tx4_HHb_cut_for_analysis,
        right_Rx3_Tx5_HHb_cut_for_analysis,
        right_Rx3_Tx6_HHb_cut_for_analysis
    ]

    superficial_long_signals = [
        left_Rx1_Tx1_O2Hb_superficial,
        left_Rx1_Tx2_O2Hb_superficial,
        left_Rx1_Tx3_O2Hb_superficial,
        left_Rx1_Tx1_HHb_superficial,
        left_Rx1_Tx2_HHb_superficial,
        left_Rx1_Tx3_HHb_superficial,
        right_Rx3_Tx4_O2Hb_superficial,
        right_Rx3_Tx5_O2Hb_superficial,
        right_Rx3_Tx6_O2Hb_superficial,
        right_Rx3_Tx4_HHb_superficial,
        right_Rx3_Tx5_HHb_superficial,
        right_Rx3_Tx6_HHb_superficial
    ]

    cleaned_long_signals = [
        left_Rx1_Tx1_O2Hb_cleaned,
        left_Rx1_Tx2_O2Hb_cleaned,
        left_Rx1_Tx3_O2Hb_cleaned,
        left_Rx1_Tx1_HHb_cleaned,
        left_Rx1_Tx2_HHb_cleaned,
        left_Rx1_Tx3_HHb_cleaned,
        right_Rx3_Tx4_O2Hb_cleaned,
        right_Rx3_Tx5_O2Hb_cleaned,
        right_Rx3_Tx6_O2Hb_cleaned,
        right_Rx3_Tx4_HHb_cleaned,
        right_Rx3_Tx5_HHb_cleaned,
        right_Rx3_Tx6_HHb_cleaned
    ]

    long_signal_names = [
        "Left Tx1 O2Hb", "Left Tx2 O2Hb", "Left Tx3 O2Hb",
        "Left Tx1 HHb", "Left Tx2 HHb", "Left Tx3 HHb",
        "Right Tx4 O2Hb", "Right Tx5 O2Hb", "Right Tx6 O2Hb",
        "Right Tx4 HHb", "Right Tx5 HHb", "Right Tx6 HHb"
    ]

    # plot_all_long_channel_regressions(original_long_signals, superficial_long_signals, cleaned_long_signals, long_signal_names, fs, ID)

    # explained_variance_percentages = plot_short_channel_explained_variance(original_long_signals, cleaned_long_signals, long_signal_names, ID)

    baseline_start_indices = np.array(pre_event_indices) - number_of_indices_cut
    training_start_indices = np.array(list_indices) - number_of_indices_cut
    training_end_indices = np.array(derived_end_indices) - number_of_indices_cut

    left_Rx1_Tx1_O2Hb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx1_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    left_Rx1_Tx2_O2Hb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx2_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    left_Rx1_Tx3_O2Hb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx3_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    left_Rx1_Tx1_HHb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx1_HHb_cleaned, baseline_start_indices, training_end_indices)
    left_Rx1_Tx2_HHb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx2_HHb_cleaned, baseline_start_indices, training_end_indices)
    left_Rx1_Tx3_HHb_cleaned_sets = extract_cleaned_sets(left_Rx1_Tx3_HHb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx4_O2Hb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx4_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx5_O2Hb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx5_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx6_O2Hb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx6_O2Hb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx4_HHb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx4_HHb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx5_HHb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx5_HHb_cleaned, baseline_start_indices, training_end_indices)
    right_Rx3_Tx6_HHb_cleaned_sets = extract_cleaned_sets(right_Rx3_Tx6_HHb_cleaned, baseline_start_indices, training_end_indices)
    print(left_Rx1_Tx1_O2Hb_cleaned_sets.shape)


    # plot_cleaned_sets_with_events(left_Rx1_Tx1_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx1_O2Hb")
    # plot_cleaned_sets_with_events(left_Rx1_Tx1_HHb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx1_HHb")
    # plot_cleaned_sets_with_events(right_Rx3_Tx4_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "right_Rx3_Tx4_O2Hb")
    # plot_cleaned_sets_with_events(right_Rx3_Tx4_HHb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "right_Rx3_Tx4_HHb")
    # plot_baseline_corrected_cleaned_sets(left_Rx1_Tx1_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx1_O2Hb")
    # plot_baseline_corrected_cleaned_sets(left_Rx1_Tx2_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx2_O2Hb")
    # plot_baseline_corrected_cleaned_sets(left_Rx1_Tx3_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx3_O2Hb")

    # plot_baseline_corrected_cleaned_sets(left_Rx1_Tx1_HHb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "left_Rx1_Tx1_HHb")
    # plot_baseline_corrected_cleaned_sets(right_Rx3_Tx4_O2Hb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "right_Rx3_Tx4_O2Hb")
    # plot_baseline_corrected_cleaned_sets(right_Rx3_Tx4_HHb_cleaned_sets, baseline_start_indices, training_start_indices, training_end_indices, fs, "right_Rx3_Tx4_HHb")

    cleaned_sets_list = [
        left_Rx1_Tx1_O2Hb_cleaned_sets,
        left_Rx1_Tx2_O2Hb_cleaned_sets,
        left_Rx1_Tx3_O2Hb_cleaned_sets,
        left_Rx1_Tx1_HHb_cleaned_sets,
        left_Rx1_Tx2_HHb_cleaned_sets,
        left_Rx1_Tx3_HHb_cleaned_sets,
        right_Rx3_Tx4_O2Hb_cleaned_sets,
        right_Rx3_Tx5_O2Hb_cleaned_sets,
        right_Rx3_Tx6_O2Hb_cleaned_sets,
        right_Rx3_Tx4_HHb_cleaned_sets,
        right_Rx3_Tx5_HHb_cleaned_sets,
        right_Rx3_Tx6_HHb_cleaned_sets
    ]

    baseline_corrected_sets_list = []
    mean_training_changes_list = []

    for cleaned_sets in cleaned_sets_list:
        baseline_corrected_sets = baseline_correct_cleaned_sets(cleaned_sets, baseline_start_indices,
                                                                training_start_indices)

        mean_training_changes = calculate_mean_training_change(baseline_corrected_sets, baseline_start_indices,
                                                               training_start_indices, training_end_indices)

        baseline_corrected_sets_list.append(baseline_corrected_sets)
        mean_training_changes_list.append(mean_training_changes)

    mean_training_changes_matrix = np.array(mean_training_changes_list)

    brain_data[ID]["Cleaned Training Sets"] = dict(zip(long_signal_names, cleaned_sets_list))
    brain_data[ID]["Baseline Corrected Training Sets"] = dict(zip(long_signal_names, baseline_corrected_sets_list))
    brain_data[ID]["Mean Training Changes"] = dict(zip(long_signal_names, mean_training_changes_list))

    for signal_name, mean_training_changes in zip(long_signal_names, mean_training_changes_list):
        for set_number, mean_training_change in enumerate(mean_training_changes, start=1):
            mean_training_change_results.append({
                "ID": ID,
                "Group": group,
                "Signal": signal_name,
                "Set": set_number,
                "Mean Training Change": mean_training_change
            })

    # plot_participant_mean_training_changes(mean_training_changes_matrix, long_signal_names, ID)



mean_training_changes_df = pd.DataFrame(mean_training_change_results)

# plot_group_boxplots_by_set(mean_training_changes_df, "Left Tx1 O2Hb")
# plot_group_boxplots_by_set(mean_training_changes_df, "Right Tx4 O2Hb")
# plot_group_boxplots_by_set(mean_training_changes_df, "Left Tx1 HHb")
# plot_group_boxplots_by_set(mean_training_changes_df, "Right Tx4 HHb")

# plot_group_boxplots_in_subplots(mean_training_changes_df, "Left Tx1 O2Hb", show_data_points=True, connect_participants=True, connect_group_mean=True)
# plot_group_boxplots_in_subplots(mean_training_changes_df, "Right Tx4 O2Hb", show_data_points=True, connect_participants=True, connect_group_mean=True)
# plot_group_boxplots_in_subplots(mean_training_changes_df, "Left Tx1 HHb", show_data_points=True, connect_participants=True, connect_group_mean=True)
# plot_group_boxplots_in_subplots(mean_training_changes_df, "Right Tx4 HHb", show_data_points=True, connect_participants=True, connect_group_mean=True)

results_spatial_error_directory = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"
os.chdir(results_spatial_error_directory)
spatial_error_df = pd.read_excel(r'Training_results.xlsx')
print(spatial_error_df.columns)


plot_hemoglobin_and_force_by_group(mean_training_changes_df, spatial_error_df, "Left Tx1 O2Hb", "Normalized Average Spatial Error", show_ID_names=True)



















