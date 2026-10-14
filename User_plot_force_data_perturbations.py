import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def normalize_summary_name(summary):
    summary_map = {"Min": "Minimum", "Minimum": "Minimum", "Average": "Average", "Median": "Median"}
    if summary not in summary_map:
        raise ValueError("summary must be 'Min', 'Minimum', 'Average', or 'Median'")
    return summary_map[summary]


def get_adaptation_columns(perturbation_results, summary, method, direction, sd_factor):
    summary_name = normalize_summary_name(summary)

    if direction not in ["Up", "Down"]:
        raise ValueError("direction must be 'Up' or 'Down'")
    if method not in ["SD", "Asymptote"]:
        raise ValueError("method must be 'SD' or 'Asymptote'")
    if summary_name == "Median" and method == "Asymptote":
        raise ValueError("Median values were only calculated for the SD method, not Asymptote")

    suffix = f"SD {sd_factor}" if method == "SD" else "Asymptote"
    pre_column = f"{summary_name} Pre {direction} Adaptation Time {suffix}"
    post_column = f"{summary_name} Post {direction} Adaptation Time {suffix}"

    missing = [col for col in [pre_column, post_column] if col not in perturbation_results.columns]
    if missing:
        raise ValueError(f"Missing Excel columns: {missing}. Load the updated results file containing median values.")

    return summary_name, pre_column, post_column


def plot_perturbation_boxplots(perturbation_results, summary, method, direction, group_colors, sd_factor, show_datapoints=True, connect_participants=False, show_ID=False):

    summary_name, pre_column, post_column = get_adaptation_columns(
        perturbation_results, summary, method, direction, sd_factor
    )

    data = perturbation_results[
        ["ID", "Group", pre_column, post_column]
    ].copy()

    data = data.melt(
        id_vars=["ID", "Group"],
        value_vars=[pre_column, post_column],
        var_name="Condition",
        value_name="Adaptation Time"
    )

    data["Condition"] = data["Condition"].replace({
        pre_column: "Pre",
        post_column: "Post"
    })

    group_order = list(group_colors.keys())

    fig, ax = plt.subplots(figsize=(9, 7))

    box_width = 0.65

    sns.boxplot(
        data=data,
        x="Condition",
        y="Adaptation Time",
        hue="Group",
        order=["Pre", "Post"],
        hue_order=group_order,
        palette=group_colors,
        width=box_width,
        showfliers=not show_datapoints,
        ax=ax
    )

    group_offsets = np.linspace(
        -box_width / 2 + box_width / (2 * len(group_order)),
        box_width / 2 - box_width / (2 * len(group_order)),
        len(group_order)
    )

    rng = np.random.default_rng(1)

    for group_index, group in enumerate(group_order):

        group_data = data[data["Group"] == group]

        IDs = group_data["ID"].unique()

        for ID in IDs:

            participant_data = group_data[group_data["ID"] == ID]

            pre_value = participant_data.loc[
                participant_data["Condition"] == "Pre",
                "Adaptation Time"
            ]

            post_value = participant_data.loc[
                participant_data["Condition"] == "Post",
                "Adaptation Time"
            ]

            if len(pre_value) == 0 or len(post_value) == 0:
                continue

            pre_value = pre_value.iloc[0]
            post_value = post_value.iloc[0]

            if pd.isna(pre_value) or pd.isna(post_value):
                continue

            jitter = rng.uniform(-0.035, 0.035)

            x_pre = 0 + group_offsets[group_index] + jitter
            x_post = 1 + group_offsets[group_index] + jitter

            if connect_participants:
                ax.plot(
                    [x_pre, x_post],
                    [pre_value, post_value],
                    color=group_colors[group],
                    alpha=0.30,
                    linewidth=1
                )

            if show_datapoints:
                ax.scatter(
                    x_pre,
                    pre_value,
                    color=group_colors[group],
                    edgecolor="black",
                    linewidth=0.5,
                    s=35,
                    zorder=3
                )

                ax.scatter(
                    x_post,
                    post_value,
                    color=group_colors[group],
                    edgecolor="black",
                    linewidth=0.5,
                    s=35,
                    zorder=3
                )

            if show_ID:
                ax.annotate(
                    str(ID),
                    (x_pre, pre_value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

                ax.annotate(
                    str(ID),
                    (x_post, post_value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

    ax.set_xlabel("")
    ax.set_ylabel("Adaptation Time (s)", fontsize=14)

    ax.set_title(
        f"{summary_name} Adaptation Time – {direction} Perturbation",
        fontsize=16
    )

    ax.tick_params(axis="both", labelsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles[:len(group_order)],
        labels[:len(group_order)],
        title="Group",
        frameon=False,
        fontsize=11,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.show()

def plot_perturbation_group_subplots(perturbation_results, summary, method, direction, group_colors, sd_factor, show_datapoints=True, connect_participants=False, show_ID=False):

    summary_name, pre_column, post_column = get_adaptation_columns(
        perturbation_results, summary, method, direction, sd_factor
    )

    data = perturbation_results[
        ["ID", "Group", pre_column, post_column]
    ].copy()

    data = data.melt(
        id_vars=["ID", "Group"],
        value_vars=[pre_column, post_column],
        var_name="Condition",
        value_name="Adaptation Time"
    )

    data["Condition"] = data["Condition"].replace({
        pre_column: "Pre",
        post_column: "Post"
    })

    group_order = list(group_colors.keys())

    fig, axes = plt.subplots(
        1,
        len(group_order),
        figsize=(15, 6),
        sharey=True
    )

    rng = np.random.default_rng(1)

    for ax, group in zip(axes, group_order):

        group_data = data[data["Group"] == group]

        sns.boxplot(
            data=group_data,
            x="Condition",
            y="Adaptation Time",
            order=["Pre", "Post"],
            color=group_colors[group],
            width=0.5,
            showfliers=not show_datapoints,
            ax=ax
        )

        IDs = group_data["ID"].unique()

        for ID in IDs:

            participant_data = group_data[group_data["ID"] == ID]

            pre_value = participant_data.loc[
                participant_data["Condition"] == "Pre",
                "Adaptation Time"
            ]

            post_value = participant_data.loc[
                participant_data["Condition"] == "Post",
                "Adaptation Time"
            ]

            if len(pre_value) == 0 or len(post_value) == 0:
                continue

            pre_value = pre_value.iloc[0]
            post_value = post_value.iloc[0]

            if pd.isna(pre_value) or pd.isna(post_value):
                continue

            jitter = rng.uniform(-0.06, 0.06)

            x_pre = 0 + jitter
            x_post = 1 + jitter

            if connect_participants:
                ax.plot(
                    [x_pre, x_post],
                    [pre_value, post_value],
                    color=group_colors[group],
                    alpha=0.35,
                    linewidth=1
                )

            if show_datapoints:
                ax.scatter(
                    x_pre,
                    pre_value,
                    color=group_colors[group],
                    edgecolor="black",
                    linewidth=0.5,
                    s=40,
                    zorder=3
                )

                ax.scatter(
                    x_post,
                    post_value,
                    color=group_colors[group],
                    edgecolor="black",
                    linewidth=0.5,
                    s=40,
                    zorder=3
                )

            if show_ID:
                ax.annotate(
                    str(ID),
                    (x_pre, pre_value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

                ax.annotate(
                    str(ID),
                    (x_post, post_value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8
                )

        ax.set_title(group, fontsize=15)
        ax.set_xlabel("")
        ax.set_ylabel("Adaptation Time (s)", fontsize=13)

        ax.tick_params(axis="both", labelsize=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"{summary_name} Adaptation Time – {direction} Perturbation",
        fontsize=17
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def plot_perturbation_difference_boxplots(perturbation_results, summary, method, group_colors, sd_factor, show_datapoints=True):

    summary_name = normalize_summary_name(summary)

    data = []

    for direction in ["Down", "Up"]:

        _, pre_column, post_column = get_adaptation_columns(
            perturbation_results, summary, method, direction, sd_factor
        )

        direction_data = perturbation_results[
            ["ID", "Group", pre_column, post_column]
        ].copy()

        direction_data["Difference"] = (
            direction_data[post_column] -
            direction_data[pre_column]
        )

        direction_data["Direction"] = direction

        data.append(
            direction_data[
                ["ID", "Group", "Direction", "Difference"]
            ]
        )

    data = pd.concat(data, ignore_index=True)

    group_order = list(group_colors.keys())

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.boxplot(
        data=data,
        x="Direction",
        y="Difference",
        hue="Group",
        order=["Down", "Up"],
        hue_order=group_order,
        palette=group_colors,
        width=0.65,
        showfliers=not show_datapoints,
        ax=ax
    )

    if show_datapoints:
        sns.stripplot(
            data=data,
            x="Direction",
            y="Difference",
            hue="Group",
            order=["Down", "Up"],
            hue_order=group_order,
            palette=group_colors,
            dodge=True,
            jitter=True,
            size=6,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            ax=ax
        )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7
    )

    ax.set_xlabel("Perturbation Direction", fontsize=14)
    ax.set_ylabel(
        "Change in Adaptation Time (Post - Pre) (s)",
        fontsize=14
    )

    ax.set_title(
        f"Change in {summary_name} Adaptation Time",
        fontsize=16
    )

    ax.tick_params(axis="both", labelsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles[:len(group_order)],
        labels[:len(group_order)],
        title="Group",
        frameon=False,
        fontsize=11,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.show()

def plot_perturbation_percentage_boxplots(perturbation_results, summary, method, group_colors, sd_factor, show_datapoints=True):

    summary_name = normalize_summary_name(summary)
    df = perturbation_results[["ID", "Group"]].copy()

    if df["ID"].duplicated().any():
        raise ValueError("Duplicate participant IDs found.")

    for direction in ["Up", "Down"]:

        _, pre_column, post_column = get_adaptation_columns(
            perturbation_results, summary, method, direction, sd_factor
        )

        pre = pd.to_numeric(perturbation_results[pre_column], errors="coerce")
        post = pd.to_numeric(perturbation_results[post_column], errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            percentage = (pre - post) / pre * 100

        df[f"Percentage of change for {direction.lower()}"] = percentage.where((pre > 0) & (post >= 0) & np.isfinite(percentage))

    # Sort participants
    df["_order"] = pd.Categorical(df["Group"], categories=["Pink", "Sine", "White"], ordered=True)
    df["_number"] = pd.to_numeric(df["ID"].astype(str).str.extract(r"_(\d+)$")[0], errors="coerce")
    df = df.sort_values(["_order", "_number", "ID"]).drop(columns=["_order", "_number"]).reset_index(drop=True)

    # Convert to long format for plotting
    plot_data = df.melt(
        id_vars=["ID", "Group"],
        value_vars=["Percentage of change for up", "Percentage of change for down"],
        var_name="Direction",
        value_name="Percentage Change"
    )

    plot_data["Direction"] = plot_data["Direction"].replace({
        "Percentage of change for up": "Up",
        "Percentage of change for down": "Down"
    })

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.boxplot(
        data=plot_data, x="Direction", y="Percentage Change",
        hue="Group", order=["Down", "Up"],
        hue_order=list(group_colors), palette=group_colors,
        width=0.65, showfliers=not show_datapoints, ax=ax
    )

    if show_datapoints:
        sns.stripplot(
            data=plot_data, x="Direction", y="Percentage Change",
            hue="Group", order=["Down", "Up"],
            hue_order=list(group_colors), palette=group_colors,
            dodge=True, jitter=True, size=6, alpha=0.75,
            edgecolor="black", linewidth=0.5, ax=ax
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Perturbation Direction", fontsize=14)
    ax.set_ylabel("Improvement in Adaptation Time (%)", fontsize=14)
    ax.set_title(f"Percentage Change in {summary_name} Adaptation Time", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(group_colors)], labels[:len(group_colors)], title="Group", frameon=False)

    plt.tight_layout()
    plt.show()

    return df

def export_adaptation_results(perturbation_results, results_directory, sd_factor=2, method="SD", summary="Min"):
    summary_name = normalize_summary_name(summary)
    summary_label = "Min" if summary_name == "Minimum" else summary_name
    label = summary_label.lower()
    group_numbers = {"Pink": 1, "White": 2, "Sine": 3}

    df = pd.DataFrame(index=perturbation_results.index)
    df["Actual ID"] = perturbation_results["ID"]
    df["Group name"] = perturbation_results["Group"]
    df["Group number"] = df["Group name"].map(group_numbers)

    if df["Group number"].isna().any():
        raise ValueError("Unknown group name found in the data")

    for direction in ["Down", "Up"]:
        _, pre_column, post_column = get_adaptation_columns(perturbation_results, summary, method, direction, sd_factor)
        pre = pd.to_numeric(perturbation_results[pre_column], errors="coerce")
        post = pd.to_numeric(perturbation_results[post_column], errors="coerce")
        direction_lower = direction.lower()

        df[f"{summary_label} of {direction_lower} pre"] = pre
        df[f"{summary_label} of {direction_lower} post"] = post
        df[f"Difference of {label} {direction_lower}"] = post - pre

        with np.errstate(divide="ignore", invalid="ignore"):
            percentage = (pre - post) / pre * 100

        df[f"Percentage of difference {label} {direction_lower}"] = percentage.where((pre > 0) & (post >= 0) & np.isfinite(percentage))

    column_order = [
        "Actual ID", "Group name", "Group number",
        f"Difference of {label} down", f"Difference of {label} up",
        f"Percentage of difference {label} down", f"Percentage of difference {label} up",
        f"{summary_label} of down pre", f"{summary_label} of up pre",
        f"{summary_label} of down post", f"{summary_label} of up post"
    ]
    df = df[column_order]

    file_name = f"{summary_name}_Adaptation_Results_{method}.xlsx"
    output_path = os.path.join(results_directory, file_name)
    df.to_excel(output_path, index=False)

    print(f"Excel saved to: {output_path}")
    return df


def export_min_adaptation_results(perturbation_results, results_directory, sd_factor=2, method="SD", summary="Min"):
    """Backward-compatible name. Set summary='Median' or 'Average' to export those measures."""
    return export_adaptation_results(perturbation_results, results_directory, sd_factor, method, summary)


# Parameters
results_directory = r"C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"

sd_factor = 3
asymptote_fraction = 0.95
summary = "Median"  # Change to "Min" or "Average" as needed
method = "SD"

group_colors = {
    "White": "#BDBDBD",
    "Pink": "#E78AC3",
    "Sine": "#4C78A8"
}


# Load data
file_name = f"Perturbation_results_SD_factor_{sd_factor}_Asymptote_{asymptote_fraction}.xlsx"
perturbation_results = pd.read_excel(os.path.join(results_directory, file_name))


# plot_perturbation_group_subplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     direction="Down",
#     group_colors=group_colors,
#     show_datapoints=True,
#     sd_factor=sd_factor,
#     connect_participants=True,
#     show_ID=True
# )
# plot_perturbation_group_subplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     direction="Up",
#     group_colors=group_colors,
#     show_datapoints=True,
#     sd_factor=sd_factor,
#     connect_participants=True,
#     show_ID=True
# )
#
# plot_perturbation_boxplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     direction="Down",
#     group_colors=group_colors,
#     show_datapoints=False,
#     sd_factor=sd_factor,
#     connect_participants=False,
#     show_ID=False
# )
# plot_perturbation_boxplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     direction="Up",
#     group_colors=group_colors,
#     show_datapoints=False,
#     sd_factor=sd_factor,
#     connect_participants=False,
#     show_ID=False
# )
#
#
# plot_perturbation_difference_boxplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     group_colors=group_colors,
#     show_datapoints=False,
#     sd_factor=sd_factor
#
# )
#
# plot_perturbation_percentage_boxplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     group_colors=group_colors,
#     sd_factor=sd_factor,
#     show_datapoints=True
# )

# Plot percentage improvement for the selected summary (Median by default)
percentage_change_df = plot_perturbation_percentage_boxplots(
    perturbation_results, summary=summary, method=method,
    group_colors=group_colors, sd_factor=sd_factor, show_datapoints=True
)

# Create and save the 11-column Excel for the selected summary
# adaptation_summary_df = export_adaptation_results(
#     perturbation_results, results_directory,
#     sd_factor=sd_factor, method=method, summary=summary
# )

