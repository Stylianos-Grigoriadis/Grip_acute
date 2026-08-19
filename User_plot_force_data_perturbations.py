import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


################################
##### Load perturbation data #####
################################

results_directory = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"

sd_factor = 3
asymptote_fraction = 0.95

file_name = f"Perturbation_results_SD_factor_{sd_factor}_Asymptote_{asymptote_fraction}.xlsx"

perturbation_results = pd.read_excel(
    results_directory + "\\" + file_name
)


########################
##### Group colors #####
########################

group_colors = {
    "White": "#BDBDBD",
    "Pink": "#E78AC3",
    "Sine": "#4C78A8"
}


#####################################
##### Perturbation boxplot function #####
#####################################

def plot_perturbation_boxplots(perturbation_results, summary, method, direction, group_colors, sd_factor, show_datapoints=True, connect_participants=False, show_ID=False):

    if summary == "Min":
        summary_name = "Minimum"
    elif summary == "Minimum":
        summary_name = "Minimum"
    elif summary == "Average":
        summary_name = "Average"
    else:
        raise ValueError("summary must be 'Average', 'Min', or 'Minimum'")

    if direction not in ["Up", "Down"]:
        raise ValueError("direction must be 'Up' or 'Down'")

    if method == "SD":
        pre_column = f"{summary_name} Pre {direction} Adaptation Time SD {sd_factor}"
        post_column = f"{summary_name} Post {direction} Adaptation Time SD {sd_factor}"

    elif method == "Asymptote":
        pre_column = f"{summary_name} Pre {direction} Adaptation Time Asymptote"
        post_column = f"{summary_name} Post {direction} Adaptation Time Asymptote"

    else:
        raise ValueError("method must be 'SD' or 'Asymptote'")

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


def plot_perturbation_difference_boxplots(perturbation_results, summary, method, group_colors, sd_factor, show_datapoints=True):

    if summary == "Min":
        summary_name = "Minimum"
    elif summary == "Minimum":
        summary_name = "Minimum"
    elif summary == "Average":
        summary_name = "Average"
    else:
        raise ValueError("summary must be 'Average', 'Min', or 'Minimum'")

    if method not in ["SD", "Asymptote"]:
        raise ValueError("method must be 'SD' or 'Asymptote'")

    data = []

    for direction in ["Down", "Up"]:

        if method == "SD":
            pre_column = f"{summary_name} Pre {direction} Adaptation Time SD {sd_factor}"
            post_column = f"{summary_name} Post {direction} Adaptation Time SD {sd_factor}"

        else:
            pre_column = f"{summary_name} Pre {direction} Adaptation Time Asymptote"
            post_column = f"{summary_name} Post {direction} Adaptation Time Asymptote"

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


plot_perturbation_boxplots(
    perturbation_results,
    summary="Min",
    method="SD",
    direction="Down",
    group_colors=group_colors,
    show_datapoints=True,
    sd_factor=sd_factor,
    connect_participants=True,
    show_ID=True
)
plot_perturbation_boxplots(
    perturbation_results,
    summary="Min",
    method="SD",
    direction="Up",
    group_colors=group_colors,
    show_datapoints=True,
    sd_factor=sd_factor,
    connect_participants=True,
    show_ID=True
)


plot_perturbation_difference_boxplots(
    perturbation_results,
    summary="Min",
    method="SD",
    group_colors=group_colors,
    show_datapoints=True,
    sd_factor=sd_factor

)

