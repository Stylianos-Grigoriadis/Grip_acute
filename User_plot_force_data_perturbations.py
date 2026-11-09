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

def plot_perturbation_percentage_boxplots(perturbation_results, summary, method, group_colors, sd_factor, show_datapoints=True, show_id=False):
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

        df[f"Percentage of change for {direction.lower()}"] = percentage.where(
            (pre > 0) & (post >= 0) & np.isfinite(percentage)
        )

    # Sort participants
    df["_order"] = pd.Categorical(
        df["Group"],
        categories=["Pink", "Sine", "White"],
        ordered=True
    )
    df["_number"] = pd.to_numeric(
        df["ID"].astype(str).str.extract(r"_(\d+)$")[0],
        errors="coerce"
    )
    df = (
        df.sort_values(["_order", "_number", "ID"])
        .drop(columns=["_order", "_number"])
        .reset_index(drop=True)
    )

    # Convert to long format for plotting
    plot_data = df.melt(
        id_vars=["ID", "Group"],
        value_vars=[
            "Percentage of change for up",
            "Percentage of change for down"
        ],
        var_name="Direction",
        value_name="Percentage Change"
    )

    plot_data["Direction"] = plot_data["Direction"].replace({
        "Percentage of change for up": "Up",
        "Percentage of change for down": "Down"
    })

    direction_order = ["Down", "Up"]
    group_order = list(group_colors)

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.boxplot(
        data=plot_data,
        x="Direction",
        y="Percentage Change",
        hue="Group",
        order=direction_order,
        hue_order=group_order,
        palette=group_colors,
        width=0.65,
        showfliers=not show_datapoints,
        ax=ax
    )

    if show_datapoints:
        # Exclude missing values so IDs match the plotted points.
        point_data = plot_data.dropna(subset=["Percentage Change"])
        collection_start = len(ax.collections)

        sns.stripplot(
            data=point_data,
            x="Direction",
            y="Percentage Change",
            hue="Group",
            order=direction_order,
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

        if show_id:
            # Seaborn creates one point collection per direction/group.
            point_collections = ax.collections[collection_start:]

            subsets = [
                point_data.loc[
                    (point_data["Direction"] == direction)
                    & (point_data["Group"] == group)
                ]
                for direction in direction_order
                for group in group_order
            ]

            for collection, subset in zip(point_collections, subsets):
                for (x, y), participant_id in zip(
                    collection.get_offsets(), subset["ID"]
                ):
                    ax.annotate(
                        str(participant_id),
                        xy=(x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        ha="left",
                        va="bottom"
                    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Perturbation Direction", fontsize=14)
    ax.set_ylabel("Improvement in Adaptation Time (%)", fontsize=14)
    ax.set_title(
        f"Percentage Change in {summary_name} Adaptation Time",
        fontsize=16
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(group_colors)],
        labels[:len(group_colors)],
        title="Group",
        frameon=False
    )

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


def plot_perturbation_rainclouds(
    perturbation_results, summary, method, group_colors, sd_factor,
    direction="Down", plot_type="pre_post", show_datapoints=True,
    connect_participants=False, show_ID=False, random_seed=1, vibrant_colors=True,
    orientation="horizontal", show_boxplot=True
):
    """Plot horizontal rainclouds, with a separate row per group and condition.

    plot_type: 'pre_post', 'difference' (Post - Pre), or 'percentage'
    ((Pre - Post) / Pre * 100). Direction accepts 'Down', 'Up', or 'Both';
    both directions remain separate rows on one graph. Returns (fig, axes, data).
    orientation: horizontal or vertical. show_boxplot toggles the boxes.
    Requires ptitprince (pip install ptitprince).
    """
    orientation = str(orientation).strip().lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    horizontal = orientation == "horizontal"

    def coordinates(values, positions):
        return (values, positions) if horizontal else (positions, values)

    try:
        import ptitprince as pt
    except ImportError as exc:
        raise ImportError(
            "Raincloud plots require ptitprince. Install it in your plotting "
            "environment with: pip install ptitprince"
        ) from exc

    if plot_type not in {"pre_post", "difference", "percentage"}:
        raise ValueError("plot_type must be 'pre_post', 'difference', or 'percentage'")
    if not group_colors:
        raise ValueError("group_colors must contain at least one group")
    if perturbation_results["ID"].duplicated().any():
        raise ValueError("Duplicate participant IDs found.")
    if perturbation_results["Group"].isna().any():
        raise ValueError("Missing participant group found.")
    unknown = set(perturbation_results["Group"]) - set(group_colors)
    if unknown:
        raise ValueError(f"Missing group colours for: {unknown}")

    # Copy the palette so other plots retain the caller's original colours.
    group_colors = dict(group_colors)
    if vibrant_colors:
        vivid = {"Pink": "#ED4596", "Sine": "#2587DF", "White": "#E6A126"}
        group_colors.update({g: vivid[g] for g in group_colors if g in vivid})
    summary_name = normalize_summary_name(summary)
    frames = []
    direction = str(direction).strip().capitalize()
    if direction not in {"Down", "Up", "Both"}:
        raise ValueError("direction must be 'Down', 'Up', or 'Both'")
    directions = ["Down", "Up"] if direction == "Both" else [direction]
    for current_direction in directions:
        _, pre_column, post_column = get_adaptation_columns(
            perturbation_results, summary, method, current_direction, sd_factor
        )
        pre = pd.to_numeric(perturbation_results[pre_column], errors="coerce")
        post = pd.to_numeric(perturbation_results[post_column], errors="coerce")
        base = perturbation_results[["ID", "Group"]].copy()
        if plot_type == "pre_post":
            for condition, values in [("Pre", pre), ("Post", post)]:
                frame = base.copy()
                frame["Condition"] = f"{current_direction} {condition}" if direction == "Both" else condition
                frame["Value"] = values
                frames.append(frame)
        else:
            base["Condition"] = current_direction
            if plot_type == "difference":
                base["Value"] = post - pre
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    percentage = (pre - post) / pre * 100
                base["Value"] = percentage.where(
                    (pre > 0) & (post >= 0) & np.isfinite(percentage)
                )
            frames.append(base)

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[np.isfinite(data["Value"])].copy()
    if data.empty:
        raise ValueError("No finite values available for this raincloud plot.")

    if plot_type == "pre_post":
        order = ([f"{d} {c}" for d in directions for c in ["Pre", "Post"]]
                 if direction == "Both" else ["Pre", "Post"])
    else:
        order = directions
    group_order = list(group_colors)
    # Each condition has its own row, preserving the group colour throughout.
    rows = [(group, condition) for group in group_order for condition in order]
    row_names = [f"{group} | {condition}" for group, condition in rows]
    row_lookup = {row: name for row, name in zip(rows, row_names)}
    data["Row"] = [row_lookup[(g, c)] for g, c in zip(data["Group"], data["Condition"])]
    figsize = (11, max(5.5, 1.12 * len(rows))) if horizontal else (max(9, 1.3 * len(rows)), 7)
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    axes = np.array([ax])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for group_index, group in enumerate(group_order):
        if group_index % 2 == 0:
            first = group_index * len(order)
            span = ax.axhspan if horizontal else ax.axvspan
            span(first - 0.55, first + len(order) - 0.45,
                       color="#F3F6FA", zorder=0)
    rng = np.random.default_rng(random_seed)

    # Draw clouds above the rows; points and boxes sit just below them.
    # Constant/singleton samples still get boxes and points, without a KDE.
    for position, (group, condition) in enumerate(rows):
        subset = data.loc[(data["Group"] == group) & (data["Condition"] == condition)]
        if subset["Value"].nunique() > 1:
            pt.half_violinplot(
                data=subset, x="Value" if horizontal else "Row",
                y="Row" if horizontal else "Value", order=row_names,
                color=group_colors[group], orient="h" if horizontal else "v", inner=None,
                width=0.85, cut=0, linewidth=0.8, saturation=1, ax=ax
            )

    for group in group_order:
        group_data = data.loc[data["Group"] == group]
        offsets = {participant_id: rng.uniform(0.11, 0.30)
                   for participant_id in group_data["ID"].unique()}
        for condition in order:
            position = rows.index((group, condition))
            subset = group_data.loc[group_data["Condition"] == condition]
            if subset.empty:
                if horizontal:
                    ax.text(0.98, position, "No valid data", transform=ax.get_yaxis_transform(), ha="right", va="center")
                else:
                    ax.text(position, 0.98, "No valid data", transform=ax.get_xaxis_transform(), ha="center", va="top")
                continue
            if show_boxplot:
                style = {"color": "#536174", "linewidth": 1.15}
                ax.boxplot(
                    [subset["Value"].to_numpy()], vert=not horizontal,
                    positions=[position + 0.20], widths=0.22, patch_artist=True,
                    manage_ticks=False, showfliers=not show_datapoints,
                    showcaps=False, boxprops={"facecolor": "white", "edgecolor": "#536174", "linewidth": 1.15, "alpha": 0.85}, whiskerprops=style,
                    medianprops={"color": "#27364B", "linewidth": 1.9}, zorder=4
                )
            if show_datapoints:
                y = position + subset["ID"].map(offsets).to_numpy()
                ax.scatter(*coordinates(subset["Value"], y), color=group_colors[group],
                           s=46, alpha=0.85, linewidth=0.65,
                           edgecolor="white", zorder=3)
                if show_ID:
                    for value, y_value, participant_id in zip(subset["Value"], y, subset["ID"]):
                        ax.annotate(str(participant_id), coordinates(value, y_value),
                                    xytext=(4, 4), textcoords="offset points", fontsize=8)

        if connect_participants and plot_type == "pre_post":
            for current_direction in directions:
                pair_order = ([f"{current_direction} Pre", f"{current_direction} Post"]
                              if direction == "Both" else ["Pre", "Post"])
                for participant_id, participant in group_data.groupby("ID", sort=False):
                    paired = participant.set_index("Condition")["Value"].reindex(pair_order)
                    if paired.notna().all():
                        y = [rows.index((group, c)) + offsets[participant_id] for c in pair_order]
                        ax.plot(*coordinates(paired.to_numpy(), y), color=group_colors[group],
                                alpha=0.25, linewidth=1, zorder=1)

    labels = {
        "pre_post": "Adaptation Time (s)",
        "difference": "Change in Adaptation Time (Post - Pre) (s)",
        "percentage": "Improvement in Adaptation Time (%)"
    }
    titles = {
        "pre_post": f"{summary_name} Adaptation Time",
        "difference": f"Change in {summary_name} Adaptation Time",
        "percentage": f"Percentage Change in {summary_name} Adaptation Time"
    }
    if plot_type != "pre_post":
        zero_line = ax.axvline if horizontal else ax.axhline
        zero_line(0, color="#7A8797", linestyle=(0, (4, 4)), linewidth=1, alpha=0.7)
    set_ticks = ax.set_yticks if horizontal else ax.set_xticks
    set_ticks(range(len(rows)))
    # A single direction only needs group names, as in the reference image.
    tick_labels = []
    for group, condition in rows:
        n = len(data.loc[(data["Group"] == group) & (data["Condition"] == condition)])
        label = group if len(order) == 1 else f"{group}  ·  {condition}"
        tick_labels.append(f"{label}   (n={n})")
    if horizontal:
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    category_ticks = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    for tick, (group, _) in zip(category_ticks, rows):
        tick.set_color(group_colors[group])
        tick.set_fontweight("semibold")
    if horizontal:
        ax.set_ylim(len(rows) - 0.45, -0.65)
        ax.set_xlabel(labels[plot_type], fontsize=14)
        ax.set_ylabel("")
    else:
        ax.set_xlim(-0.65, len(rows) - 0.45)
        ax.set_ylabel(labels[plot_type], fontsize=14)
        ax.set_xlabel("")
    direction_label = "Down and Up" if direction == "Both" else direction
    ax.set_title(f"{titles[plot_type]}\n{direction_label} perturbations", fontsize=17, fontweight="semibold", color="#27364B", loc="left", pad=20)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    category_axis = "y" if horizontal else "x"
    value_axis = "x" if horizontal else "y"
    ax.spines["left" if horizontal else "bottom"].set_visible(False)
    ax.spines["bottom" if horizontal else "left"].set_color("#D5DCE5")
    ax.tick_params(axis=value_axis, colors="#536174")
    ax.tick_params(axis=category_axis, length=0, pad=12)
    (ax.xaxis if horizontal else ax.yaxis).label.set_color("#27364B")
    ax.grid(axis=value_axis, color="#DCE3EC", linewidth=0.75, alpha=0.65)
    ax.margins(**{value_axis: 0.08})
    fig.tight_layout()
    plt.show()
    return fig, axes, data



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
#     show_datapoints=True,
#     sd_factor=sd_factor,
#     connect_participants=True,
#     show_ID=True
# )
# plot_perturbation_boxplots(
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
#
# plot_perturbation_difference_boxplots(
#     perturbation_results,
#     summary=summary,
#     method=method,
#     group_colors=group_colors,
#     show_datapoints=True,
#     sd_factor=sd_factor
#
# )

# Plot percentage improvement for the selected summary (Median by default)
percentage_change_df = plot_perturbation_percentage_boxplots(
    perturbation_results, summary=summary, method=method,
    group_colors=group_colors, sd_factor=sd_factor, show_datapoints=True, show_id=False
)

fig, axes, raincloud_data = plot_perturbation_rainclouds(
    perturbation_results,
    summary=summary,
    method=method,
    group_colors=group_colors,
    sd_factor=sd_factor,
    direction="Down",
    plot_type="percentage",
    orientation="vertical",  # "vertical" or "horizontal"
    show_boxplot=False,      # True to display boxplots
    show_datapoints=True,
    show_ID=False
)

# Create and save the 11-column Excel for the selected summary
adaptation_summary_df = export_adaptation_results(
    perturbation_results, results_directory,
    sd_factor=sd_factor, method=method, summary=summary
)

