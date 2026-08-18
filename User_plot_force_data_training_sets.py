import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

"Average Spatial Error"
"Variable Error"
"Normalized Average Spatial Error"
"Normalized Variable Error"
"SaEn"
"Sample Lag"
"Max Correlation"

group_colors = {
    "White": "#BDBDBD",
    "Pink": "#E78AC3",
    "Sine": "#4C78A8"
}

def plot_training_variable(training_results, variable, group_colors, show_datapoints=False):
    trials = np.arange(1, 11)

    variable_columns = [
        f"{variable} Training {trial}"
        for trial in trials
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for group in group_colors.keys():
        group_data = training_results[training_results["Group"] == group]

        mean_values = group_data[variable_columns].mean().to_numpy()
        sem_values = group_data[variable_columns].sem().to_numpy()

        if show_datapoints:
            for _, participant in group_data.iterrows():
                ax.scatter(
                    trials,
                    participant[variable_columns].to_numpy(dtype=float),
                    color=group_colors[group],
                    alpha=0.25,
                    s=25
                )

        ax.plot(
            trials,
            mean_values,
            marker="o",
            markersize=7,
            linewidth=2.5,
            color=group_colors[group],
            label=group
        )

        ax.fill_between(
            trials,
            mean_values - sem_values,
            mean_values + sem_values,
            color=group_colors[group],
            alpha=0.15
        )

    ax.set_xlabel("Training Set", fontsize=14)
    ax.set_ylabel(variable, fontsize=14)
    ax.set_title(f"{variable} Across Training", fontsize=16)

    ax.set_xticks(trials)
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(axis="y", alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        title="Group",
        frameon=False,
        fontsize=11,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.show()

def plot_training_variable_boxplots(training_results, variable, group_colors, show_datapoints=False):
    variable_columns = [
        f"{variable} Training {trial}"
        for trial in range(1, 11)
    ]

    data_long = training_results.melt(
        id_vars=["ID", "Group"],
        value_vars=variable_columns,
        var_name="Training",
        value_name=variable
    )

    data_long["Training"] = data_long["Training"].str.extract(r"(\d+)$").astype(int)

    group_order = list(group_colors.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.boxplot(
        data=data_long,
        x="Training",
        y=variable,
        hue="Group",
        hue_order=group_order,
        palette=group_colors,
        dodge=True,
        width=0.7,
        showfliers=False,
        ax=ax
    )

    if show_datapoints:
        sns.stripplot(
            data=data_long,
            x="Training",
            y=variable,
            hue="Group",
            hue_order=group_order,
            palette=group_colors,
            dodge=True,
            jitter=True,
            size=4,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
            ax=ax
        )

    ax.set_xlabel("Training Set", fontsize=14)
    ax.set_ylabel(variable, fontsize=14)
    ax.set_title(f"{variable} Across Training Sets", fontsize=16)

    ax.tick_params(axis="both", labelsize=12)

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



results_directory = r'C:\Users\Administrator\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results'
os.chdir(results_directory)
training_results = pd.read_excel(r'Training_results.xlsx')
print(training_results.columns)

plot_training_variable(training_results, "Average Spatial Error", group_colors, show_datapoints=False)
# plot_training_variable(training_results, "Variable Error", group_colors, show_datapoints=False)
plot_training_variable(training_results, "Normalized Average Spatial Error", group_colors, show_datapoints=False)
# plot_training_variable(training_results, "Normalized Variable Error", group_colors, show_datapoints=False)
plot_training_variable(training_results, "SaEn", group_colors, show_datapoints=False)


plot_training_variable_boxplots(training_results,"Average Spatial Error",  group_colors, show_datapoints=True)
# plot_training_variable_boxplots(training_results,"Variable Error",  group_colors, show_datapoints=True)
plot_training_variable_boxplots(training_results,"Normalized Average Spatial Error",  group_colors, show_datapoints=True)
# plot_training_variable_boxplots(training_results,"Normalized Variable Error",  group_colors, show_datapoints=True)
plot_training_variable_boxplots(training_results,"SaEn",  group_colors, show_datapoints=True)



