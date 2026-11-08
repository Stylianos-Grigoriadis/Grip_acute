import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

results_directory = r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training older adults\Results"

sd_factor = 3
asymptote_fraction = 0.95
summary = "Median"  # Change to "Min" or "Average" as needed
method = "SD"




# Statistical functions: install scipy, statsmodels and pingouin in your environment.

def load_perturbation_results(results_directory, sd_factor=3, asymptote_fraction=0.95):
    """Read the same results workbook used by the plotting script."""
    filename = f"Perturbation_results_SD_factor_{sd_factor}_Asymptote_{asymptote_fraction}.xlsx"
    return pd.read_excel(os.path.join(results_directory, filename))


def _prepare_perturbation_statistics(perturbation_results, summary, method, direction, sd_factor):
    summary_map = {"Min": "Minimum", "Minimum": "Minimum", "Average": "Average", "Median": "Median"}
    if summary not in summary_map:
        raise ValueError("summary must be Min, Minimum, Average, or Median")
    if direction not in {"Up", "Down"}:
        raise ValueError("Choose one direction: 'Up' or 'Down'")
    if method not in {"SD", "Asymptote"}:
        raise ValueError("method must be SD or Asymptote")
    if summary_map[summary] == "Median" and method == "Asymptote":
        raise ValueError("Median adaptation times are available only for SD")
    suffix = f"SD {sd_factor}" if method == "SD" else "Asymptote"
    pre = f"{summary_map[summary]} Pre {direction} Adaptation Time {suffix}"
    post = f"{summary_map[summary]} Post {direction} Adaptation Time {suffix}"
    columns = ["ID", "Group", pre, post]
    missing = [column for column in columns if column not in perturbation_results]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = perturbation_results[columns].copy().rename(columns={pre: "Pre", post: "Post"})
    if data["ID"].isna().any() or data["ID"].duplicated().any():
        raise ValueError("Participant IDs must be present and unique")
    for column in ["Pre", "Post"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data["Group"].notna() & np.isfinite(data["Pre"]) & np.isfinite(data["Post"])
    excluded = data.loc[~valid].copy()
    data = data.loc[valid].copy()
    if (data[["Pre", "Post"]] < 0).any().any():
        raise ValueError("Negative adaptation times found; check the source data")
    sizes = data.groupby("Group", observed=True).size()
    if len(sizes) < 2 or (sizes < 3).any():
        raise ValueError("At least two groups with three complete participants each are required")
    if set(perturbation_results["Group"].dropna()) - set(sizes.index):
        raise ValueError("A group has no complete Pre/Post pairs")
    data["Change"] = data["Post"] - data["Pre"]
    return data, excluded


def _perturbation_pairwise(model, prediction_data, alpha):
    """Model-based differences, individual CIs and Bonferroni-adjusted pairwise p-values."""
    from itertools import combinations
    from patsy import build_design_matrices
    from statsmodels.stats.multitest import multipletests
    design = np.asarray(build_design_matrices([model.model.data.design_info], prediction_data)[0])
    rows = []
    for i, j in combinations(range(len(prediction_data)), 2):
        test = model.t_test(design[i] - design[j])
        ci = np.asarray(test.conf_int(alpha=alpha)).ravel()
        rows.append({"Group A": prediction_data.iloc[i]["Group"],
                     "Group B": prediction_data.iloc[j]["Group"],
                     "A - B": np.asarray(test.effect).item(),
                     "SE": np.asarray(test.sd).item(),
                     "CI lower": ci[0], "CI upper": ci[1],
                     "p uncorrected": np.asarray(test.pvalue).item()})
    result = pd.DataFrame(rows)
    result["p Bonferroni"] = multipletests(result["p uncorrected"], alpha=alpha, method="bonferroni")[1]
    return result


def _perturbation_diagnostics(model, data, column):
    from scipy import stats
    residuals = np.asarray(model.resid)
    shapiro_p = stats.shapiro(residuals).pvalue if 3 <= len(residuals) <= 5000 and np.ptp(residuals) > 0 else np.nan
    samples = [frame[column].to_numpy() for _, frame in data.groupby("Group", observed=True)]
    levene_p = stats.levene(*samples, center="median").pvalue
    return pd.Series({"Shapiro residual p": shapiro_p, "Levene p (median-centred)": levene_p})


def _print_perturbation_statistics(result, title):
    print(f"\n{title}")
    print(f"Complete participants: {len(result['data'])}; excluded: {len(result['excluded'])}")
    print("\nDescriptive statistics:\n", result["descriptives"].to_string())
    print("\nANOVA table:\n", result["anova"].to_string())
    print("\nBetween-group comparisons:\n", result["pairwise"].to_string(index=False))
    print("\nCIs are individual intervals; Bonferroni correction covers pairwise p-values in this call only.")
    print("\nDiagnostics:\n", result["diagnostics"].to_string())


def run_perturbation_mixed_anova(
    perturbation_results, summary="Median", method="SD", direction="Down",
    sd_factor=3, alpha=0.05, verbose=True
):
    """Group x Time mixed ANOVA for ONE direction, using complete Pre/Post pairs.

    Returns tables, data, exclusions and a change-score model for diagnostics.
    Pairwise tests compare Post-minus-Pre changes between groups. With two
    time points, this is the between-group component of the interaction.
    Unequal group sizes use Pingouin's Type II mixed ANOVA. No sphericity
    correction is needed with two time points. Cross-direction/method testing
    is not multiplicity-adjusted here. Do not treat non-adaptation as ordinary
    missingness without considering censoring.
    """
    import pingouin as pg
    from statsmodels.formula.api import ols
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    data, excluded = _prepare_perturbation_statistics(perturbation_results, summary, method, direction, sd_factor)
    long_data = data.melt(id_vars=["ID", "Group"], value_vars=["Pre", "Post"],
                          var_name="Time", value_name="AdaptationTime")
    table = pg.mixed_anova(data=long_data, dv="AdaptationTime", within="Time",
                           between="Group", subject="ID", effsize="np2")
    model = ols("Change ~ C(Group)", data=data).fit()
    predictions = pd.DataFrame({"Group": sorted(data["Group"].unique())})
    result = {
        "anova": table,
        "pairwise": _perturbation_pairwise(model, predictions, alpha),
        "descriptives": data.groupby("Group", observed=True)[["Pre", "Post", "Change"]].agg(["count", "mean", "std", "median"]),
        "diagnostics": _perturbation_diagnostics(model, data, "Change"),
        "data": data, "long_data": long_data, "excluded": excluded,
        "change_model": model
    }
    if verbose:
        _print_perturbation_statistics(result, f"{direction}: Group x Time mixed ANOVA ({summary}, {method})")
    return result


def run_perturbation_ancova(
    perturbation_results, summary="Median", method="SD", direction="Down",
    sd_factor=3, alpha=0.05, verbose=True
):
    """ONE-direction ANCOVA: Post ~ Group + Pre, with a common baseline slope.

    Adjusted means are predictions at the pooled mean Pre value. Returns the
    fitted model, Type II table, adjusted means, contrasts and diagnostics.
    The slope-interaction test is diagnostic; do not automatically choose a
    model based only on its p-value. Inspect residuals and baseline overlap.
    CIs are individual; multiplicity adjustment covers pairwise p-values only
    within this call. Missing Pre/Post pairs are explicitly returned.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    data, excluded = _prepare_perturbation_statistics(perturbation_results, summary, method, direction, sd_factor)
    for group, frame in data.groupby("Group", observed=True):
        if frame["Pre"].nunique() < 2:
            raise ValueError(f"Baseline variation is required within group {group}")
    model = ols("Post ~ C(Group) + Pre", data=data).fit()
    slope_model = ols("Post ~ C(Group) * Pre", data=data).fit()
    if np.linalg.matrix_rank(slope_model.model.exog) < slope_model.model.exog.shape[1]:
        raise ValueError("The slope-interaction model is not identifiable")
    table = anova_lm(model, typ=2)
    error_ss = table.loc["Residual", "sum_sq"]
    table["partial_eta_squared"] = table["sum_sq"] / (table["sum_sq"] + error_ss)
    table.loc["Residual", "partial_eta_squared"] = np.nan
    predictions = pd.DataFrame({"Group": sorted(data["Group"].unique()), "Pre": data["Pre"].mean()})
    adjusted = predictions.join(model.get_prediction(predictions).summary_frame(alpha=alpha)[["mean", "mean_se", "mean_ci_lower", "mean_ci_upper"]])
    diagnostic_data = data.copy()
    diagnostic_data["Residual"] = model.resid
    diagnostics = _perturbation_diagnostics(model, diagnostic_data, "Residual")
    diagnostics["Group x Pre slope interaction p"] = anova_lm(model, slope_model).iloc[1]["Pr(>F)"]
    result = {
        "anova": table, "adjusted_means": adjusted,
        "pairwise": _perturbation_pairwise(model, predictions, alpha),
        "descriptives": data.groupby("Group", observed=True)[["Pre", "Post", "Change"]].agg(["count", "mean", "std", "median"]),
        "diagnostics": diagnostics, "model": model, "slope_model": slope_model,
        "data": data, "excluded": excluded
    }
    if verbose:
        _print_perturbation_statistics(result, f"{direction}: baseline-adjusted ANCOVA ({summary}, {method})")
        print("\nAdjusted Post means at the pooled baseline mean:\n", adjusted.to_string(index=False))
    return result


# Example calls (uncomment only when needed):
perturbation_results = load_perturbation_results(results_directory, sd_factor, asymptote_fraction)
# mixed_down = run_perturbation_mixed_anova(perturbation_results, summary, method, "Down", sd_factor)
# mixed_up = run_perturbation_mixed_anova(perturbation_results, summary, method, "Up", sd_factor)
ancova_down = run_perturbation_ancova(perturbation_results, summary, method, "Down", sd_factor)
# ancova_up = run_perturbation_ancova(perturbation_results, summary, method, "Up", sd_factor)
