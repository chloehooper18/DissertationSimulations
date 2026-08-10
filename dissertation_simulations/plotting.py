def plot_sweep_results(
    x_values,
    mean_exps,
    errors,
    true_exp=None,
    std_exps=None,
    xlabel="Number of electrodes",
    title_prefix="Exponent estimation"
):
    """
    Plot sweep results with optional error bars.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    x_values = np.asarray(x_values)
    mean_exps = np.asarray(mean_exps)

    if std_exps is not None:
        std_exps = np.asarray(std_exps)

    if true_exp is not None:
        true_exp = np.mean(true_exp)

    errors = np.asarray(errors).squeeze()

    fig, ax = plt.subplots()

    # Plot bias
    if true_exp is not None:

        errors = mean_exps + true_exp

        if std_exps is not None:
            ax.errorbar(
                x_values,
                errors,
                yerr=std_exps,
                marker="o",
                capsize=5
            )
        else:
            ax.plot(x_values, errors, marker="o")

        ax.axhline(0, linestyle="--")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bias (Estimated - True)")
        ax.set_title(f"{title_prefix} error vs electrodes")

    # Plot means
    else:

        if std_exps is not None:
            ax.errorbar(
                x_values,
                mean_exps,
                yerr=std_exps,
                marker="o",
                capsize=5
            )
        else:
            ax.plot(x_values, mean_exps, marker="o")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean estimated exponent")
        ax.set_title(f"{title_prefix} vs electrodes")

    plt.show()

    return fig

# =============================
# =============================

def plot_summary_boxplots(
    summary_paths,
    labels,
    true_mean_exp=None,
    distribution_index=0,
    title="Exponent estimates by averaging method",
    ylabel="Estimated exponent",
    figsize=(7, 5),
    ylim=None,
    yticks=None,
    ax=None,
    show=True,
):
    """Plot saved exponent distributions as styled box plots."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from dissertation_simulations.loading_summaries import load_distribution_summary

    if true_mean_exp is not None:
        true_mean_exp = np.mean(true_mean_exp)

    plot_data = []
    for path in summary_paths:
        _, distributions = load_distribution_summary(path)
        plot_data.append(distributions[distribution_index])

    # Create a standalone figure only when an axis was not supplied.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.boxplot(
        plot_data,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        medianprops=dict(color="orange", linewidth=2),
        meanprops=dict(color="green", linestyle="--", linewidth=2),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=5,
            linestyle="none",
        ),
    )

    legend_handles = [
        Line2D([0], [0], color="orange", linewidth=2, label="Median"),
        Line2D([0], [0], color="green", linestyle="--", linewidth=2, label="Mean"),
        Line2D(
            [0], [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            linestyle="None",
            label="Outliers",
        ),
    ]

    if true_mean_exp is not None:
        ax.axhline(true_mean_exp, color="black", linestyle="--", linewidth=1)
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Estimated true mean exponent",
            )
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.legend(handles=legend_handles)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show:
        fig.tight_layout()
        plt.show()

    return fig, ax

# =======================================================================
# =======================================================================

def add_significance_bracket(ax, x1, x2, y, height, text):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color="black",
        linewidth=1,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )

# =======================================================================
# =======================================================================

def add_panel_tukey_brackets(ax, panel_data, base_ylim, labels):
    """Add significant within-panel Tukey brackets and preserve axis space."""

    from dissertation_simulations.plotting import tukey_with_cohens_d

    panel_tukey, panel_posthoc = tukey_with_cohens_d(
        panel_data,
        "method",
    )

    significant_results = panel_posthoc.loc[
        panel_posthoc["significant"]
    ].reset_index(drop=True)

    if significant_results.empty:
        return panel_posthoc

    method_positions = {
        label: position
        for position, label in enumerate(labels, start=1)
    }

    y_min, y_max = base_ylim
    y_range = y_max - y_min

    bracket_height = 0.025 * y_range
    bracket_spacing = 0.10 * y_range
    start_y = y_max + 0.04 * y_range

    for i, row in significant_results.iterrows():
        p_value = row["p_adjusted"]

        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        else:
            stars = "*"

        add_significance_bracket(
            ax=ax,
            x1=method_positions[row["group1"]],
            x2=method_positions[row["group2"]],
            y=start_y + i * bracket_spacing,
            height=bracket_height,
            text=stars,
        )

    # Reserve identical room for up to three comparisons.
    # This keeps Panels A and B on a matched final y-scale.
    ax.set_ylim(y_min, y_max + 0.38 * y_range)

    return panel_posthoc

# =======================================================================
# =======================================================================

def cohens_d(group1_values, group2_values):
    """Cohen's d: group 1 mean minus group 2 mean."""

    import numpy as np

    group1_values = np.asarray(group1_values)
    group2_values = np.asarray(group2_values)

    n1 = len(group1_values)
    n2 = len(group2_values)

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * np.var(group1_values, ddof=1)
            + (n2 - 1) * np.var(group2_values, ddof=1)
        ) / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(group1_values) - np.mean(group2_values)) / pooled_sd

# =======================================================================
# =======================================================================

def tukey_with_cohens_d(data, factor_column):
    """Run Tukey HSD and add Cohen's d to every comparison."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from dissertation_simulations.plotting import cohens_d
    import numpy as np
    import pandas as pd

    tukey = pairwise_tukeyhsd(
        endog=data["estimate"],
        groups=data[factor_column],
        alpha=0.05,
    )


    rows = []

    for row in tukey.summary().data[1:]:
        group1, group2 = row[0], row[1]

        values1 = data.loc[data[factor_column] == group1, "estimate"]
        values2 = data.loc[data[factor_column] == group2, "estimate"]

        rows.append({
            "group1": group1,
            "group2": group2,
            "mean_difference": float(row[2]),
            "p_adjusted": float(row[3]),
            "ci_lower": float(row[4]),
            "ci_upper": float(row[5]),
            "significant": str(row[6]).lower() == "true",
            "cohens_d": cohens_d(values1, values2),
        })

    return tukey, pd.DataFrame(rows)