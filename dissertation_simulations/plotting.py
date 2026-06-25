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
    yticks=None
):
    """
    Load multiple saved summary JSON files and plot their exponent distributions
    as box plots on one axis.

    Parameters
    ----------
    summary_paths : list of str
        Paths to saved JSON summary files.

    labels : list of str
        Labels for each box plot.

    true_mean_exp : float or None
        Optional horizontal reference line.

    distribution_index : int
        Which distribution to plot from each file.
        Use 0 if each file only contains one grid size.

    title : str
        Plot title.

    ylabel : str
        Y-axis label.

    figsize : tuple
        Figure size.

    ylim : tuple or None
        Optional y-axis limits, e.g. (-2.2, -0.8).

    yticks : list or array or None
        Optional y-axis tick values.
    """
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from dissertation_simulations.loading_summaries import load_distribution_summary

    if true_mean_exp is not None:
        true_mean_exp = np.mean(true_mean_exp)

    plot_data = []

    for path in summary_paths:
        electrodes, distributions = load_distribution_summary(path)
        plot_data.append(distributions[distribution_index])

    fig, ax = plt.subplots(figsize=figsize)

    ax.boxplot(
        plot_data,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True
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
            label="Outliers"
        )
    ]

    if true_mean_exp is not None:
        ax.axhline(
            true_mean_exp,
            color="black",
            linestyle="--",
            linewidth=1
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Estimated true mean exponent"
            )
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.legend(handles=legend_handles)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return fig, ax