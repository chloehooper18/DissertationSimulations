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

    Parameters
    ----------
    x_values : array-like
        X-axis values.

    mean_exps : array-like
        Mean estimated exponents.

    errors : array-like
        Bias values (estimated - true).

    true_exp : float or None
        True exponent value.

    std_exps : array-like or None
        Standard deviations for error bars.

    xlabel : str
        Label for x-axis.

    title_prefix : str
        Plot title prefix.
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # ---------------------------------
    # Plot bias
    # ---------------------------------
    if true_exp is not None:

        if std_exps is not None:

            plt.errorbar(
                x_values,
                errors,
                yerr=std_exps,
                marker='o',
                capsize=5
            )

        else:

            plt.plot(x_values, errors, marker='o')

        plt.axhline(0, linestyle='--')

        plt.xlabel(xlabel)
        plt.ylabel("Bias (Estimated - True)")
        plt.title(f"{title_prefix} error vs electrodes")

    # ---------------------------------
    # Plot means
    # ---------------------------------
    else:

        if std_exps is not None:

            plt.errorbar(
                x_values,
                mean_exps,
                yerr=std_exps,
                marker='o',
                capsize=5
            )

        else:

            plt.plot(x_values, mean_exps, marker='o')

        plt.xlabel(xlabel)
        plt.ylabel("Mean estimated exponent")
        plt.title(f"{title_prefix} vs electrodes")

    plt.show()
    return fig