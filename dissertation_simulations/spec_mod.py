def avg_specmod_exp(psds, freq_range):
    """
    Fit spectral models to PSDs and compute aperiodic exponents.
    Works when electrodes are generated as a 1D array

    Parameters
    ----------
    psds : dict
        {name: (freqs, powers)}
    freq_range : list or tuple
        Frequency range to fit (e.g., [1, 50])

    Returns
    -------
    spectral_models : dict
        {name: SpectralModel object}
    exponents : dict
        {name: exponent value}
    avg_exponent : float
        Mean exponent across signals
    """
    import numpy as np
    from specparam import SpectralModel

    spectral_models = {}
    exponents = {}

    for name, (freqs, powers) in psds.items():

        # Fit model
        fm = SpectralModel()
        fm.report(freqs, powers, freq_range)

        spectral_models[name] = fm

        # Extract exponent
        exponent = fm.get_params('aperiodic', 'exponent')
        exponents[name] = exponent

        print(f"{name}: {exponent:.3f}")

    # Average exponent
    avg_exponent = np.mean(list(exponents.values()))

    print(f"\nAverage exponent: {avg_exponent:.3f}")

    return spectral_models, exponents, avg_exponent

# ===========================================
# ===========================================

def avg_specmod_exponent_distribution(
    n_repeats,
    n_electrodes,
    plot=True,
    base_params=None,
    **param_overrides
):
    """
    Simulate electrode signals, compute individual PSDs, fit spectral models
    to each electrode separately, average the resulting aperiodic exponents,
    and return a distribution across repeated simulations.

    This approach differs from other pipelines in that spectral parameterization
    is performed at the single-electrode level before any averaging.

    Parameters
    ----------
    n_repeats : int
        Number of simulation iterations to run.

    n_electrodes : int
        Number of electrodes (signals) per simulation.

    plot : bool, optional, default: True
        If True, plot histogram of averaged exponent distribution.

    base_params : dict, optional
        Dictionary of default simulation parameters. If None, uses
        `electrode_sim_params_ap`.

    **param_overrides
        Keyword arguments used to override values in `base_params`.
        Example: exponent=-1, n_seconds=10, s_rate=500

    Returns
    -------
    all_mean_exponents : np.ndarray
        Mean exponent from each simulation iteration.

    mean_exp : float
        Mean of simulation-level averaged exponents.

    std_exp : float
        Standard deviation of simulation-level averaged exponents.

    Notes
    -----
    - PSDs are computed per electrode using Welch’s method.
    - Each PSD is fit independently with a spectral model.
    - Exponents are averaged across electrodes within each simulation.
    - Final output is a distribution of these per-simulation means.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        spectral_model_params,
        freq_range,
        welch_params
    )
    from dissertation_simulations.generate_electrodes import generate_1D_electrodes_ap

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    # -------------------------
    # Setup params
    # -------------------------
    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if "exponent" not in params:
        raise ValueError("You must specify 'exponent' in params or overrides.")

    all_mean_exponents = []

    # -------------------------
    # Simulation loop
    # -------------------------
    for _ in range(n_repeats):

        exponents = [params["exponent"]] * n_electrodes

        signals, _, _ = generate_1D_electrodes_ap(
            n_electrodes,
            exponents,
            params
        )

        welch_dict = welch_params(params)

        electrode_exponents = []

        # -------------------------
        # Fit each electrode individually
        # -------------------------
        for signal in signals.values():

            freqs, powers = compute_spectrum_welch(
                signal,
                fs=params["s_rate"],
                **welch_dict
            )

            fm = SpectralModel(**spectral_model_params)
            fm.fit(freqs, powers, freq_range)

            exp = fm.get_params('aperiodic', 'exponent')
            electrode_exponents.append(exp)

        # -------------------------
        # Average across electrodes
        # -------------------------
        all_mean_exponents.append(np.mean(electrode_exponents))

    # -------------------------
    # Summary stats
    # -------------------------
    all_mean_exponents = np.array(all_mean_exponents)
    mean_exp = np.mean(all_mean_exponents)
    std_exp = np.std(all_mean_exponents)

    # -------------------------
    # Plot distribution
    # -------------------------
    if plot:
        plt.hist(all_mean_exponents, bins=20, alpha=0.7)
        plt.axvline(mean_exp, linestyle='dashed')
        plt.title(
            f"Individual-PSD exponent distribution\n"
            f"(Exponent={params['exponent']}, electrodes={n_electrodes})"
        )
        plt.xlabel("Mean electrode exponent (per simulation)")
        plt.ylabel("Count")
        plt.show()

    return all_mean_exponents, float(mean_exp), float(std_exp)

# ===========================================
# ===========================================

def sweep_electrodes_specmod_1D(
    electrode_range,
    n_repeats,
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep number of electrodes and compute exponent estimation
    using individual-electrode spectral parameterization.

    Parameters
    ----------
    electrode_range : iterable
        e.g. range(1, 10)

    n_repeats : int
        Simulations per electrode count.

    base_params : dict or None
        Base simulation parameters.

    plot : bool
        Whether to plot error curve.

    return_full : bool
        If True, also returns full distributions.

    **param_overrides :
        ANY simulation parameter override
        (exponent, s_rate, etc.)

    Returns
    -------
    electrode_counts, mean_exps, std_exps, errors

    optionally:
        full_exponent_distributions
    """

    import matplotlib.pyplot as plt
    import numpy as np

    from dissertation_simulations.spec_mod import avg_specmod_exponent_distribution
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    # -----------------------------------
    # Sweep electrode counts
    # -----------------------------------
    for n_elec in electrode_range:

        all_exps, mean_exp, std_exp = (
            avg_specmod_exponent_distribution(
                n_repeats=n_repeats,
                n_electrodes=n_elec,
                base_params=base_params,
                plot=False,
                **param_overrides
            )
        )

        mean_exps.append(mean_exp)
        std_exps.append(std_exp)

        if true_exp is not None:

            errors.append(mean_exp - - true_exp)

        else:

            errors.append(np.nan)

        if return_full:

            all_distributions.append(all_exps)

    # Plotting function
    if plot:

        fig = plot_sweep_results(
            x_values=list(electrode_range),
            mean_exps=mean_exps,
            errors=errors,
            true_exp=true_exp,
            std_exps=std_exps,
            xlabel="Number of electrodes",
            title_prefix="1D spectral model exponent estimation"
        )

# Returns
    if return_full:

        return (
            np.array(list(electrode_range)),
            np.array(mean_exps),
            np.array(std_exps),
            np.array(errors),
            all_distributions
        )

    return (
        np.array(list(electrode_range)),
        np.array(mean_exps),
        np.array(std_exps),
        np.array(errors),
        fig
    )

# ===========================================
# ===========================================

def avg_specmod_exponent_distribution_2D(
    n_repeats,
    grid_shape,
    plot=True,
    base_params=None,
    fit_freq_range=None,
    verbose=False,
    **param_overrides
):
    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        spectral_model_params,
        freq_range,
        welch_params
    )

    from dissertation_simulations.generate_electrodes import (
        generate_2D_electrodes_ap
    )

    from dissertation_simulations.exponents import (
        generate_random_electrode_exponents
    )

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if fit_freq_range is None:
        fit_freq_range = freq_range

    if verbose:
        print("Using fit_freq_range:", fit_freq_range)

    if (
        "exponent" not in params
        and not params.get("random_exponents", False)
    ):
        raise ValueError(
            "You must specify 'exponent' in params or overrides."
        )

    all_mean_exponents = []

    n_rows, n_cols = grid_shape
    n_total = n_rows * n_cols

    for _ in range(n_repeats):

        if "exponent_matrix" in params:

            exponent_grid = params["exponent_matrix"]

        elif params.get("random_exponents", False):

            exponent_grid = generate_random_electrode_exponents(
                n_electrodes=n_total,
                exp_range=params["exp_range"],
                distribution=params.get("distribution", "uniform")
            )

            exponent_grid = np.array(exponent_grid).reshape(n_rows, n_cols)

        else:

            exponent_grid = np.full(
                (n_rows, n_cols),
                params["exponent"]
            )

        grid_signals, times = generate_2D_electrodes_ap(
            n_rows,
            n_cols,
            exponent_grid,
            params=params
        )

        welch_dict = welch_params(params)

        electrode_exponents = []

        for i in range(n_rows):
            for j in range(n_cols):

                signal = grid_signals[i, j]

                freqs, powers = compute_spectrum_welch(
                    signal,
                    fs=params["s_rate"],
                    **welch_dict
                )

                fm = SpectralModel(**spectral_model_params)

                fm.fit(
                    freqs,
                    powers,
                    fit_freq_range
                )

                exp = fm.get_params(
                    "aperiodic",
                    "exponent"
                )

                electrode_exponents.append(exp)

        all_mean_exponents.append(
            np.mean(electrode_exponents)
        )

    all_mean_exponents = np.array(all_mean_exponents)

    mean_exp = np.mean(all_mean_exponents)
    std_exp = np.std(all_mean_exponents)

    if plot:

        plt.hist(
            all_mean_exponents,
            bins=20,
            alpha=0.7
        )

        plt.axvline(
            mean_exp,
            linestyle="dashed"
        )

        if params.get("random_exponents", False):
            title_exp = params["exp_range"]
        else:
            title_exp = params["exponent"]

        plt.title(
            f"2D Individual-PSD Exponent Distribution\n"
            f"(Exponent={title_exp}, "
            f"grid={grid_shape})"
        )

        plt.xlabel(
            "Mean electrode exponent (per simulation)"
        )

        plt.ylabel("Count")

        plt.show()

    return (
        all_mean_exponents,
        float(mean_exp),
        float(std_exp)
    )

# ===========================================
# ===========================================

def sweep_electrodes_specmod_2D(
    grid_range,
    n_repeats,
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep electrode grid sizes (e.g. 2x2, 3x3) and compute exponent
    estimation error using individually-fit spectral models per electrode.

    For each simulation:
    - PSDs are computed separately for each electrode
    - A spectral model is fit to each PSD individually
    - Exponents are averaged across electrodes
    - The distribution of these averages is analyzed across repeats

    Parameters
    ----------
    grid_range : iterable
        e.g. [(2,2), (3,3), (4,4)]

    n_repeats : int
        Number of simulations per grid size.

    base_params : dict or None
        Base simulation parameters.

    plot : bool, optional, default: True
        Whether to plot estimation error / mean exponent.

    return_full : bool, optional, default: False
        If True, also return full exponent distributions.

    **param_overrides
        Any simulation parameter overrides
        (exponent, s_rate, n_seconds, etc.)

    Returns
    -------
    grid_sizes : np.ndarray
        Total electrode counts for each grid.

    mean_exps : np.ndarray
        Mean estimated exponent per grid size.

    std_exps : np.ndarray
        Standard deviation of estimated exponents.

    errors : np.ndarray
        Bias values (estimated - true exponent).

    all_distributions : list, optional
        Returned only if return_full=True.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.spec_mod import avg_specmod_exponent_distribution_2D
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    # ---------------------------------
    # Sweep across grid sizes
    # ---------------------------------
    for grid_shape in grid_range:

        all_exps, mean_exp, std_exp = (
            avg_specmod_exponent_distribution_2D(
                n_repeats=n_repeats,
                grid_shape=grid_shape,
                base_params=base_params,
                plot=False,
                **param_overrides
            )
        )

        mean_exps.append(mean_exp)
        std_exps.append(std_exp)

        if true_exp is not None:
            errors.append(mean_exp - - true_exp)
        else:
            errors.append(np.nan)

        if return_full:
            all_distributions.append(all_exps)

        grid_sizes = np.array([r * c for r, c in grid_range])

#Plot
    if plot:

        fig = plot_sweep_results(
            x_values=grid_sizes,
            mean_exps=mean_exps,
            errors=errors,
            true_exp=true_exp,
            std_exps=std_exps,
            xlabel="Number of electrodes",
            title_prefix="2D spectral model exponent estimation"
        )

# Returns
    grid_sizes = np.array(grid_sizes)

    if return_full:
        return (
            grid_sizes,
            np.array(mean_exps),
            np.array(std_exps),
            np.array(errors),
            all_distributions,
            fig
        )

    return (
        grid_sizes,
        np.array(mean_exps),
        np.array(std_exps),
        np.array(errors),
        fig
    )


