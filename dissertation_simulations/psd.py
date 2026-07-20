def avg_psd(signals, params, plot_individual=False):
    """
    Compute PSDs for multiple signals, plot them, and plot the average PSD.
    Works when electrodes are generated as a 1D array

    Parameters
    ----------
    signals : dict
        {name: signal_array}
    params : dict
        Must contain 's_rate'
    plot_individual : bool
        Whether to plot each electrode PSD

    Returns
    -------
    psds : dict
        {name: (freqs, powers)}
    avg_powers : np.ndarray
        Averaged PSD across signals
    freqs : np.ndarray
        Frequency vector
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from specparam.plts import plot_spectra
    from dissertation_simulations.params import electrode_sim_params_ap
    from dissertation_simulations.params import welch_params
    from neurodsp.spectral import compute_spectrum_welch

    psds = {}

    params = electrode_sim_params_ap
    welch_dict = welch_params(params)

    # -------------------------
    # Step 1: individual PSDs
    # -------------------------
    for name, signal in signals.items():

        freqs, powers = compute_spectrum_welch(
            signal,
            fs=params["s_rate"],
            **welch_dict
        )

        psds[name] = (freqs, powers)

        if plot_individual:
            fig, ax = plt.subplots()

            plot_spectra(freqs, powers, log_powers=True, ax=ax)

            ax.set_title(name)
            ax.set_xlim(0, 55)
            ax.set_ylim(-4, 0)

            plt.show()

    # -------------------------
    # Step 2: average PSD
    # -------------------------
    power_array = np.array([powers for freqs, powers in psds.values()])
    avg_powers = np.mean(power_array, axis=0)

    freqs = list(psds.values())[0][0]

    # -------------------------
    # Step 3: plot average PSD
    # -------------------------
    fig, ax = plt.subplots()

    plot_spectra(freqs, avg_powers, log_powers=True, ax=ax)

    ax.set_title("Average PSD")
    ax.set_xlim(0, 55)
    ax.set_ylim(-4, 0)

    plt.show()

    return psds, avg_powers, freqs

# ===========================================
# ===========================================

def avg_psd_2D(grid_signals, params, plot_individual=False):
    """
    Compute PSDs for a 2D electrode grid, plot them, and plot the average PSD.

    Parameters
    ----------
    grid_signals : np.ndarray
        3D array with shape (n_rows, n_cols, n_timepoints)
    params : dict
        Must contain 's_rate'
    plot_individual : bool
        Whether to plot each electrode PSD

    Returns
    -------
    psds : dict
        {(i,j): (freqs, powers)}
    avg_powers : np.ndarray
        Averaged PSD across all electrodes
    freqs : np.ndarray
        Frequency vector
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from specparam.plts import plot_spectra
    from dissertation_simulations.params import welch_params
    from neurodsp.spectral import compute_spectrum_welch

    psds = {}

    welch_dict = welch_params(params)

    n_rows, n_cols, _ = grid_signals.shape

    # -------------------------
    # Step 1: individual PSDs
    # -------------------------
    for i in range(n_rows):
        for j in range(n_cols):

            signal = grid_signals[i, j]

            freqs, powers = compute_spectrum_welch(
                signal,
                fs=params["s_rate"],
                **welch_dict
            )

            psds[(i, j)] = (freqs, powers)

            if plot_individual:
                fig, ax = plt.subplots()

                plot_spectra(freqs, powers, log_powers=True, ax=ax)

                ax.set_title(f"Electrode ({i},{j})")
                ax.set_xlim(0, 55)
                ax.set_ylim(-4, 0)

                plt.show()

    # -------------------------
    # Step 2: average PSD
    # -------------------------
    power_array = np.array([powers for freqs, powers in psds.values()])
    avg_powers = np.mean(power_array, axis=0)

    freqs = next(iter(psds.values()))[0]

    # -------------------------
    # Step 3: plot average PSD
    # -------------------------
    fig, ax = plt.subplots()

    plot_spectra(freqs, avg_powers, log_powers=True, ax=ax)

    ax.set_title("Average PSD")
    ax.set_xlim(0, 55)
    ax.set_ylim(-4, 0)

    plt.show()

    return psds, avg_powers, freqs

# ===========================================
# ===========================================

def avg_psd_distribution(
    n_repeats,
    n_electrodes,
    plot=True,
    base_params=None,
    **param_overrides
):
    """
    Simulate electrode signals, compute and average power spectral densities (PSDs),
    fit a spectral model to the averaged PSD, and return a distribution of estimated
    aperiodic exponents across repeated simulations.

    This function mirrors `avg_time_series_distribution`, but performs averaging in
    the frequency domain (PSD space) rather than the time domain.

    Parameters
    ----------
    n_repeats : int
        Number of simulation iterations to run.

    n_electrodes : int
        Number of electrodes (signals) to simulate and average per iteration.

    plot : bool, optional, default: True
        If True, plot a histogram of the estimated exponent distribution.

    base_params : dict, optional
        Dictionary of default simulation parameters. If None, uses
        `electrode_sim_params_ap`.

    **param_overrides
        Keyword arguments used to override values in `base_params`.
        Example: exponent=-1, n_seconds=10, s_rate=500

        Returns
    -------
    all_exponents : np.ndarray
        Array of estimated aperiodic exponents from each simulation iteration.

    mean_exp : float
        Mean of the estimated exponents across simulations.

    std_exp : float
        Standard deviation of the estimated exponents.

    Notes
    -----
    - Signals are generated independently for each electrode and simulation.
    - Power spectral densities are computed using Welch’s method.
    - PSDs are averaged across electrodes before spectral parameterization.
    - A spectral model is fit to the averaged PSD to estimate the aperiodic exponent.
    - This approach differs from time-domain averaging by performing averaging
      after transformation into the frequency domain.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import (electrode_sim_params_ap, spectral_model_params, freq_range, welch_params)
    from dissertation_simulations.generate_electrodes import generate_1D_electrodes_ap

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    # -------------------------
    # Setup parameters
    # -------------------------
    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if "exponent" not in params:
        raise ValueError("You must specify 'exponent' in params or overrides.")

    all_exponents = []

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

        # -------------------------
        # Compute PSDs per electrode
        # -------------------------
        welch_dict = welch_params(params)

        psd_list = []
        freqs = None

        for signal in signals.values():

            freqs, powers = compute_spectrum_welch(
                signal,
                fs=params["s_rate"],
                **welch_dict
            )

            psd_list.append(powers)

        # -------------------------
        # Average PSD across electrodes
        # -------------------------
        avg_powers = np.mean(psd_list, axis=0)

        # -------------------------
        # Fit spectral model
        # -------------------------
        fm = SpectralModel(**spectral_model_params)
        fm.fit(freqs, avg_powers, freq_range)

        exp = fm.get_params('aperiodic', 'exponent')
        all_exponents.append(exp)

    # -------------------------
    # Summary stats
    # -------------------------
    all_exponents = np.array(all_exponents)
    mean_exp = np.mean(all_exponents)
    std_exp = np.std(all_exponents)

    # -------------------------
    # Plot distribution
    # -------------------------
    if plot:
        plt.hist(all_exponents, bins=20, alpha=0.7)
        plt.axvline(mean_exp, linestyle='dashed')
        plt.title(
            f"PSD-averaged exponent distribution\n"
            f"(Exponent={params['exponent']}, electrodes={n_electrodes})"
        )
        plt.xlabel("Estimated exponent")
        plt.ylabel("Count")
        plt.show()

    return all_exponents, float(mean_exp), float(std_exp)

# ===========================================
# ===========================================

def sweep_electrodes_psd_1D(
    electrode_range,
    n_repeats,
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep number of electrodes and compute exponent estimation
    using PSD averaging.

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
        ANY simulation parameter override.

    Returns
    -------
    electrode_counts, mean_exps, std_exps, errors
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from dissertation_simulations.psd import avg_psd_distribution
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    # -------------------------
    # Sweep electrode counts
    # -------------------------
    for n_elec in electrode_range:

        all_exps, mean_exp, std_exp = avg_psd_distribution(
            n_repeats=n_repeats,
            n_electrodes=n_elec,
            base_params=base_params,
            plot=False,
            **param_overrides
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
            title_prefix="1D PSD exponent estimation"
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

def avg_psd_distribution_2D(
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
            "Parameter 'exponent' must be specified."
        )

    all_exponents = []

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

        psd_list = []

        for i in range(n_rows):
            for j in range(n_cols):

                signal = grid_signals[i, j]

                freqs, powers = compute_spectrum_welch(
                    signal,
                    fs=params["s_rate"],
                    **welch_dict
                )

                psd_list.append(powers)

        psd_array = np.array(psd_list)

        psd_average_space = params.get("psd_average_space", "linear")

        if psd_average_space == "linear":

            avg_powers = np.mean(psd_array, axis=0)

        elif psd_average_space == "log":

            avg_log_powers = np.mean(np.log10(psd_array), axis=0)
            avg_powers = 10 ** avg_log_powers

        else:

            raise ValueError(
                "psd_average_space must be either 'linear' or 'log'."
            )

        fm = SpectralModel(**spectral_model_params)

        fm.fit(
            freqs,
            avg_powers,
            fit_freq_range
        )

        exp = fm.get_params(
            "aperiodic",
            "exponent"
        )

        all_exponents.append(exp)

    all_exponents = np.array(all_exponents)

    mean_exp = np.mean(all_exponents)
    std_exp = np.std(all_exponents)

    if plot:

        plt.hist(all_exponents, bins=20)

        plt.axvline(
            mean_exp,
            linestyle="dashed"
        )

        if params.get("random_exponents", False):
            title_exp = params["exp_range"]
        else:
            title_exp = params["exponent"]

        plt.title(
            f"2D PSD-Averaged Exponent Distribution\n"
            f"(Exponent={title_exp}, "
            f"grid={grid_shape})"
        )

        plt.xlabel("Estimated exponent")
        plt.ylabel("Count")

        plt.show()

    return (
        all_exponents,
        float(mean_exp),
        float(std_exp)
    )

# ===========================================
# ===========================================

def sweep_electrodes_psd_2D(
    grid_range,
    n_repeats,
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep electrode grid sizes (e.g. 2x2, 3x3) and compute exponent
    estimation using PSD averaging.

    Parameters
    ----------
    grid_range : iterable
        e.g. [(2,2), (3,3), (4,4)]
        or [(r, r) for r in range(2, 6)]

    n_repeats : int
        Simulations per grid size.

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
    grid_sizes, mean_exps, std_exps, errors

    optionally:
        full_exponent_distributions
    """

    import matplotlib.pyplot as plt
    import numpy as np

    from dissertation_simulations.psd import avg_psd_distribution_2D
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    # -----------------------------------
    # Sweep grid sizes
    # -----------------------------------
    for grid_shape in grid_range:

        all_exps, mean_exp, std_exp = (
            avg_psd_distribution_2D(
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
            title_prefix="2D PSD exponent estimation"
        )

    print(param_overrides)

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