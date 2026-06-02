def plot_multiple_time_series(signals, times):
    """
    Plot multiple time series as separate subplots.
    Works when electrodes are generated as a 1D array

    Parameters
    ----------
    signals : dict
        Dictionary of signals {name: signal_array}
    times : array-like
        Time vector
    """
    import matplotlib.pyplot as plt
    from neurodsp.plts import plot_time_series

    fig, axes = plt.subplots(len(signals), 1, sharex=True)

    # Handle case where there's only one signal
    if len(signals) == 1:
        axes = [axes]

    for ax, (name, signal) in zip(axes, signals.items()):
        plot_time_series(times, signal, ax=ax)
        ax.set_title(name)

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# ===========================================
# ===========================================


def avg_ts(signals, times):
    """
    Compute and plot the average time series across multiple signals.
    Works when electrodes are generated as a 1D array

    Parameters
    ----------
    signals : dict
        Dictionary of signals {name: signal_array}
    times : array-like
        Time vector

    """

    import numpy as np
    from neurodsp.plts import plot_time_series

    # Convert to array (n_electrodes × n_timepoints)
    signal_array = np.array(list(signals.values()))

    # Average across electrodes
    avg_signal = np.mean(signal_array, axis=0)

    # Plot
    plot_time_series(times, avg_signal, title="Average Signal")

# ===========================================
# ===========================================


def plot_multiple_time_series_2D(grid_signals, times):
    """
    Plot each signal from a 2D grid of electrodes as separate time series.

    Parameters
    ----------
    grid_signals : np.ndarray
        3D array of signals with shape (n_rows, n_cols, n_timepoints).
    times : array-like
        Time vector corresponding to the signals.

    Returns
    -------
    None
        Displays one plot per electrode.
    """
    import matplotlib.pyplot as plt
    from neurodsp.plts import plot_time_series

    n_rows, n_cols, _ = grid_signals.shape

    for i in range(n_rows):
        for j in range(n_cols):

            fig, ax = plt.subplots()

            signal = grid_signals[i, j]

            plot_time_series(times, signal, ax=ax)
            ax.set_title(f"Electrode ({i},{j})")

            plt.show()

# ===========================================
# ===========================================

def avg_ts_2D(signals, times):
    """
    Compute and plot the average time series across a 2D electrode grid.

    Parameters
    ----------
    grid_signals : np.ndarray
        3D array of signals with shape (n_rows, n_cols, n_timepoints).
    times : array-like
        Time vector corresponding to the signals.

    Returns
    -------
    avg_signal : np.ndarray
        The averaged time series across all electrodes.
    """
    import numpy as np
    from neurodsp.plts import plot_time_series

    # Average across spatial dimensions (rows and columns)
    avg_signal = np.mean(grid_signals, axis=(0, 1))

    # Plot
    plot_time_series(times, avg_signal, title="Average Signal")

    return avg_signal

# ===========================================
# ===========================================

def avg_time_series_distribution(
    n_repeats,
    n_electrodes,
    plot=True,
    base_params=None,
    **param_overrides
):
    """
    Simulate electrode signals, compute and average the time series, compute the PSD,
    fit a spectral model to the PSD, and return a distribution of estimated
    aperiodic exponents across repeated simulations.

    Parameters
    ----------
    n_repeats : int
        Number of simulation iterations to run.

    n_electrodes : int, optional, default: 3
        Number of electrodes (signals) to simulate and average.

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
        Array of estimated exponents from each simulation.

    mean_exp : float
        Mean of the estimated exponents across simulations.

    std_exp : float
        Standard deviation of the estimated exponents.

    Notes
    -----
    - All simulation parameters (including exponent) are controlled via `base_params`
      and `param_overrides`.
    - Signals are averaged in the time domain before spectral estimation.
    - Power spectra are computed using Welch’s method.
    - A spectral model is fit to estimate the aperiodic exponent.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import electrode_sim_params_ap, spectral_model_params, freq_range, welch_params
    from dissertation_simulations.generate_electrodes import generate_1D_electrodes_ap

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    from dissertation_simulations.exponents import generate_random_electrode_exponents

    # --- Set base params ---
    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    # --- Apply overrides ---
    params.update(param_overrides)

    # --- Ensure exponent is specified ---
    if "exponent" not in params:
        raise ValueError("Parameter 'exponent' must be specified in base_params or overrides.")

    all_exponents = []

    for i in range(n_repeats):

        # Use exponent from params
        if params.get("random_exponents", False):

            exponents = generate_random_electrode_exponents(
                n_electrodes=n_electrodes,
                exp_range=params["exp_range"],
                distribution=params.get("distribution", "uniform"))

        else:

            exponents = np.full(
                n_electrodes,
                params["exponent"])

        signals, params_list, times = generate_1D_electrodes_ap(n_electrodes, exponents, params)

        signal_array = np.array(list(signals.values()))
        avg_signal = np.mean(signal_array, axis=0)

        welch_dict = welch_params(params)

        freqs, powers = compute_spectrum_welch(
            avg_signal,
            fs=params["s_rate"],
            **welch_dict
        )

        fm = SpectralModel(**spectral_model_params)
        fm.fit(freqs, powers, freq_range)

        exp = fm.get_params('aperiodic', 'exponent')
        all_exponents.append(exp)

    all_exponents = np.array(all_exponents)
    mean_exp = np.mean(all_exponents)
    std_exp = np.std(all_exponents)

    if plot:
        plt.hist(all_exponents, bins=20)
        plt.axvline(mean_exp, linestyle='dashed')
        plt.title(
            f"Exponent distribution (Exponent={params['exponent']}, electrodes per ppt={n_electrodes})"
        )
        plt.xlabel("Estimated exponent")
        plt.ylabel("Count")
        plt.show()

    return all_exponents, float(mean_exp), float(std_exp)

# ===========================================
# ===========================================

def sweep_electrodes_ts_1D(electrode_range, n_repeats, base_params=None, plot=True, return_full=False, **param_overrides):
    """
    Sweep number of electrodes and compute exponent estimation error.

    Parameters
    ----------
    electrode_range : iterable
        e.g. range(1, 10)

    n_repeats : int
        simulations per electrode count

    base_params : dict or None
        base simulation parameters

    plot : bool
        whether to plot error curve

    return_full : bool
        if True, also returns full distributions

    **param_overrides :
        ANY simulation parameter override (exponent, s_rate, etc.)

    Returns
    -------
    electrode_counts, mean_exps, std_exps, errors
    (optionally full_exponent_distributions)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from dissertation_simulations.time_series import avg_time_series_distribution
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    for n_elec in electrode_range:

        all_exps, mean_exp, std_exp = avg_time_series_distribution(
            n_repeats=n_repeats,
            n_electrodes=n_elec,
            base_params=base_params,
            plot=False,
            **param_overrides
        )

        mean_exps.append(mean_exp)
        std_exps.append(std_exp)

        if true_exp is not None:
            errors.append(mean_exp - -  true_exp)
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
            title_prefix="1D time-series exponent estimation"
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

def avg_time_series_distribution_2D(
    n_repeats,
    grid_shape,
    plot=True,
    base_params=None,
    **param_overrides
):
    """
    Simulate 2D electrode grid signals, average the time series,
    compute the PSD of the averaged signal, fit a spectral model, and return
    a distribution of estimated aperiodic exponents across repeated simulations.

    Parameters
    ----------
    n_repeats : int
        Number of simulation iterations to run.

    grid_shape : tuple of int
        Shape of the electrode grid as (n_rows, n_cols).

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
        Array of estimated exponents from each simulation.

    mean_exp : float
        Mean of the estimated exponents across simulations.

    std_exp : float
        Standard deviation of the estimated exponents.

    Notes
    -----
    - Signals are averaged across the 2D spatial grid before spectral estimation.
    - Power spectra are computed using Welch’s method.
    - A spectral model is fit to estimate the aperiodic exponent.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import electrode_sim_params_ap, spectral_model_params, freq_range, welch_params

    from dissertation_simulations.generate_electrodes import generate_2D_electrodes_ap

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    from dissertation_simulations.exponents import generate_random_electrode_exponents

    # Setup params
    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if "exponent" not in params:
        raise ValueError("Parameter 'exponent' must be specified.")

    all_exponents = []

    n_rows, n_cols = grid_shape

    # Simulation loop
    for _ in range(n_repeats):

        # Create exponent grid
        n_total = n_rows * n_cols

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
        
        # Generate signals
        grid_signals, times = generate_2D_electrodes_ap(n_rows, n_cols, exponent_grid)

        # Average time series (2D)
        avg_signal = np.mean(grid_signals, axis=(0, 1))

        # PSD
        welch_dict = welch_params(params)

        freqs, powers = compute_spectrum_welch(
            avg_signal,
            fs=params["s_rate"],
            **welch_dict
        )

        # Spectral model
        fm = SpectralModel(**spectral_model_params)
        fm.fit(freqs, powers, freq_range)

        exp = fm.get_params('aperiodic', 'exponent')
        all_exponents.append(exp)

    # Summary stats
    all_exponents = np.array(all_exponents)
    mean_exp = np.mean(all_exponents)
    std_exp = np.std(all_exponents)

    # Plot
    if plot:
        plt.hist(all_exponents, bins=20)
        plt.axvline(mean_exp, linestyle='dashed')
        plt.title(
            f"2D Avg Signal Exponent Distribution\n"
            f"(Exponent={params['exponent']}, grid={grid_shape})"
        )
        plt.xlabel("Estimated exponent")
        plt.ylabel("Count")
        plt.show()

    return all_exponents, float(mean_exp), float(std_exp)

# ===========================================
# ===========================================

def sweep_electrodes_ts_2D(
    grid_range,
    n_repeats,
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep electrode grid sizes (e.g. 2x2, 3x3) and compute exponent estimation error.

    Parameters
    ----------
    grid_range : iterable
        e.g. [(2,2), (3,3), (4,4)] or [(r, r) for r in range(2, 6)]

    n_repeats : int
        simulations per grid size

    base_params : dict or None
        base simulation parameters

    plot : bool
        whether to plot error curve

    return_full : bool
        if True, also returns full distributions

    **param_overrides :
        ANY simulation parameter override (exponent, s_rate, etc.)

    Returns
    -------
    grid_sizes, mean_exps, std_exps, errors
    (optionally full_exponent_distributions)
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from dissertation_simulations.time_series import avg_time_series_distribution_2D
    from dissertation_simulations.plotting import plot_sweep_results

    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig=None

    true_exp = param_overrides.get("exponent", None)

    for grid_shape in grid_range:

        all_exps, mean_exp, std_exp = avg_time_series_distribution_2D(
            n_repeats=n_repeats,
            grid_shape=grid_shape,
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
            title_prefix="2D time-series exponent estimation"
        )

# Returns
    grid_sizes = np.array(grid_sizes)

    if return_full:

        return (
            grid_sizes,
            np.array(mean_exps),
            np.array(std_exps),
            np.array(errors),
            all_distributions
        )

    return (
        grid_sizes,
        np.array(mean_exps),
        np.array(std_exps),
        np.array(errors),
        fig
    )