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
    Simulate and estimate aperiodic exponents from averaged electrode signals,
    allowing flexible overriding of simulation parameters.

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
        exponents = [params["exponent"]] * n_electrodes

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