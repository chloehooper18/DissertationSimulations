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

    