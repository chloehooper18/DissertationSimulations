def plot_multiple_time_series(signals, times):
    """
    Plot multiple time series as separate subplots.

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