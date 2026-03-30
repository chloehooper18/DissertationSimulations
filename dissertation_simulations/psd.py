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
    from dissertation_simulations.params import cz_sim_params_ap
    from dissertation_simulations.params import welch_params
    from neurodsp.spectral import compute_spectrum_welch

    psds = {}

    params = cz_sim_params_ap.copy()
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