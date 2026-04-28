def generate_1D_electrodes_ap(n_electrodes, exponents, base_params):
    """
    Generate multiple aperiodic signals with different exponents.

    Parameters
    ----------
    n_electrodes : int
        Number of signals to generate.
    exponents : list of float
        List of exponents (one per electrode).
    base_params : dict
        Dictionary of simulation parameters.

    Returns
    -------
    signals : dict
        Dictionary of signals {electrode_1: signal, ...}
    params_list : list of dict
        Parameters used for each signal.
    times : np.ndarray
        Shared time vector.
    """

    from neurodsp.sim import sim_powerlaw
    from dissertation_simulations.params import electrode_times

    # Safety check
    if len(exponents) != n_electrodes:
        raise ValueError("Number of exponents must match number of electrodes")

    # Extract params
    n_seconds = base_params["n_seconds"]
    fs = base_params["s_rate"]
    high_pass_filter = base_params["high_pass_filter"]
    low_pass_filter = base_params["low_pass_filter"]

    times = electrode_times(base_params)

    signals = {}
    params_list = []

    for i in range(n_electrodes):

        # Copy params safely
        params = base_params.copy()
        params["exponent"] = exponents[i]

        signal = sim_powerlaw(
            n_seconds,
            fs,
            exponent=params["exponent"],
            f_range=[high_pass_filter, low_pass_filter]
        )

        signals[f"electrode_{i+1}"] = signal
        params_list.append(params)

    return signals, params_list, times

# ========================================
# ========================================

def generate_1D_electrodes_full(n_electrodes, exponents):
    """
    Generate multiple signals with different exponents.

    Parameters
    ----------
    n_electrodes : int
        Number of signals to generate.
    exponents : list of float
        List of exponents (one per electrode).

    Returns
    -------
    signals : dict
        Dictionary of signals {electrode_1: signal, ...}
    params_list : list of dict
        Parameters used for each signal.
    times : np.ndarray
        Shared time vector.
    """

    from dissertation_simulations.params import electrode_sim_params_full, electrode_times
    from neurodsp.sim import sim_powerlaw, sim_combined, sim_oscillation

    # Safety check
    if len(exponents) != n_electrodes:
        raise ValueError("Number of exponents must match number of electrodes")

    # Base params
    base_params = electrode_sim_params_full

    n_seconds = base_params["n_seconds"]
    fs = base_params["s_rate"]
    high_pass_filter = base_params["high_pass_filter"]
    low_pass_filter = base_params["low_pass_filter"]
    freq = base_params["oscillation"]

    times = electrode_times(base_params)

    signals = {}
    params_list = []

    for i in range(n_electrodes):

        # Copy params for this electrode
        params = base_params
        params["exponent"] = exponents[i]

        # Generate signal
        components = {
        'sim_powerlaw': {'exponent': params["exponent"], 'f_range': [high_pass_filter, low_pass_filter]},
        'sim_oscillation': {'freq': freq}
        }

        signal = sim_combined(n_seconds, fs, components=components) 

        # Store
        signals[f"electrode_{i+1}"] = signal
        params_list.append(params)

    return signals, params_list, times

# ============================================
# ============================================

def generate_2D_electrodes_ap(n_rows, n_cols, exponents):
    """
    Generate a 2D grid of simulated aperiodic (1/f) EEG signals.

    This function creates a grid of electrodes arranged in a 2D layout
    (n_rows × n_cols), where each electrode is assigned an aperiodic
    signal generated using a power-law model. Each electrode can have
    a different exponent, allowing spatial variation in spectral properties.

    Parameters
    ----------
    n_rows : int
        Number of rows in the electrode grid.
    n_cols : int
        Number of columns in the electrode grid.
    exponents : array-like
        List or array of exponent values (length must equal n_rows * n_cols).
        Each exponent defines the 1/f slope of the corresponding electrode signal.

    Returns
    -------
    grid : np.ndarray
        3D array of simulated signals with shape (n_rows, n_cols, n_timepoints).
        Each entry grid[i, j, :] contains the time series for one electrode.
    times : np.ndarray
        Time vector corresponding to the simulated signals.

    Notes
    -----
    - Signals are generated using the `sim_powerlaw` function from NeuroDSP.
    - Simulation parameters (e.g., duration, sampling rate, frequency range)
      are defined in `cz_sim_params_ap`.
    - The time vector is created using `cz_times` with the same parameters.
    - The order of exponents is assigned row-wise across the grid.
    """
    import numpy as np
    from dissertation_simulations.params import electrode_sim_params_ap, electrode_times
    from neurodsp.sim import sim_powerlaw

    params = electrode_sim_params_ap
    times = electrode_times(params)

    grid = np.zeros((n_rows, n_cols, len(times)))

    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):

            exp = exponents[idx]

            signal = sim_powerlaw(
                params["n_seconds"],
                params["s_rate"],
                exponent=exp,
                f_range=[params["high_pass_filter"], params["low_pass_filter"]]
            )

            grid[i, j, :] = signal
            idx += 1

    return grid, times