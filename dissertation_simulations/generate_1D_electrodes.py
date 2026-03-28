def generate_1D_electrodes_ap(n_electrodes, exponents):
    """
    Generate multiple aperiodic signals with different exponents.

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

    from dissertation_simulations.params import cz_sim_params_ap, cz_times
    from neurodsp.sim import sim_powerlaw

    # Safety check
    if len(exponents) != n_electrodes:
        raise ValueError("Number of exponents must match number of electrodes")

    # Base params
    base_params = cz_sim_params_ap.copy()

    n_seconds = base_params["n_seconds"]
    fs = base_params["s_rate"]
    high_pass_filter = base_params["high_pass_filter"]
    low_pass_filter = base_params["low_pass_filter"]

    times = cz_times(base_params)

    signals = {}
    params_list = []

    for i in range(n_electrodes):

        # Copy params for this electrode
        params = base_params.copy()
        params["exponent"] = exponents[i]

        # Generate signal
        signal = sim_powerlaw(
            n_seconds,
            fs,
            exponent=params["exponent"],
            f_range=[high_pass_filter, low_pass_filter]
        )

        # Store
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

    from dissertation_simulations.params import cz_sim_params_full, cz_times
    from neurodsp.sim import sim_powerlaw, sim_combined, sim_oscillation

    # Safety check
    if len(exponents) != n_electrodes:
        raise ValueError("Number of exponents must match number of electrodes")

    # Base params
    base_params = cz_sim_params_full.copy()

    n_seconds = base_params["n_seconds"]
    fs = base_params["s_rate"]
    high_pass_filter = base_params["high_pass_filter"]
    low_pass_filter = base_params["low_pass_filter"]
    freq = base_params["oscillation"]

    times = cz_times(base_params)

    signals = {}
    params_list = []

    for i in range(n_electrodes):

        # Copy params for this electrode
        params = base_params.copy()
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