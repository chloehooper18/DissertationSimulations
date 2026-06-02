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

    import numpy as np
    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        electrode_times
    )
    from neurodsp.sim import sim_powerlaw

    params = electrode_sim_params_ap
    times = electrode_times(params)

    grid = np.zeros((n_rows, n_cols, len(times)))

    # -------------------------
    # Handle exponent input
    # -------------------------
    exponents = np.array(exponents)

    if exponents.ndim == 1:

        if len(exponents) != n_rows * n_cols:
            raise ValueError(
                "1D exponent array must have length n_rows * n_cols"
            )

        exponents = exponents.reshape(n_rows, n_cols)

    elif exponents.ndim == 2:

        if exponents.shape != (n_rows, n_cols):
            raise ValueError(
                "2D exponent matrix shape mismatch"
            )

    else:
        raise ValueError(
            "Exponents must be 1D or 2D"
        )

    # -------------------------
    # Generate signals
    # -------------------------
    for i in range(n_rows):
        for j in range(n_cols):

            exp = exponents[i, j]

            signal = sim_powerlaw(
                params["n_seconds"],
                params["s_rate"],
                exponent=exp,
                f_range=[
                    params["high_pass_filter"],
                    params["low_pass_filter"]
                ]
            )

            grid[i, j, :] = signal

    return grid, times