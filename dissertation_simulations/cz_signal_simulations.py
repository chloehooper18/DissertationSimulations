# simulate_cz_signal_ap.py

from neurodsp.sim import sim_powerlaw, sim_oscillation, sim_combined

"""Simulating Cz signal that contains aperiodic noise only"""

def generate_cz_signal_ap():

    """
    Generate a Cz aperiodic signal using parameters
    from setup_Cz_simulation_ap().

    Returns
    -------
    cz_signal_ap : array-like
        Simulated aperiodic signal.
    params : dict
        Dictionary of parameters used to generate the signal.
    times : array-like
        Time vector corresponding to the signal.
    """
    # Import the setup function
    from dissertation_simulations.params import setup_Cz_simulation_ap

    # Get parameters and time vector
    params, times = setup_Cz_simulation_ap() # Params dict that contains simulation setting for Cz signal and "times" for creating time series

    n_seconds = params["n_seconds"]
    fs = params["s_rate"]
    exponent = params["exponent"]
    high_pass_filter = params["high_pass_filter"]

    # Generate the aperiodic signal using the parameters
    cz_signal_ap = sim_powerlaw (n_seconds, fs, exponent = exponent, f_range = [high_pass_filter, None])
    return cz_signal_ap


# =====================================================================
# =====================================================================


"""Simulating Cz signal that contains aperiodic and periodic noise"""

def generate_cz_signal_full():

    """
    Generate a Cz signal using parameters
    from setup_Cz_simulation_full().

    Returns
    -------
    cz_signal_full : array-like
        Simulated aperiodic signal.
    params : dict
        Dictionary of parameters used to generate the signal.
    times : array-like
        Time vector corresponding to the signal.
    """
    # Import the setup function
    from dissertation_simulations.params import setup_Cz_simulation_full

    # Get parameters and time vector
    params, times = setup_Cz_simulation_full()

    # Generate the signal using the parameters
    components = {
    'sim_powerlaw': {'exponent': params["exponent"], 'f_range': [params["high_pass_filter"], None]},
    'sim_oscillation': {'freq': params["oscillation"]}
    }

    cz_signal_full = sim_combined(params["n_seconds"], params["s_rate"], components=components) 

    return cz_signal_full, params, times