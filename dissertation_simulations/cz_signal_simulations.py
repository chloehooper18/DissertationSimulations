# simulate_cz_signal_ap.py

#, sim_oscillation, sim_combined

"""Simulating Cz signal that contains aperiodic noise only"""

def generate_cz_signal_ap():

    """
    Generate a simulated aperiodic Cz EEG signal.

    This function uses predefined simulation parameters to create an aperiodic
    (1/f) signal for the Cz electrode. It generates a time vector, and produces 
    the signal using a power-law simulation.

    Returns
    -------
    cz_signal_ap : np.ndarray
        The simulated aperiodic Cz signal (voltage values in μV).
    params : dict
        Dictionary containing all simulation parameters used to generate the signal.
        Keys include:
            - 'n_seconds' : Duration of the signal in seconds
            - 's_rate' : Sampling rate in Hz
            - 'exponent' : Power-law exponent for the aperiodic signal
            - 'high_pass_filter' : Minimum frequency for the simulated signal
            - ...other simulation parameters
    times : np.ndarray
        Time vector corresponding to the signal (in seconds).

    Notes
    -----
    - The signal is generated using the `sim_powerlaw` function from NeuroDSP.
    - The time vector is created using `cz_times` with the same simulation parameters.
    """
    
    # Import the setup function
    from dissertation_simulations.params import cz_sim_params_ap
    from dissertation_simulations.params import cz_seed
    from dissertation_simulations.params import cz_times
    from neurodsp.sim import sim_powerlaw

    # Get parameters and time vector
    params = cz_sim_params_ap # Params dict that contains simulation setting for aperiodic Cz signal

    n_seconds = params["n_seconds"]
    fs = params["s_rate"]
    exponent = params["exponent"]
    high_pass_filter = params["high_pass_filter"]
    low_pass_filter = params["low_pass_filter"]

    times = cz_times(params) # Created time vector for Cz signal


    # Generate the aperiodic signal using the parameters
    cz_signal_ap = sim_powerlaw (n_seconds, fs, exponent = exponent, f_range = [high_pass_filter, low_pass_filter])
  

    return cz_signal_ap, params, times


# =====================================================================
# =====================================================================


"""Simulating Cz signal that contains aperiodic and periodic noise"""

def generate_cz_signal_full():

    """
    Generate a simulated Cz EEG signal that contains aperiodic and periodic noise.

    This function uses predefined simulation parameters to create a signal for the Cz 
    electrode. It generates a time vector, and produces the signal using a power-law simulation.

    Returns
    -------
    cz_signal_full : np.ndarray
        The simulated aperiodic Cz signal (voltage values in μV).
    params : dict
        Dictionary containing all simulation parameters used to generate the signal.
        Keys include:
            - 'n_seconds' : Duration of the signal in seconds
            - 's_rate' : Sampling rate in Hz
            - 'exponent' : Power-law exponent for the aperiodic signal
            - 'high_pass_filter' : Minimum frequency for the simulated signal
            - ...other simulation parameters
    times : np.ndarray
        Time vector corresponding to the signal (in seconds).

    Notes
    -----
    - The signal is generated using the `sim_powerlaw` function from NeuroDSP.
    - The time vector is created using `cz_times` with the same simulation parameters.
    """

    # Import the setup function
    from dissertation_simulations.params import cz_sim_params_full
    from dissertation_simulations.params import cz_seed
    from dissertation_simulations.params import cz_times
    from neurodsp.sim import sim_powerlaw, sim_oscillation, sim_combined

    # Get parameters and time vector
    params = cz_sim_params_full # Params dict that contains simulation setting for aperiodic Cz signal

    n_seconds = params["n_seconds"]
    fs = params["s_rate"]
    exponent = params["exponent"]
    high_pass_filter = params["high_pass_filter"]
    low_pass_filter = params["low_pass_filter"]
    freq = params ["oscillation"]

    cz_seed(params) # Sets random seed for consistency
    times = cz_times(params) # Created time vector for Cz signal


    # Generate the aperiodic signal using the parameters
    components = {
    'sim_powerlaw': {'exponent': params["exponent"], 'f_range': [params["high_pass_filter"], params["low_pass_filter"]]},
    'sim_oscillation': {'freq': params["oscillation"]}
    }

    cz_signal_full = sim_combined(params["n_seconds"], params["s_rate"], components=components) 
  

    return cz_signal_full, params, times