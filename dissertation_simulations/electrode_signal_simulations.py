# simulate_electrode_signal_ap.py


"""Simulating electrode signal that contains aperiodic noise only"""

def generate_electrode_signal_ap():

    """
    Generate a simulated aperiodic EEG signal for a single electrode.

    This function uses predefined simulation parameters to create an aperiodic
    (1/f) signal for a single electrode. It generates a time vector, and produces 
    the signal using a power-law simulation.

    Returns
    -------
    electrode_signal_ap : np.ndarray
        The simulated aperiodic electrode signal (voltage values in μV).
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
    - The time vector is created using `electrode_times` with the same simulation parameters.
    """
    
    # Import the setup function
    from dissertation_simulations.params import electrode_sim_params_ap
    from dissertation_simulations.params import electrode_seed
    from dissertation_simulations.params import electrode_times
    from neurodsp.sim import sim_powerlaw

    # Get parameters and time vector
    params = electrode_sim_params_ap # Params dict that contains simulation setting for aperiodic electrode signal

    n_seconds = params["n_seconds"]
    fs = params["s_rate"]
    exponent = params["exponent"]
    high_pass_filter = params["high_pass_filter"]
    low_pass_filter = params["low_pass_filter"]

    times = electrode_times(params) # Created time vector for Cz signal


    # Generate the aperiodic signal using the parameters
    electrode_signal_ap = sim_powerlaw (n_seconds, fs, exponent = exponent, f_range = [high_pass_filter, low_pass_filter])
  

    return electrode_signal_ap, params, times


# =====================================================================
# =====================================================================


"""Simulating electrode signal that contains aperiodic and periodic noise"""

def generate_electrode_signal_full():

    """
    Generate a simulated EEG signal that contains aperiodic and periodic noise for a single electrode.

    This function uses predefined simulation parameters to create a signal for a single 
    electrode. It generates a time vector, and produces the signal using a power-law simulation.

    Returns
    -------
    electrode_signal_full : np.ndarray
        The simulated aperiodic electrode signal (voltage values in μV).
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
    - The time vector is created using `electrode_times` with the same simulation parameters.
    """

    # Import the setup function
    from dissertation_simulations.params import electrode_sim_params_full
    from dissertation_simulations.params import electrode_seed
    from dissertation_simulations.params import electrode_times
    from neurodsp.sim import sim_powerlaw, sim_oscillation, sim_combined

    # Get parameters and time vector
    params = electrode_sim_params_full # Params dict that contains simulation setting for aperiodic electrode signal

    n_seconds = params["n_seconds"]
    fs = params["s_rate"]
    exponent = params["exponent"]
    high_pass_filter = params["high_pass_filter"]
    low_pass_filter = params["low_pass_filter"]
    freq = params ["oscillation"]

    electrode_seed(params) # Sets random seed for consistency
    times = electrode_times(params) # Created time vector for electrode signal


    # Generate the aperiodic signal using the parameters
    components = {
    'sim_powerlaw': {'exponent': params["exponent"], 'f_range': [params["high_pass_filter"], params["low_pass_filter"]]},
    'sim_oscillation': {'freq': params["oscillation"]}
    }

    electrode_signal_full = sim_combined(params["n_seconds"], params["s_rate"], components=components) 
  

    return electrode_signal_full, params, times