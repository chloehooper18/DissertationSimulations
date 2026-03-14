""" Parameters for setting up an aperiodic Cz signal"""
def setup_Cz_simulation_ap():
    """
    Initialise simulation parameters for Cz electrode signal with aperiodic noise only.

    This function stores key simulation parameters in a dictionary,
    sets the random seed for reproducibility, and creates the
    simulation time vector based on the specified duration and
    sampling rate.

    Returns
    -------
    params : dict
        Dictionary containing simulation parameters:
        - 'seed' : int
            Random seed used for reproducibility.
        - 'n_seconds' : int
            Duration of the simulation in seconds.
        - 's_rate' : int
            Sampling rate in Hz (samples per second).
        - 'exponent' : float
            Exponent value used in the power-law simulation.
        - 'high_pass_filter' : float
            High-pass filter cutoff frequency in Hz.    
    times : array-like
        Time vector generated from the simulation duration and
        sampling rate.
    """
    # Imports
    import numpy as np

    from neurodsp.utils import create_times
    from neurodsp.sim import set_random_seed

    # Parameter store
    params = {
        "seed": 36,        # Seed number
        "n_seconds": 25,   # Number of seconds
        "s_rate": 1000,     # Sampling rate
        "exponent": -1.5,   # Exponent value 
        "high_pass_filter": 1 # High pass filter
    }

    # Set seed for consistency
    set_random_seed(params["seed"])

    # Simulation settings
    times = create_times(params["n_seconds"], params["s_rate"])

    return params, times

# =====================================================================
# =====================================================================

""" Parameters for setting up an Cz signal that contains both aperiodic and periodic nosie"""
def setup_Cz_simulation_full():
    """
    Initialise simulation parameters for Cz electrode signal with both aperiodic and periodic noise.

    This function stores key simulation parameters in a dictionary,
    sets the random seed for reproducibility, and creates the
    simulation time vector based on the specified duration and
    sampling rate.

    Returns
    -------
    params : dict
        Dictionary containing simulation parameters:
        - 'seed' : int
            Random seed used for reproducibility.
        - 'n_seconds' : int
            Duration of the simulation in seconds.
        - 's_rate' : int
            Sampling rate in Hz (samples per second).
        - 'exponent' : float
            Exponent value used in the power-law simulation.
        - 'high_pass_filter' : float
            High-pass filter cutoff frequency in Hz.  
        - 'oscillation' : float
            Frequency of the periodic oscillation in Hz.  
    times : array-like
        Time vector generated from the simulation duration and
        sampling rate.
    """
    # Imports
    import numpy as np

    from neurodsp.utils import create_times
    from neurodsp.sim import set_random_seed

    # Parameter store
    params = {
        "seed": 36,        # Seed number
        "n_seconds": 25,   # Number of seconds
        "s_rate": 1000,     # Sampling rate
        "exponent": -1.5,   # Exponent value 
        "high_pass_filter": 1, # High pass filter
        "oscillation": 10   # Frequency of periodic oscillation
    }

    # Set seed for consistency
    set_random_seed(params["seed"])

    # Simulation settings
    times = create_times(params["n_seconds"], params["s_rate"])

    return params, times


# =====================================================================
# =====================================================================

# Parameters for Welch spectral estimation

params, times = setup_Cz_simulation_ap()
welch_params = {
    "nperseg": params["s_rate"],        # length of each segment in samples
    "noverlap": params["s_rate"] // 2   # number of samples overlapping between segments
}

# =====================================================================
# =====================================================================

# Parameters for Spectral Model fitting 

spectral_model_params = {
    "min_peak_height": 0.05,  # minimum height for peak detection
    "verbose": False           # do not print info while fitting
}

# =====================================================================
# =====================================================================

# Paramters for freq_range when fitting SpectralModel()

freq_range = [2, 40]  # frequency range of interest in Hz