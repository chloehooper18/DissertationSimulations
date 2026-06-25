""" Parameters for setting up an aperiodic electrode signal"""
electrode_sim_params_ap = {
    "n_seconds": 25,       # Duration of simulation (seconds)
    "s_rate": 1000,        # Sampling rate (Hz)
    #"exponent": -1.5,      # Exponent for power-law signal
    "high_pass_filter": 1,  # High-pass filter cutoff (Hz)
    "low_pass_filter": 50   # Low-pass filter cutoff (Hz)
}

# =====================================================================
# =====================================================================

""" Parameters for setting up an electrode signal that contains both aperiodic and periodic nosie"""
electrode_sim_params_full = {
        "n_seconds": 25,   # Number of seconds
        "s_rate": 1000,     # Sampling rate
        #"exponent": -1.5,   # Exponent value 
        "high_pass_filter": 1, # High pass filter
        "low_pass_filter": 50,   # Low-pass filter cutoff (Hz)   
        "oscillation": 10   # Frequency of periodic oscillation
    }

# =====================================================================
# =====================================================================

def electrode_seed(params):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters.
        Must include the key 'seed'.
    """
    from neurodsp.sim import set_random_seed

    set_random_seed(params["seed"])

# =====================================================================
# =====================================================================

def electrode_times(params):
    """
    Create the time vector for the single electrode simulation.

    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters.
        Must include 'n_seconds' and 's_rate'.

    Returns
    -------
    times : array-like
        Time vector generated from the simulation duration
        and sampling rate.
    """
    from neurodsp.utils import create_times

    times = create_times(params["n_seconds"], params["s_rate"])

    return times

# =====================================================================
# =====================================================================

# Parameters for Welch spectral estimation

def welch_params(params):
    """
    Create Welch spectral estimation parameters.

    Parameters
    ----------
    params : dict
        Simulation parameter dictionary containing 's_rate'.

    Returns
    -------
    dict
        Dictionary of Welch parameters.
    """

    return {
        "nperseg": params["s_rate"],
        "noverlap": params["s_rate"] // 2
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