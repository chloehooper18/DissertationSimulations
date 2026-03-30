def avg_specmod_exp(psds, freq_range):
    """
    Fit spectral models to PSDs and compute aperiodic exponents.
    Works when electrodes are generated as a 1D array

    Parameters
    ----------
    psds : dict
        {name: (freqs, powers)}
    freq_range : list or tuple
        Frequency range to fit (e.g., [1, 50])

    Returns
    -------
    spectral_models : dict
        {name: SpectralModel object}
    exponents : dict
        {name: exponent value}
    avg_exponent : float
        Mean exponent across signals
    """
    import numpy as np
    from specparam import SpectralModel

    spectral_models = {}
    exponents = {}

    for name, (freqs, powers) in psds.items():

        # Fit model
        fm = SpectralModel()
        fm.report(freqs, powers, freq_range)

        spectral_models[name] = fm

        # Extract exponent
        exponent = fm.get_params('aperiodic', 'exponent')
        exponents[name] = exponent

        print(f"{name}: {exponent:.3f}")

    # Average exponent
    avg_exponent = np.mean(list(exponents.values()))

    print(f"\nAverage exponent: {avg_exponent:.3f}")

    return spectral_models, exponents, avg_exponent