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

# ===========================================
# ===========================================

def avg_specmod_exponent_distribution(
    n_repeats,
    n_electrodes,
    plot=True,
    base_params=None,
    **param_overrides
):
    """
    Simulate electrode signals, compute individual PSDs, fit spectral models
    to each electrode separately, average the resulting aperiodic exponents,
    and return a distribution across repeated simulations.

    This approach differs from other pipelines in that spectral parameterization
    is performed at the single-electrode level before any averaging.

    Parameters
    ----------
    n_repeats : int
        Number of simulation iterations to run.

    n_electrodes : int
        Number of electrodes (signals) per simulation.

    plot : bool, optional, default: True
        If True, plot histogram of averaged exponent distribution.

    base_params : dict, optional
        Dictionary of default simulation parameters. If None, uses
        `electrode_sim_params_ap`.

    **param_overrides
        Keyword arguments used to override values in `base_params`.
        Example: exponent=-1, n_seconds=10, s_rate=500

    Returns
    -------
    all_mean_exponents : np.ndarray
        Mean exponent from each simulation iteration.

    mean_exp : float
        Mean of simulation-level averaged exponents.

    std_exp : float
        Standard deviation of simulation-level averaged exponents.

    Notes
    -----
    - PSDs are computed per electrode using Welch’s method.
    - Each PSD is fit independently with a spectral model.
    - Exponents are averaged across electrodes within each simulation.
    - Final output is a distribution of these per-simulation means.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        spectral_model_params,
        freq_range,
        welch_params
    )
    from dissertation_simulations.generate_electrodes import generate_1D_electrodes_ap

    from neurodsp.spectral import compute_spectrum_welch
    from specparam import SpectralModel

    # -------------------------
    # Setup params
    # -------------------------
    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if "exponent" not in params:
        raise ValueError("You must specify 'exponent' in params or overrides.")

    all_mean_exponents = []

    # -------------------------
    # Simulation loop
    # -------------------------
    for _ in range(n_repeats):

        exponents = [params["exponent"]] * n_electrodes

        signals, _, _ = generate_1D_electrodes_ap(
            n_electrodes,
            exponents,
            params
        )

        welch_dict = welch_params(params)

        electrode_exponents = []

        # -------------------------
        # Fit each electrode individually
        # -------------------------
        for signal in signals.values():

            freqs, powers = compute_spectrum_welch(
                signal,
                fs=params["s_rate"],
                **welch_dict
            )

            fm = SpectralModel(**spectral_model_params)
            fm.fit(freqs, powers, freq_range)

            exp = fm.get_params('aperiodic', 'exponent')
            electrode_exponents.append(exp)

        # -------------------------
        # Average across electrodes
        # -------------------------
        all_mean_exponents.append(np.mean(electrode_exponents))

    # -------------------------
    # Summary stats
    # -------------------------
    all_mean_exponents = np.array(all_mean_exponents)
    mean_exp = np.mean(all_mean_exponents)
    std_exp = np.std(all_mean_exponents)

    # -------------------------
    # Plot distribution
    # -------------------------
    if plot:
        plt.hist(all_mean_exponents, bins=20, alpha=0.7)
        plt.axvline(mean_exp, linestyle='dashed')
        plt.title(
            f"Individual-PSD exponent distribution\n"
            f"(Exponent={params['exponent']}, electrodes={n_electrodes})"
        )
        plt.xlabel("Mean electrode exponent (per simulation)")
        plt.ylabel("Count")
        plt.show()

    return all_mean_exponents, float(mean_exp), float(std_exp)