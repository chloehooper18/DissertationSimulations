def avg_psd_distribution_2D_eeg(
    n_repeats,
    grid_shape,
    data_dir,
    electrode_labels,
    status="ec",
    plot=True,
    base_params=None,
    fit_freq_range=None,
    verbose=False,
    **param_overrides
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        spectral_model_params,
        freq_range
    )

    from dissertation_simulations.eeg_data import get_electrode_power
    from specparam import SpectralModel

    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if fit_freq_range is None:
        fit_freq_range = freq_range

    if verbose:
        print("Using fit_freq_range:", fit_freq_range)

    data_dir = Path(data_dir)

    file_list = os.listdir(data_dir)
    file_list = [cfile for cfile in file_list if cfile[0] != "."]
    file_list = [cfile for cfile in file_list if status in cfile]

    if len(file_list) == 0:
        raise ValueError(f"No files found for status '{status}' in {data_dir}")

    all_exponents = []

    for _ in range(n_repeats):

        for cfile in file_list:

            loaded_data = np.load(data_dir / cfile)

            freqs = loaded_data["arr_0"]
            powers = loaded_data["arr_1"]
            electrodes = loaded_data["arr_2"]

            block_index = params.get("block_index", 0)
            powers = powers[block_index, :, :]

            psd_list = []

            for e_label in electrode_labels:

                electrode_power = get_electrode_power(
                    powers=powers,
                    electrodes=electrodes,
                    e_label=e_label,
                    freqs=freqs,
                    cfile=cfile
                )

                psd_list.append(electrode_power)

            psd_array = np.vstack(psd_list)

            psd_average_space = params.get("psd_average_space", "linear")

            if psd_average_space == "linear":
                avg_powers = np.mean(psd_array, axis=0)

            elif psd_average_space == "log":
                avg_log_powers = np.mean(np.log10(psd_array), axis=0)
                avg_powers = 10 ** avg_log_powers

            else:
                raise ValueError(
                    "psd_average_space must be either 'linear' or 'log'."
                )

            fm = SpectralModel(**spectral_model_params)
            fm.fit(freqs, avg_powers, fit_freq_range)

            exp = fm.get_params("aperiodic", "exponent")
            all_exponents.append(exp)

    all_exponents = np.array(all_exponents)

    mean_exp = np.mean(all_exponents)
    std_exp = np.std(all_exponents)

    if plot:
        plt.hist(all_exponents, bins=20)

        plt.axvline(
            mean_exp,
            linestyle="dashed"
        )

        plt.title(
            f"File-Level PSD-Averaged Exponent Distribution\n"
            f"status={status}, electrodes={electrode_labels}"
        )

        plt.xlabel("Estimated exponent")
        plt.ylabel("Count")
        plt.show()

    return (
        all_exponents,
        float(mean_exp),
        float(std_exp)
    )

# ======================================================================
# ======================================================================

def sweep_electrodes_psd_2D_eeg(
    electrode_sweeps,
    data_dir,
    status="ec",
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep across user-defined electrode groups and estimate the exponent
    from PSDs averaged across each selected group.

    Parameters
    ----------
    electrode_sweeps : list of list of str
        Each inner list defines one electrode group to average.
        Example:
        [
            ["E76"],
            ["E76", "E75"],
            ["E76", "E75", "E70"]
        ]

    data_dir : str or pathlib.Path
        Folder containing the saved PSD data files.

    status : str, optional, default: "ec"
        File-status string used to filter files, for example "ec" or "eo".

    base_params : dict, optional
        Base parameter dictionary.

    plot : bool, optional, default: True
        Whether to plot sweep results.

    return_full : bool, optional, default: False
        If True, also return all exponent distributions.

    **param_overrides
        Extra parameters passed into avg_psd_distribution_2D.

    Returns
    -------
    electrode_counts : np.ndarray
        Number of electrodes in each sweep group.

    mean_exps : np.ndarray
        Mean estimated exponent for each electrode group.

    std_exps : np.ndarray
        Standard deviation of estimated exponents for each electrode group.

    errors : np.ndarray
        Difference between mean estimated exponent and true exponent,
        if true exponent is provided.

    fig : matplotlib figure or None
        Figure object if plot=True.
    """

    import numpy as np

    from dissertation_simulations.eeg_data import avg_psd_distribution_2D_eeg
    from dissertation_simulations.plotting import plot_sweep_results

    electrode_counts = []
    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig = None

    true_exp = param_overrides.get("exponent", None)

    # -----------------------------------
    # Sweep defined electrode groups
    # -----------------------------------
    for electrode_labels in electrode_sweeps:

        all_exps, mean_exp, std_exp = avg_psd_distribution_2D_eeg(
            n_repeats=1,
            grid_shape=(1, len(electrode_labels)),
            data_dir=data_dir,
            electrode_labels=electrode_labels,
            status=status,
            base_params=base_params,
            plot=False,
            **param_overrides
        )

        electrode_counts.append(len(electrode_labels))
        mean_exps.append(mean_exp)
        std_exps.append(std_exp)

        if true_exp is not None:
            errors.append(mean_exp - true_exp)
        else:
            errors.append(np.nan)

        if return_full:
            all_distributions.append(all_exps)

    electrode_counts = np.array(electrode_counts)
    mean_exps = np.array(mean_exps)
    std_exps = np.array(std_exps)
    errors = np.array(errors)

    # -----------------------------------
    # Plot
    # -----------------------------------
    if plot:

        fig = plot_sweep_results(
            x_values=electrode_counts,
            mean_exps=mean_exps,
            errors=errors,
            true_exp=true_exp,
            std_exps=std_exps,
            xlabel="Number of electrodes",
            title_prefix="EEG PSD exponent estimation"
        )

    print(param_overrides)

    # -----------------------------------
    # Return
    # -----------------------------------
    if return_full:
        return (
            electrode_counts,
            mean_exps,
            std_exps,
            errors,
            all_distributions,
            fig
        )

    return (
        electrode_counts,
        mean_exps,
        std_exps,
        errors,
        fig
    )


# ===================================================================
# ===================================================================

def avg_specmod_distribution_2D_eeg(
    n_repeats,
    grid_shape,
    data_dir,
    electrode_labels,
    status="ec",
    plot=True,
    base_params=None,
    fit_freq_range=None,
    verbose=False,
    **param_overrides
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from dissertation_simulations.params import (
        electrode_sim_params_ap,
        spectral_model_params,
        freq_range
    )

    from specparam import SpectralModel

    if base_params is None:
        params = electrode_sim_params_ap.copy()
    else:
        params = base_params.copy()

    params.update(param_overrides)

    if fit_freq_range is None:
        fit_freq_range = freq_range

    if verbose:
        print("Using fit_freq_range:", fit_freq_range)

    data_dir = Path(data_dir)

    file_list = os.listdir(data_dir)
    file_list = [cfile for cfile in file_list if cfile[0] != "."]
    file_list = [cfile for cfile in file_list if status in cfile]

    if len(file_list) == 0:
        raise ValueError(
            f"No files found for status '{status}' in {data_dir}"
        )

    all_mean_exponents = []

    for _ in range(n_repeats):

        for cfile in file_list:

            loaded_data = np.load(data_dir / cfile)

            freqs = loaded_data["arr_0"]
            powers = loaded_data["arr_1"]
            electrodes = loaded_data["arr_2"]

            # Data shape is n_blocks x electrodes x freqs.
            block_index = params.get("block_index", 0)
            powers = powers[block_index, :, :]

            electrode_exponents = []

            for e_label in electrode_labels:

                if e_label not in electrodes:
                    raise ValueError(
                        f"Electrode {e_label} not found in file {cfile}"
                    )

                e_ind = np.where(electrodes == e_label)[0][0]

                electrode_power = powers[e_ind, :]

                fm = SpectralModel(**spectral_model_params)

                fm.fit(
                    freqs,
                    electrode_power,
                    fit_freq_range
                )

                exp = fm.get_params(
                    "aperiodic",
                    "exponent"
                )

                electrode_exponents.append(exp)

            all_mean_exponents.append(
                np.mean(electrode_exponents)
            )

    all_mean_exponents = np.array(all_mean_exponents)

    mean_exp = np.mean(all_mean_exponents)
    std_exp = np.std(all_mean_exponents)

    if plot:

        plt.hist(
            all_mean_exponents,
            bins=20,
            alpha=0.7
        )

        plt.axvline(
            mean_exp,
            linestyle="dashed"
        )

        plt.title(
            f"File-Level Individual-PSD Exponent Distribution\n"
            f"status={status}, electrodes={electrode_labels}"
        )

        plt.xlabel(
            "Mean electrode exponent per file"
        )

        plt.ylabel("Count")

        plt.show()

    return (
        all_mean_exponents,
        float(mean_exp),
        float(std_exp)
    )


# ====================================================
# ====================================================

def sweep_electrodes_specmod_2D_eeg(
    electrode_sweeps,
    data_dir,
    status="ec",
    base_params=None,
    plot=True,
    return_full=False,
    **param_overrides
):
    """
    Sweep across user-defined electrode groups using real EEG PSD data.

    For each electrode group:
    - Load all matching EEG PSD files
    - Fit a spectral model to each selected electrode PSD individually
    - Average the fitted exponents across selected electrodes per file
    - Analyze the distribution of file-level mean exponents

    Parameters
    ----------
    electrode_sweeps : list of list of str
        Each inner list defines one electrode group to average.
        Example:
        [
            ["E76"],
            ["E76", "E75"],
            ["E76", "E75", "E70"]
        ]

    data_dir : str or pathlib.Path
        Folder containing saved EEG PSD files.

    status : str, optional, default: "ec"
        File-status string used to filter files, for example "ec" or "eo".

    base_params : dict or None
        Base parameter dictionary.

    plot : bool, optional, default: True
        Whether to plot estimation error / mean exponent.

    return_full : bool, optional, default: False
        If True, also return full exponent distributions.

    **param_overrides
        Extra parameters passed to avg_specmod_exponent_distribution_2D.

    Returns
    -------
    electrode_counts : np.ndarray
        Number of electrodes in each sweep group.

    mean_exps : np.ndarray
        Mean estimated exponent per electrode group.

    std_exps : np.ndarray
        Standard deviation of estimated exponents.

    errors : np.ndarray
        Bias values, estimated - true exponent.

    all_distributions : list, optional
        Returned only if return_full=True.

    fig : matplotlib figure or None
        Returned plot figure.
    """

    import numpy as np

    from dissertation_simulations.eeg_data import (
        avg_specmod_distribution_2D_eeg
    )
    from dissertation_simulations.plotting import plot_sweep_results

    electrode_counts = []
    mean_exps = []
    std_exps = []
    errors = []
    all_distributions = []
    fig = None

    true_exp = param_overrides.get("exponent", None)

    # ---------------------------------
    # Sweep across user-defined electrodes
    # ---------------------------------
    for electrode_labels in electrode_sweeps:

        all_exps, mean_exp, std_exp = (avg_specmod_distribution_2D_eeg(
            n_repeats=1,
            grid_shape=(1, len(electrode_labels)),
            data_dir=data_dir,
            electrode_labels=electrode_labels,
            status=status,
            base_params=base_params,
            plot=False,
            **param_overrides
        )
)

        electrode_counts.append(len(electrode_labels))
        mean_exps.append(mean_exp)
        std_exps.append(std_exp)

        if true_exp is not None:
            errors.append(mean_exp - true_exp)
        else:
            errors.append(np.nan)

        if return_full:
            all_distributions.append(all_exps)

    electrode_counts = np.array(electrode_counts)
    mean_exps = np.array(mean_exps)
    std_exps = np.array(std_exps)
    errors = np.array(errors)

    # ---------------------------------
    # Plot
    # ---------------------------------
    if plot:

        fig = plot_sweep_results(
            x_values=electrode_counts,
            mean_exps=mean_exps,
            errors=errors,
            true_exp=true_exp,
            std_exps=std_exps,
            xlabel="Number of electrodes",
            title_prefix="EEG spectral model exponent estimation"
        )

    # ---------------------------------
    # Return
    # ---------------------------------
    if return_full:
        return (
            electrode_counts,
            mean_exps,
            std_exps,
            errors,
            all_distributions,
            fig
        )

    return (
            electrode_counts,
            mean_exps,
            std_exps,
            errors,
            fig
        )


def get_electrode_power(powers, electrodes, e_label, freqs, cfile):
    import numpy as np

    electrodes = np.asarray(electrodes)
    powers = np.squeeze(powers)

    if e_label not in electrodes:
        raise ValueError(
            f"Electrode {e_label} not found in file {cfile}"
        )

    e_ind = np.where(electrodes == e_label)[0][0]

    # Case 1: powers is electrodes x freqs
    if powers.ndim == 2 and powers.shape[0] == len(electrodes):
        electrode_power = powers[e_ind, :]

    # Case 2: powers is freqs x electrodes
    elif powers.ndim == 2 and powers.shape[1] == len(electrodes):
        electrode_power = powers[:, e_ind]

    else:
        raise ValueError(
            f"Cannot match powers shape {powers.shape} to "
            f"{len(electrodes)} electrodes in file {cfile}"
        )

    electrode_power = np.squeeze(electrode_power)

    if electrode_power.ndim != 1:
        raise ValueError(
            f"Electrode PSD is not 1D after extraction. "
            f"Got shape {electrode_power.shape} for {e_label} in {cfile}"
        )

    if len(electrode_power) != len(freqs):
        raise ValueError(
            f"PSD length does not match freqs length for {e_label} in {cfile}. "
            f"PSD length={len(electrode_power)}, freqs length={len(freqs)}"
        )

    return electrode_power