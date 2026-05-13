def gen_exponents(n_exponents, exp_range):
    """
    Generate random exponents within a given range.

    Parameters
    ----------
    n_exponents : int
        Number of exponents to generate
    exp_range : tuple or list
        (min, max) range for exponents

    Returns
    -------
    exponents : list
        List of randomly generated exponents
    """

    import numpy as np

    exponents = np.random.uniform(exp_range[0], exp_range[1], n_exponents)

    return exponents.tolist()

# ============================================
# ============================================

def generate_random_electrode_exponents(
    n_electrodes,
    exp_range,
    distribution="uniform"
):
    """
    Generate random exponents for simulated electrodes.

    Parameters
    ----------
    n_electrodes : int
        Number of electrode exponents to generate.

    exp_range : tuple
        (min_exp, max_exp)

    distribution : str, optional, default: "uniform"
        Distribution type.
        Options:
        - "uniform"
        - "normal"

    Returns
    -------
    exponents : np.ndarray
        Array of generated exponents.
    """

    import numpy as np

    low, high = exp_range

    if distribution == "uniform":

        exponents = np.random.uniform(
            low,
            high,
            n_electrodes
        )

    elif distribution == "normal":

        mean = (low + high) / 2
        std = (high - low) / 4

        exponents = np.random.normal(
            mean,
            std,
            n_electrodes
        )

        exponents = np.clip(exponents, low, high)

    else:
        raise ValueError(
            "distribution must be 'uniform' or 'normal'"
        )

    return exponents