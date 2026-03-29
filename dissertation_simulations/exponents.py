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