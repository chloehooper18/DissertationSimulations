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

# ============================================
# ============================================

def generate_2D_exponent_matrix(
    grid_shape,
    exp_range=(-2, -1),
    distribution="uniform",
    gradient_axis=None,
    gradient_range=None,
    random_state=None
):
    """
    Generate a 2D matrix of exponents for electrode simulations.

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of electrode grid as (n_rows, n_cols).

    exp_range : tuple, optional, default: (-2, -1)
        Range of exponent values for random generation.

    distribution : {"uniform", "normal"}, optional
        Distribution used for random exponent generation.

    gradient_axis : {None, "rows", "cols"}, optional
        If specified, generate a smooth gradient instead of random values.

        - "rows" : vary across rows
        - "cols" : vary across columns

    gradient_range : tuple or None, optional
        Min/max exponent values for gradient.
        Example: (-2, -1)

    random_state : int or None, optional
        Random seed.

    Returns
    -------
    exponent_matrix : np.ndarray
        2D array of shape (n_rows, n_cols).
    """

    import numpy as np

    if random_state is not None:
        np.random.seed(random_state)

    n_rows, n_cols = grid_shape

    # ---------------------------------
    # Gradient matrix
    # ---------------------------------
    if gradient_axis is not None:

        if gradient_range is None:
            gradient_range = exp_range

        start_exp, end_exp = gradient_range

        # Vary across rows
        if gradient_axis == "rows":

            values = np.linspace(start_exp, end_exp, n_rows)

            exponent_matrix = np.tile(
                values[:, np.newaxis],
                (1, n_cols)
            )

        # Vary across columns
        elif gradient_axis == "cols":

            values = np.linspace(start_exp, end_exp, n_cols)

            exponent_matrix = np.tile(
                values,
                (n_rows, 1)
            )

        else:
            raise ValueError(
                "gradient_axis must be None, 'rows', or 'cols'"
            )

    # ---------------------------------
    # Random matrix
    # ---------------------------------
    else:

        if distribution == "uniform":

            exponent_matrix = np.random.uniform(
                exp_range[0],
                exp_range[1],
                size=(n_rows, n_cols)
            )

        elif distribution == "normal":

            mean_exp = np.mean(exp_range)
            std_exp = (exp_range[1] - exp_range[0]) / 4

            exponent_matrix = np.random.normal(
                mean_exp,
                std_exp,
                size=(n_rows, n_cols)
            )

        else:
            raise ValueError(
                "distribution must be 'uniform' or 'normal'"
            )

    return exponent_matrix

# ============================================
# ============================================

def generate_spatial_exponent_matrix(
    grid_shape,
    exp_min=-2,
    exp_max=-0.5,
    pattern="horizontal"
):
    """
    Generate a spatially varying exponent matrix.

    Parameters
    ----------
    grid_shape : tuple
        (n_rows, n_cols)

    exp_min : float
        Minimum exponent value

    exp_max : float
        Maximum exponent value

    pattern : str
        Spatial pattern:
        - "horizontal"
        - "vertical"
        - "radial"
        - "diagonal"

    Returns
    -------
    exponent_matrix : np.ndarray
    """

    import numpy as np

    n_rows, n_cols = grid_shape

    # Coordinate grids
    x = np.linspace(0, 1, n_cols)
    y = np.linspace(0, 1, n_rows)

    X, Y = np.meshgrid(x, y)

    # -------------------------
    # Spatial patterns
    # -------------------------
    if pattern == "horizontal":

        values = X

    elif pattern == "vertical":

        values = Y

    elif pattern == "diagonal":

        values = (X + Y) / 2

    elif pattern == "radial":

        cx, cy = 0.5, 0.5
        values = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Normalize radial distances
        range_val = values.max() - values.min()

        if range_val == 0:
            values = np.zeros_like(values)
        else:
            values = (values - values.min()) / range_val

    else:
        raise ValueError("Unknown pattern")

    # Scale into exponent range
    exponent_matrix = exp_min + values * (exp_max - exp_min)

    return exponent_matrix