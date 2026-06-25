def load_distribution_summary(path):
    """Load one sweep summary JSON file."""

    import json
    import numpy as np

    with open(path, "r") as f:
        data = json.load(f)

    electrodes = np.array(data["electrodes"])
    distributions = [np.array(d) for d in data["all_distributions"]]

    return electrodes, distributions