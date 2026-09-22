"""Total Variation Distance."""
import numpy as np
from ._base import distributions


def tv(obs_data, model_data, dim=None):
    """
    Compute the Total Variation distance between two empirical distributions.

    TV(P, Q) = 0.5 * sum(|p - q|) where P and Q are probability distributions
    formed by normalising the inputs over `dim`. Inputs must be non-negative.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The total variation distance.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    return 0.5 * np.abs(obs_prob - model_prob).sum(dim=dim, min_count=1)
