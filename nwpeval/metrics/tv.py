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

    P and Q describe where the mass is (e.g. where the rain falls), so this
    compares the two fields point by point, not the spread of their values.
    For that, use :func:`wasserstein` or pass histograms (counts per bin).

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The total variation distance.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    return 0.5 * np.abs(obs_prob - model_prob).sum(dim=dim, min_count=1)
