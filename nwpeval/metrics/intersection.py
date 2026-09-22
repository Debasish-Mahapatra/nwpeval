"""Histogram Intersection."""
import numpy as np
from ._base import distributions


def intersection(obs_data, model_data, dim=None):
    """
    Compute the histogram-intersection similarity.

    intersection(P, Q) = sum(min(p, q)) where P and Q are probability
    distributions formed by normalising the inputs over `dim`. Inputs
    must be non-negative.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The intersection score.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    return np.minimum(obs_prob, model_prob).sum(dim=dim, min_count=1)
