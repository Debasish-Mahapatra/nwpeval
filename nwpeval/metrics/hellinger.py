"""Hellinger Distance."""
import numpy as np
from ._base import distributions


def hellinger(obs_data, model_data, dim=None):
    """
    Compute the Hellinger distance between two empirical distributions.

    H(P, Q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2)) where P and Q are
    probability distributions formed by normalising the inputs over `dim`.
    Inputs must be non-negative.

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
        xarray.DataArray: The Hellinger distance.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    return np.sqrt(0.5 * ((np.sqrt(obs_prob) - np.sqrt(model_prob)) ** 2).sum(dim=dim, min_count=1))
