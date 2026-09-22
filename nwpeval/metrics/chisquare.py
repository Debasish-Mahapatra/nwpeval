"""Chi-Square Distance."""
import numpy as np
import xarray as xr
from ._base import distributions


def chisquare(obs_data, model_data, dim=None):
    """
    Compute the (Pearson) chi-square distance between two empirical
    distributions.

    chi2(P, Q) = sum((p - q)^2 / q) where P and Q are probability distributions
    formed by normalising the inputs over `dim`. Inputs must be non-negative.
    Bins where q == 0 contribute +inf when p > 0 and 0 when p == 0.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The chi-square distance.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)

    diff_sq = (obs_prob - model_prob) ** 2
    term = xr.where(
        model_prob == 0,
        xr.where(obs_prob == 0, 0.0, np.inf),
        diff_sq / xr.where(model_prob == 0, 1.0, model_prob),
    )
    return term.sum(dim=dim, min_count=1)
