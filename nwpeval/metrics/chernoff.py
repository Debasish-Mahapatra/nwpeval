"""Chernoff Distance."""
import numpy as np
import xarray as xr
from ._base import distributions


def chernoff(obs_data, model_data, alpha, dim=None):
    """
    Compute the Chernoff Distance.

    chernoff = -log(sum(p^alpha * q^(1-alpha))) where p, q are probability
    distributions formed by normalising `obs_data` and `model_data` to sum
    to one over `dim`. Inputs must be non-negative.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        alpha (float): The parameter for the Chernoff Distance (0 < alpha < 1).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Chernoff Distance values.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    inner = (obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim, min_count=1)
    return xr.where(inner <= 0, np.nan, -np.log(inner))
