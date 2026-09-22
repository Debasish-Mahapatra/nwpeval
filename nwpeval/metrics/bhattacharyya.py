"""Bhattacharyya Distance."""
import numpy as np
import xarray as xr
from ._base import distributions


def bhattacharyya(obs_data, model_data, dim=None):
    """
    Compute the Bhattacharyya Distance between two empirical distributions.

    D_B(P, Q) = -log(sum(sqrt(p * q))) where p and q are probability
    distributions formed by normalising `obs_data` and `model_data` over
    `dim`. Inputs must be non-negative.

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Bhattacharyya Distance values.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    inner = (np.sqrt(obs_prob * model_prob)).sum(dim=dim, min_count=1)
    return xr.where(inner <= 0, np.nan, -np.log(inner))
