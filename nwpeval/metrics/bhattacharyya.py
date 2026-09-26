"""Bhattacharyya Distance."""
import numpy as np
import xarray as xr
from ._base import distributions, safe_log


def bhattacharyya(obs_data, model_data, dim=None):
    """
    Compute the Bhattacharyya Distance between two empirical distributions.

    D_B(P, Q) = -log(sum(sqrt(p * q))) where p and q are probability
    distributions formed by normalising `obs_data` and `model_data` over
    `dim`. Inputs must be non-negative. The distance is +inf where the two
    fields share no mass.

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
        xarray.DataArray: The computed Bhattacharyya Distance values.
    """
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    inner = (np.sqrt(obs_prob * model_prob)).sum(dim=dim, min_count=1)
    # inner == 0 means no shared mass: the distance is +inf
    return xr.where(inner > 0, -safe_log(inner), np.inf).where(inner.notnull())
