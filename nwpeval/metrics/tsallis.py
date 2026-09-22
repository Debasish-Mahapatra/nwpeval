"""Tsallis Divergence."""
import numpy as np
import xarray as xr
from ._base import distributions


def tsallis(obs_data, model_data, alpha, dim=None):
    """
    Compute the Tsallis Divergence of order alpha (alpha != 1).

    D_alpha(P || Q) = 1/(alpha - 1) * (sum(p^alpha * q^(1-alpha)) - 1)

    Points missing (NaN) or negative in either input are dropped from both
    distributions, so P and Q always cover the same bins.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        alpha (float): The order parameter, alpha != 1.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Tsallis Divergence values.
    """
    if alpha == 1:
        raise ValueError("Tsallis divergence is undefined at alpha == 1.")
    eps = np.finfo(float).tiny
    obs_prob, model_prob = distributions(obs_data, model_data, dim)
    model_prob = xr.where(model_prob == 0, eps, model_prob)
    inner = (obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim, min_count=1)
    return (inner - 1) / (alpha - 1)
