"""Renyi Divergence."""
import numpy as np
import xarray as xr


def renyi(obs_data, model_data, alpha, dim=None):
    """
    Compute the Renyi Divergence of order alpha (alpha != 1).

    D_alpha(P || Q) = 1/(alpha - 1) * log(sum(p^alpha * q^(1-alpha)))
    where p and q are probability distributions formed by normalising
    `obs_data` and `model_data` over `dim`.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        alpha (float): The order parameter, alpha != 1.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Renyi Divergence values.
    """
    if alpha == 1:
        raise ValueError("Renyi divergence is undefined at alpha == 1.")
    eps = np.finfo(float).tiny
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data > 0, model_data, eps)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)
    inner = (obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim)
    return xr.where(inner <= 0, np.nan, np.log(inner) / (alpha - 1))
