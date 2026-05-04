"""Tsallis Divergence."""
import numpy as np
import xarray as xr


def tsallis(obs_data, model_data, alpha, dim=None):
    """
    Compute the Tsallis Divergence of order alpha (alpha != 1).

    D_alpha(P || Q) = 1/(alpha - 1) * (sum(p^alpha * q^(1-alpha)) - 1)

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
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data > 0, model_data, eps)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)
    inner = (obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim)
    return (inner - 1) / (alpha - 1)
