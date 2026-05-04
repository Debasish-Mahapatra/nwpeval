"""Bhattacharyya Distance."""
import numpy as np
import xarray as xr


def bhattacharyya(obs_data, model_data, dim=None):
    """
    Compute the Bhattacharyya Distance between two empirical distributions.

    D_B(P, Q) = -log(sum(sqrt(p * q))) where p and q are probability
    distributions formed by normalising `obs_data` and `model_data` over
    `dim`. Inputs must be non-negative.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Bhattacharyya Distance values.
    """
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data >= 0, model_data, np.nan)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)
    inner = (np.sqrt(obs_prob * model_prob)).sum(dim=dim)
    return xr.where(inner <= 0, np.nan, -np.log(inner))
