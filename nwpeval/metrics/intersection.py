"""Histogram Intersection."""
import numpy as np
import xarray as xr


def intersection(obs_data, model_data, dim=None):
    """
    Compute the histogram-intersection similarity.

    intersection(P, Q) = sum(min(p, q)) where P and Q are probability
    distributions formed by normalising the inputs over `dim`. Inputs
    must be non-negative.

    Args:
        obs_data (xarray.DataArray): The observed data (must be >= 0).
        model_data (xarray.DataArray): The modeled data (must be >= 0).
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The intersection score.
    """
    obs_safe = xr.where(obs_data >= 0, obs_data, np.nan)
    model_safe = xr.where(model_data >= 0, model_data, np.nan)
    obs_total = obs_safe.sum(dim=dim)
    model_total = model_safe.sum(dim=dim)
    obs_prob = xr.where(obs_total == 0, np.nan, obs_safe / obs_total)
    model_prob = xr.where(model_total == 0, np.nan, model_safe / model_total)
    return np.minimum(obs_prob, model_prob).sum(dim=dim)
