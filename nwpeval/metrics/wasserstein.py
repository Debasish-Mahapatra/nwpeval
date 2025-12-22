"""Wasserstein Distance."""
import numpy as np


def wasserstein(obs_data, model_data, dim=None):
    """
    Compute the Wasserstein Distance.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Wasserstein Distance values.
    """
    obs_cdf = obs_data.cumsum(dim=dim) / obs_data.sum(dim=dim)
    model_cdf = model_data.cumsum(dim=dim) / model_data.sum(dim=dim)
    return np.abs(obs_cdf - model_cdf).sum(dim=dim)
