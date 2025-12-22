"""Hellinger Distance."""
import numpy as np


def hellinger(obs_data, model_data, dim=None):
    """
    Compute the Hellinger Distance.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Hellinger Distance values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return np.sqrt(0.5 * ((np.sqrt(obs_prob) - np.sqrt(model_prob)) ** 2).sum(dim=dim))
