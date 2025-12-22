"""Bhattacharyya Distance."""
import numpy as np


def bhattacharyya(obs_data, model_data, dim=None):
    """
    Compute the Bhattacharyya Distance.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Bhattacharyya Distance values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return -np.log((np.sqrt(obs_prob * model_prob)).sum(dim=dim))
