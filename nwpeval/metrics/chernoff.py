"""Chernoff Distance."""
import numpy as np


def chernoff(obs_data, model_data, alpha, dim=None):
    """
    Compute the Chernoff Distance.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        alpha (float): The parameter for the Chernoff Distance (0 < alpha < 1).
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Chernoff Distance values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return -np.log((obs_prob ** alpha * model_prob ** (1 - alpha)).sum(dim=dim))
