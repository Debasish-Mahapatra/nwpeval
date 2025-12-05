"""Rényi Divergence."""
import numpy as np


def renyi(obs_data, model_data, alpha, dim=None):
    """
    Compute the Rényi Divergence.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        alpha (float): The parameter for the Rényi Divergence (alpha != 1).
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed Rényi Divergence values.
    """
    obs_prob = obs_data / obs_data.sum(dim=dim)
    model_prob = model_data / model_data.sum(dim=dim)
    return (1 / (alpha - 1)) * np.log((obs_prob ** alpha / model_prob ** (alpha - 1)).sum(dim=dim))
