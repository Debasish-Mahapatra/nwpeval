"""Geometric Mean Bias (GMB)."""
import numpy as np


def gmb(obs_data, model_data, dim=None):
    """
    Compute the Geometric Mean Bias (GMB).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed GMB values.
    """
    model_mean = np.exp(np.log(model_data).mean(dim=dim))
    obs_mean = np.exp(np.log(obs_data).mean(dim=dim))
    return model_mean / obs_mean
