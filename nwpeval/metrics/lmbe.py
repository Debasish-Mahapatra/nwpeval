"""Logarithmic Mean Bias Error (LMBE)."""
import numpy as np


def lmbe(obs_data, model_data, dim=None):
    """
    Compute the Logarithmic Mean Bias Error (LMBE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed LMBE values.
    """
    return (np.log(model_data + 1) - np.log(obs_data + 1)).mean(dim=dim)
