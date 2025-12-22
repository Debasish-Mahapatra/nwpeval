"""Weighted Mean Absolute Error (WMAE)."""
import numpy as np


def wmae(obs_data, model_data, weights, dim=None):
    """
    Compute the Weighted Mean Absolute Error (WMAE).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        weights (xarray.DataArray): The weights for each data point.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed WMAE values.
    """
    weighted_abs_error = weights * np.abs(model_data - obs_data)
    return weighted_abs_error.sum(dim=dim) / weights.sum(dim=dim)
