"""Median Absolute Deviation (MAD)."""
import numpy as np


def mad(obs_data, model_data, dim=None):
    """
    Compute the Median Absolute Deviation (MAD).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MAD values.
    """
    return (np.abs(model_data - model_data.median(dim=dim))).median(dim=dim)
