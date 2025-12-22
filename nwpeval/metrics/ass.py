"""Absolute Skill Score (ASS)."""
import numpy as np


def ass(obs_data, model_data, reference_error, dim=None):
    """
    Compute the Absolute Skill Score (ASS).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_error (xarray.DataArray): The reference error values.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed ASS values.
    """
    abs_error = np.abs(model_data - obs_data)
    return 1 - abs_error / reference_error
