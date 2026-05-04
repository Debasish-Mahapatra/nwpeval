"""Median Absolute Deviation (MAD) of residuals."""
import numpy as np


def mad(obs_data, model_data, dim=None):
    """
    Compute the Median Absolute Deviation (MAD) of forecast residuals.

    Residuals are defined as model_data - obs_data. MAD is the median
    of |residuals - median(residuals)|.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed MAD values.
    """
    residuals = model_data - obs_data
    return np.abs(residuals - residuals.median(dim=dim)).median(dim=dim)
