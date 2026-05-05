"""Weighted Mean Absolute Error (WMAE)."""
import numpy as np
import xarray as xr


def wmae(obs_data, model_data, weights, dim=None):
    """
    Compute the Weighted Mean Absolute Error (WMAE).

    WMAE = sum(weights * |model - obs|) / sum(weights).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        weights (xarray.DataArray): The weights for each data point.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed WMAE values. Returns NaN where the
        sum of weights along `dim` is zero.
    """
    weighted_abs_error = (weights * np.abs(model_data - obs_data)).sum(dim=dim)
    weight_total = weights.sum(dim=dim)
    return xr.where(weight_total == 0, np.nan, weighted_abs_error / weight_total)
