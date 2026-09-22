"""Weighted Mean Absolute Error (WMAE)."""
import numpy as np
from ._base import paired, ratio


def wmae(obs_data, model_data, weights, dim=None):
    """
    Compute the Weighted Mean Absolute Error (WMAE).

    WMAE = sum(weights * |model - obs|) / sum(weights).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        weights (xarray.DataArray): The weights for each data point. Weights
            at missing obs/model points are ignored.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed WMAE values. Returns NaN where the
        sum of weights along `dim` is zero.
    """
    obs_data, model_data, weights = paired(obs_data, model_data, weights)
    weighted_abs_error = (weights * np.abs(model_data - obs_data)).sum(dim=dim)
    return ratio(weighted_abs_error, weights.sum(dim=dim))
