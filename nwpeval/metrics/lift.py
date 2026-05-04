"""Lift metric."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def lift(obs_data, model_data, threshold, dim=None):
    """
    Compute the Lift for a given threshold.

    Lift = precision / base_rate = [TP / (TP+FP)] / [(TP+FN) / N]
    where N = TP + FP + FN + TN.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Lift values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)

    n = tp + fp + fn + tn
    precision = xr.where((tp + fp) == 0, np.nan, tp / (tp + fp))
    base_rate = xr.where(n == 0, np.nan, (tp + fn) / n)

    return xr.where(base_rate == 0, np.nan, precision / base_rate)
