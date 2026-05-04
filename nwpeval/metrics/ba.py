"""Balanced Accuracy (BA)."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def ba(obs_data, model_data, threshold, dim=None):
    """
    Compute the Balanced Accuracy (BA) for a given threshold.

    BA = 0.5 * (TPR + TNR) where TPR = TP/(TP+FN) and TNR = TN/(TN+FP).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed BA values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)

    tpr = xr.where((tp + fn) == 0, np.nan, tp / (tp + fn))
    tnr = xr.where((tn + fp) == 0, np.nan, tn / (tn + fp))
    return 0.5 * (tpr + tnr)
