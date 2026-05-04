"""Jaccard Similarity Coefficient."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def jaccard(obs_data, model_data, threshold, dim=None):
    """
    Compute the Jaccard Similarity Coefficient for a given threshold.

    Jaccard = |obs intersect model| / |obs union model| = TP / (TP + FP + FN).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Jaccard values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)

    union = tp + fp + fn
    return xr.where(union == 0, np.nan, tp / union)
