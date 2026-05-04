"""Jaccard Similarity Coefficient.

Alias of :func:`csi`. The Jaccard similarity (in machine learning) and
the Critical Success Index (in forecast verification) are the same metric:
TP / (TP + FP + FN).
"""
from .csi import csi


def jaccard(obs_data, model_data, threshold, dim=None):
    """
    Compute the Jaccard Similarity Coefficient for a given threshold.

    Jaccard is mathematically identical to the Critical Success Index (CSI).
    This function is kept as an alias for backward compatibility.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Jaccard values (same as CSI).
    """
    return csi(obs_data, model_data, threshold, dim=dim)
