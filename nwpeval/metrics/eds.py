"""Extreme Dependency Score (EDS)."""
import numpy as np
from ._base import confusion_matrix


def eds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Extreme Dependency Score (EDS) for a given threshold.
    
    EDS is designed for rare events and measures the association between
    forecasts and observations.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed EDS values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    n = tp + fp + fn + tn
    p = (tp + fn) / n  # Base rate
    H = tp / (tp + fn)  # Hit rate
    
    return (np.log(H) - np.log(p)) / (np.log(H) + np.log(p))
