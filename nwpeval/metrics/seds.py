"""Symmetric Extreme Dependency Score (SEDS)."""
import numpy as np
from ._base import confusion_matrix


def seds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Symmetric Extreme Dependency Score (SEDS) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed SEDS values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    pod = tp / (tp + fn)
    pofd = fp / (fp + tn)
    
    return (np.log(pod) - np.log(pofd) + np.log(1 - pofd) - np.log(1 - pod)) / (np.log(pod) + np.log(1 - pofd))
