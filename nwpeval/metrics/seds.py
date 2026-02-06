"""Symmetric Extreme Dependency Score (SEDS)."""
import numpy as np
import xarray as xr
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
    
    # Avoid division by zero
    pod = xr.where((tp + fn) == 0, np.nan, tp / (tp + fn))
    pofd = xr.where((fp + tn) == 0, np.nan, fp / (fp + tn))
    
    # Clip to avoid log(0) and log(1) issues
    eps = 1e-10
    pod_safe = pod.clip(eps, 1 - eps)
    pofd_safe = pofd.clip(eps, 1 - eps)
    
    numerator = np.log(pod_safe) - np.log(pofd_safe) + np.log(1 - pofd_safe) - np.log(1 - pod_safe)
    denominator = np.log(pod_safe) + np.log(1 - pofd_safe)
    
    return xr.where(denominator == 0, np.nan, numerator / denominator)

