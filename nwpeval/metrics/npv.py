"""Negative Predictive Value (NPV)."""
from ._base import confusion_matrix


def npv(obs_data, model_data, threshold, dim=None):
    """
    Compute the Negative Predictive Value (NPV) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed NPV values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    return tn / (tn + fn)
