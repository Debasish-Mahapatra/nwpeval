"""Equitable Threat Score (ETS)."""
from ._base import confusion_matrix


def ets(obs_data, model_data, threshold, dim=None):
    """
    Compute the Equitable Threat Score (ETS) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed ETS values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    hits_random = (tp + fp) * (tp + fn) / (tp + fp + fn + tn)
    return (tp - hits_random) / (tp + fp + fn - hits_random)
