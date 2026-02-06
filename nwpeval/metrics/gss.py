"""Gilbert Skill Score (GSS)."""
import xarray as xr
from ._base import confusion_matrix


def gss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Gilbert Skill Score (GSS) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed GSS values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    n = tp + fp + fn + tn
    hits_random = xr.where(n == 0, 0.0, (tp + fp) * (tp + fn) / n)
    
    numerator = tp - hits_random
    denominator = tp + fp + fn - hits_random
    
    return xr.where(denominator == 0, 0.0, numerator / denominator)

