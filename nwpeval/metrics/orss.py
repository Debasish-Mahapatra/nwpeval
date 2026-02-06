"""Odds Ratio Skill Score (ORSS)."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def orss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Odds Ratio Skill Score (ORSS) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed ORSS values.
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    # Handle cases where fp * fn = 0
    denominator = fp * fn
    odds_ratio = xr.where(denominator == 0, np.inf, (tp * tn) / denominator)
    
    # Handle infinite odds ratio
    result = xr.where(odds_ratio == np.inf, 1.0, (odds_ratio - 1) / (odds_ratio + 1))
    
    return result

