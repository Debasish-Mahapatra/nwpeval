"""Odds Ratio Skill Score (ORSS)."""
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
    
    odds_ratio = (tp * tn) / (fp * fn)
    return (odds_ratio - 1) / (odds_ratio + 1)
