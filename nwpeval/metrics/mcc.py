"""Matthews Correlation Coefficient (MCC)."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def mcc(obs_data, model_data, threshold, dim=None):
    """
    Compute the Matthews Correlation Coefficient (MCC) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed MCC values (range: -1 to 1).
    """
    obs_binary = (obs_data >= threshold).astype(int)
    model_binary = (model_data >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(obs_binary, model_binary, dim)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Handle division by zero using xr.where to preserve xarray structure
    result = xr.where(denominator == 0, 0.0, numerator / denominator)
    
    return result.clip(-1.0, 1.0)
