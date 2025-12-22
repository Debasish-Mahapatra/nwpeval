"""Matthews Correlation Coefficient (MCC)."""
import numpy as np
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
    
    # Convert to float to avoid integer overflow
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Handle division by zero
    if denominator == 0:
        return 0.0
    
    result = numerator / denominator
    
    # Clip to valid range [-1, 1]
    return np.clip(result, -1.0, 1.0)

