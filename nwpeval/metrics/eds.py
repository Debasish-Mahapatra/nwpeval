"""Extreme Dependency Score (EDS)."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def eds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Extreme Dependency Score (EDS) for a given threshold.

    EDS is designed for rare events and measures the association between
    forecasts and observations.

    EDS = 2 * log(p) / log(p * H) - 1
    where p is the base rate and H is the hit rate.

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
    p = xr.where(n == 0, np.nan, (tp + fn) / n)
    H = xr.where((tp + fn) == 0, np.nan, tp / (tp + fn))

    eps = 1e-10
    p_safe = p.clip(eps, 1 - eps)
    H_safe = H.clip(eps, 1 - eps)

    numerator = 2 * np.log(p_safe) - np.log(p_safe * H_safe)
    denominator = np.log(p_safe * H_safe)

    return xr.where(denominator == 0, np.nan, numerator / denominator)
