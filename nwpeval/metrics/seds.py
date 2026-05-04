"""Symmetric Extreme Dependency Score (SEDS)."""
import numpy as np
import xarray as xr
from ._base import confusion_matrix


def seds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Symmetric Extreme Dependency Score (SEDS) for a given threshold.

    SEDS = [log(p) + log(p_F)] / log(s) - 1
    where p = (TP+FN)/N is the base rate, p_F = (TP+FP)/N is the forecast rate,
    and s = TP/N is the joint sample probability.

    SEDS equals 1 for a perfect forecast (p = p_F = s) and 0 for a no-skill
    forecast that is independent of observations (s = p * p_F).

    Reference: Hogan, R.J. and Mason, I.B. (2012), Forecast Verification,
    2nd ed., chapter on Deterministic Forecasts of Binary Events.

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

    n = tp + fp + fn + tn
    p = xr.where(n == 0, np.nan, (tp + fn) / n)
    p_f = xr.where(n == 0, np.nan, (tp + fp) / n)
    s = xr.where(n == 0, np.nan, tp / n)

    eps = 1e-10
    p_safe = p.clip(eps, 1 - eps)
    pf_safe = p_f.clip(eps, 1 - eps)
    s_safe = s.clip(eps, 1 - eps)

    numerator = np.log(p_safe) + np.log(pf_safe) - np.log(s_safe)
    denominator = np.log(s_safe)

    return xr.where(denominator == 0, np.nan, numerator / denominator)
