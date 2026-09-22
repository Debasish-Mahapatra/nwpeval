"""Symmetric Extreme Dependency Score (SEDS)."""
import numpy as np
import xarray as xr
from ._base import contingency, ratio, safe_log


def seds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Symmetric Extreme Dependency Score (SEDS) for a given threshold.

    SEDS = [log(p) + log(p_F)] / log(s) - 1
    where p = (TP+FN)/N is the base rate, p_F = (TP+FP)/N is the forecast rate,
    and s = TP/N is the joint sample probability.

    SEDS equals 1 for a perfect forecast (p = p_F = s) and 0 for a no-skill
    forecast that is independent of observations (s = p * p_F). With no hits
    but events both observed and forecast, it takes its limit, -1.

    Reference: Hogan, R.J. and Mason, I.B. (2012), Forecast Verification,
    2nd ed., chapter on Deterministic Forecasts of Binary Events.

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed SEDS values. NaN where no event was
        observed or forecast, and where every point is a hit (s = 1).
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    n = tp + fp + fn + tn
    p = ratio(tp + fn, n)
    p_f = ratio(tp + fp, n)
    s = ratio(tp, n)

    log_s = safe_log(s)
    regular = (safe_log(p) + safe_log(p_f)) / log_s.where(log_s != 0) - 1
    no_hits = (tp == 0) & (p > 0) & (p_f > 0)
    return xr.where(no_hits, -1.0, xr.where(tp > 0, regular, np.nan))
