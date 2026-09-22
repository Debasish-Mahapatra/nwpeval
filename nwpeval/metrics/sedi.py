"""Symmetric Extremal Dependence Index (SEDI)."""
import numpy as np
import xarray as xr
from ._base import contingency, ratio, safe_log


def sedi(obs_data, model_data, threshold, dim=None):
    """
    Compute the Symmetric Extremal Dependence Index (SEDI) for a given threshold.

    SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)]
    with hit rate H = TP/(TP+FN) and false-alarm rate F = FP/(FP+TN)
    (Ferro and Stephenson, 2011).

    When H or F is 0 or 1 a logarithm diverges and SEDI takes its limit:
    1 when H = 1 or F = 0, -1 when H = 0 or F = 1. Where two terms with
    opposite limits diverge together (H = F = 0 or H = F = 1) the limit
    depends on the path and NaN is returned.

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed SEDI values. NaN where H or F is
        undefined (no event or no non-event observed).
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    h = ratio(tp, tp + fn)
    f = ratio(fp, fp + tn)

    log_f, log_h = safe_log(f), safe_log(h)
    log_1f, log_1h = safe_log(1 - f), safe_log(1 - h)
    regular = (log_f - log_h - log_1f + log_1h) / (log_f + log_h + log_1f + log_1h)

    to_plus = (h == 1).astype(int) + (f == 0).astype(int)
    to_minus = (h == 0).astype(int) + (f == 1).astype(int)
    limit = xr.where(to_minus == 0, 1.0, xr.where(to_plus == 0, -1.0, np.nan))
    value = xr.where(to_plus + to_minus == 0, regular, limit)
    return value.where(h.notnull() & f.notnull())
