"""Extreme Dependency Score (EDS)."""
import numpy as np
import xarray as xr
from ._base import contingency, ratio, safe_log


def eds(obs_data, model_data, threshold, dim=None):
    """
    Compute the Extreme Dependency Score (EDS) for a given threshold.

    EDS is designed for rare events and measures the association between
    forecasts and observations (Stephenson et al., 2008):

    EDS = 2 * log(p) / log(p * H) - 1
    where p = (TP+FN)/N is the base rate and H = TP/(TP+FN) the hit rate.

    With no hits (H = 0) EDS takes its limit, -1. When every point is an
    observed and forecast event (p = H = 1) it is undefined (NaN).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed EDS values. NaN where no event was
        observed.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    p = ratio(tp + fn, tp + fp + fn + tn)
    h = ratio(tp, tp + fn)

    log_ph = safe_log(p * h)
    regular = 2 * safe_log(p) / log_ph.where(log_ph != 0) - 1
    return xr.where(h == 0, -1.0, xr.where(p.notnull(), regular, np.nan))
