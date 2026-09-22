"""Equitable Threat Score (ETS)."""
from ._base import contingency, ratio


def ets(obs_data, model_data, threshold, dim=None):
    """
    Compute the Equitable Threat Score (ETS) for a given threshold.

    ETS = (TP - TP_random) / (TP + FP + FN - TP_random),
    TP_random = (TP + FP) * (TP + FN) / N.

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ETS values. NaN where the
        denominator is zero (no event observed or forecast).
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    n = tp + fp + fn + tn
    hits_random = ratio((tp + fp) * (tp + fn), n)
    return ratio(tp - hits_random, tp + fp + fn - hits_random)
