"""Odds Ratio Skill Score (ORSS)."""
from ._base import contingency, ratio


def orss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Odds Ratio Skill Score (ORSS) for a given threshold.

    ORSS = (TP*TN - FP*FN) / (TP*TN + FP*FN).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ORSS values. NaN where the
        denominator is zero.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(tp * tn - fp * fn, tp * tn + fp * fn)
