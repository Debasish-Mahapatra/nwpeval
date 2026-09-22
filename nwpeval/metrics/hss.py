"""Heidke Skill Score (HSS)."""
from ._base import contingency, ratio


def hss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Heidke Skill Score (HSS) for a given threshold.

    HSS = 2 (TP*TN - FP*FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)].

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed HSS values. NaN where the
        denominator is zero.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return ratio(numerator, denominator)
