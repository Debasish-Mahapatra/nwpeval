"""Matthews Correlation Coefficient (MCC)."""
from ._base import contingency, ratio


def mcc(obs_data, model_data, threshold, dim=None):
    """
    Compute the Matthews Correlation Coefficient (MCC) for a given threshold.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed MCC values. Range -1 to 1;
        NaN where any marginal total is zero.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return ratio(numerator, denominator).clip(-1.0, 1.0)
