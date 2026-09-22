"""F1 Score."""
from ._base import contingency, ratio


def f1(obs_data, model_data, threshold, dim=None):
    """
    Compute the F1 Score for a given threshold.

    F1 = 2 TP / (2 TP + FP + FN), the harmonic mean of precision and recall.

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed F1 values. NaN where no event
        was observed or forecast.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(2 * tp, 2 * tp + fp + fn)
