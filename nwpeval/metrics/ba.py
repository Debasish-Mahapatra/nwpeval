"""Balanced Accuracy (BA)."""
from ._base import contingency, ratio


def ba(obs_data, model_data, threshold, dim=None):
    """
    Compute the Balanced Accuracy (BA) for a given threshold.

    BA = 0.5 * (TPR + TNR) where TPR = TP/(TP+FN) and TNR = TN/(TN+FP).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed BA values. NaN where no event
        or no non-event was observed.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return 0.5 * (ratio(tp, tp + fn) + ratio(tn, tn + fp))
