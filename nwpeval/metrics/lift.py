"""Lift metric."""
from ._base import contingency, ratio


def lift(obs_data, model_data, threshold, dim=None):
    """
    Compute the Lift for a given threshold.

    Lift = precision / base_rate = [TP / (TP+FP)] / [(TP+FN) / N]
    where N = TP + FP + FN + TN.

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed Lift values. NaN where no event
        was forecast or observed.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    n = tp + fp + fn + tn
    return ratio(ratio(tp, tp + fp), ratio(tp + fn, n))
