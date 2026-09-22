"""Probability of Detection (POD)."""
from ._base import contingency, ratio


def pod(obs_data, model_data, threshold, dim=None):
    """
    Compute the Probability of Detection (POD) for a given threshold.

    POD = TP / (TP + FN).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed POD values. NaN where no event
        was observed.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(tp, tp + fn)
