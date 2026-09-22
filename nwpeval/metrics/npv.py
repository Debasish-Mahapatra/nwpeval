"""Negative Predictive Value (NPV)."""
from ._base import contingency, ratio


def npv(obs_data, model_data, threshold, dim=None):
    """
    Compute the Negative Predictive Value (NPV) for a given threshold.

    NPV = TN / (TN + FN).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed NPV values. NaN where no
        non-event was forecast.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(tn, tn + fn)
