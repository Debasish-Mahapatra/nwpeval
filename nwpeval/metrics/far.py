"""False Alarm Ratio (FAR)."""
from ._base import contingency, ratio


def far(obs_data, model_data, threshold, dim=None):
    """
    Compute the False Alarm Ratio (FAR) for a given threshold.

    FAR = FP / (TP + FP).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed FAR values. NaN where no event
        was forecast.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(fp, tp + fp)
