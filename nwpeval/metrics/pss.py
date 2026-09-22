"""Peirce Skill Score (PSS)."""
from ._base import contingency, ratio


def pss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Peirce Skill Score (PSS) for a given threshold.

    PSS = POD - POFD = TP / (TP + FN) - FP / (FP + TN).

    An event is ``value >= threshold``. Points missing (NaN) in either input
    are excluded from the contingency table.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed PSS values. NaN where no event
        was observed or no non-event was observed.
    """
    tn, fp, fn, tp = contingency(obs_data, model_data, threshold, dim)
    return ratio(tp, tp + fn) - ratio(fp, fp + tn)
