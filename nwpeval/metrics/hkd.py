"""Hanssen-Kuipers Discriminant (HKD).

Alias of :func:`pss`. The Hanssen-Kuipers Discriminant and the Peirce
Skill Score are the same metric (POD - POFD) under different historical
names.
"""
from .pss import pss


def hkd(obs_data, model_data, threshold, dim=None):
    """
    Compute the Hanssen-Kuipers Discriminant (HKD) for a given threshold.

    HKD is mathematically identical to the Peirce Skill Score (PSS).
    This function is kept as an alias for backward compatibility.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed HKD values (same as PSS).
    """
    return pss(obs_data, model_data, threshold, dim=dim)
