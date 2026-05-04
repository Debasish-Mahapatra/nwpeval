"""Gilbert Skill Score (GSS).

Alias of :func:`ets`. The Gilbert Skill Score and the Equitable Threat
Score are the same metric under different historical names.
"""
from .ets import ets


def gss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Gilbert Skill Score (GSS) for a given threshold.

    GSS is mathematically identical to the Equitable Threat Score (ETS).
    This function is kept as an alias for backward compatibility.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed GSS values (same as ETS).
    """
    return ets(obs_data, model_data, threshold, dim=dim)
