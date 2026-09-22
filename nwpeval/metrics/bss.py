"""Brier Skill Score (BSS)."""
from ._base import binarize, paired, ratio


def bss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Brier Skill Score (BSS) for a given threshold.

    BSS = 1 - BS_model / BS_climatology, where the observations are turned
    into events (``obs >= threshold``) and ``model_data`` is the forecast
    probability of that event, in [0, 1]. The reference is the sample
    climatology (the observed event frequency over ``dim``).

    Points missing (NaN) in either input are excluded.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The forecast event probability (0-1).
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed BSS values. NaN where the observed
        event frequency is 0 or 1 (the climatological Brier score is zero).
    """
    obs_data, model_data = paired(obs_data, model_data)
    obs_binary = binarize(obs_data, threshold)

    bs_model = ((model_data - obs_binary) ** 2).mean(dim=dim)
    base_rate = obs_binary.mean(dim=dim)
    bs_climo = ((base_rate - obs_binary) ** 2).mean(dim=dim)

    return 1 - ratio(bs_model, bs_climo)
