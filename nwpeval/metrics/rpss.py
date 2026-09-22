"""Ranked Probability Skill Score (RPSS)."""
from ._base import binarize, paired, ratio


def rpss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Ranked Probability Skill Score (RPSS) for a given threshold.

    Note: This is a simplified binary version. For multi-category probabilistic
    forecasts, a more complex implementation is needed. Both inputs are turned
    into events (``value >= threshold``), so for two categories the RPS is the
    Brier score of the deterministic forecast, against the sample climatology.

    Points missing (NaN) in either input are excluded.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed RPSS values. NaN where the observed
        event frequency is 0 or 1.
    """
    obs_data, model_data = paired(obs_data, model_data)
    obs_binary = binarize(obs_data, threshold)
    model_binary = binarize(model_data, threshold)

    rps_model = ((model_binary - obs_binary) ** 2).mean(dim=dim)
    base_rate = obs_binary.mean(dim=dim)
    rps_climo = ((base_rate - obs_binary) ** 2).mean(dim=dim)

    return 1 - ratio(rps_model, rps_climo)
