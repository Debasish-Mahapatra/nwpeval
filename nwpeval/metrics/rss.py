"""Relative Skill Score (RSS)."""
import numpy as np
import xarray as xr


def rss(obs_data, model_data, reference_skill, dim=None):
    """
    Compute the Relative Skill Score (RSS).

    Model skill is defined as 1 - MAPE, where MAPE is the mean absolute
    percentage error of the model relative to observations:
        model_skill = 1 - mean(|model - obs| / |obs|).
    RSS = (model_skill - reference_skill) / (1 - reference_skill).

    Args:
        obs_data (xarray.DataArray): The observed data (must be non-zero).
        model_data (xarray.DataArray): The modeled data.
        reference_skill (xarray.DataArray or float): The reference skill score.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed RSS values.
    """
    obs_safe = xr.where(obs_data == 0, np.nan, obs_data)
    rel_err = (np.abs(model_data - obs_data) / np.abs(obs_safe)).mean(dim=dim)
    model_skill = 1 - rel_err
    return xr.where(reference_skill == 1, np.nan, (model_skill - reference_skill) / (1 - reference_skill))
