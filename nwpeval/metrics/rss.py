"""Relative Skill Score (RSS)."""
import numpy as np


def rss(obs_data, model_data, reference_skill, dim=None):
    """
    Compute the Relative Skill Score (RSS).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_skill (xarray.DataArray): The reference skill values.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed RSS values.
    """
    model_skill = 1 - np.abs(model_data - obs_data) / obs_data
    return (model_skill - reference_skill) / (1 - reference_skill)
