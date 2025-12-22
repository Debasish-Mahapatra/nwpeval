"""Ranked Probability Skill Score (RPSS)."""


def rpss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Ranked Probability Skill Score (RPSS) for a given threshold.
    
    Note: This is a simplified binary version. For multi-category probabilistic
    forecasts, a more complex implementation is needed.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed RPSS values.
    """
    obs_binary = (obs_data >= threshold).astype(float)
    model_binary = (model_data >= threshold).astype(float)
    
    # For binary case, RPS reduces to Brier Score
    rps_model = ((model_binary - obs_binary) ** 2).mean(dim=dim)
    
    base_rate = obs_binary.mean(dim=dim)
    rps_climo = ((base_rate - obs_binary) ** 2).mean(dim=dim)
    
    return 1 - rps_model / rps_climo
