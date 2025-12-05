"""Brier Skill Score (BSS)."""


def bss(obs_data, model_data, threshold, dim=None):
    """
    Compute the Brier Skill Score (BSS) for a given threshold.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed BSS values.
    """
    obs_binary = (obs_data >= threshold).astype(float)
    
    # Brier score for model
    bs_model = ((model_data - obs_binary) ** 2).mean(dim=dim)
    
    # Brier score for climatology
    base_rate = obs_binary.mean(dim=dim)
    bs_climo = ((base_rate - obs_binary) ** 2).mean(dim=dim)
    
    return 1 - bs_model / bs_climo
