"""Quadratic Skill Score (QSS)."""


def qss(obs_data, model_data, reference_forecast, dim=None):
    """
    Compute the Quadratic Skill Score (QSS).
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_forecast (xarray.DataArray): The reference forecast values.
        dim (str, list, or None): Dimension(s) to compute over.
    
    Returns:
        xarray.DataArray: The computed QSS values.
    """
    mse_model = ((model_data - obs_data) ** 2).mean(dim=dim)
    mse_ref = ((reference_forecast - obs_data) ** 2).mean(dim=dim)
    return 1 - mse_model / mse_ref
