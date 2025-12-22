"""Fractions Skill Score (FSS)."""
import numpy as np


def fss(obs_data, model_data, threshold, neighborhood_size, spatial_dims=None, reduction_dim=None):
    """
    Compute the Fractions Skill Score (FSS) for a given threshold and neighborhood size.
    
    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        neighborhood_size (int): The size of the neighborhood window.
        spatial_dims (str, list, or None): The spatial dimension(s) for rolling window.
                                           If None, auto-detects ['lat', 'lon'] or ['x', 'y'].
        reduction_dim (str, list, or None): The dimension(s) along which to reduce.
    
    Returns:
        xarray.DataArray: The computed FSS values.
    """
    # Convert data to binary based on the threshold
    obs_binary = (obs_data >= threshold).astype(float)
    model_binary = (model_data >= threshold).astype(float)
    
    # Determine spatial dimensions for rolling
    if spatial_dims is None:
        dims = list(obs_data.dims)
        if 'lat' in dims and 'lon' in dims:
            spatial_dims = ['lat', 'lon']
        elif 'x' in dims and 'y' in dims:
            spatial_dims = ['x', 'y']
        elif len(dims) >= 2:
            spatial_dims = dims[-2:]
        else:
            spatial_dims = dims
    
    if isinstance(spatial_dims, str):
        spatial_dims = [spatial_dims]
    
    # Create rolling window dict for all spatial dimensions
    rolling_dict = {d: neighborhood_size for d in spatial_dims if d in obs_data.dims}
    
    if not rolling_dict:
        raise ValueError(f"None of the spatial dimensions {spatial_dims} found in data dimensions {list(obs_data.dims)}")
    
    # Compute the fractions within each neighborhood
    obs_fractions = obs_binary.rolling(rolling_dict, center=True).mean()
    model_fractions = model_binary.rolling(rolling_dict, center=True).mean()
    
    # Calculate the mean squared error (MSE) of the fractions
    mse = ((obs_fractions - model_fractions) ** 2).mean(dim=reduction_dim)
    
    # Calculate the reference MSE (worst case: no skill)
    # MSE_ref = mean(O^2) + mean(M^2) for the "no-skill" forecast
    mse_ref = (obs_fractions ** 2).mean(dim=reduction_dim) + (model_fractions ** 2).mean(dim=reduction_dim)
    
    # Calculate the FSS
    return 1 - mse / mse_ref
