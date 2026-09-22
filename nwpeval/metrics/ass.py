"""Absolute Skill Score (ASS)."""
import numpy as np
import xarray as xr
from ._base import paired, ratio


def ass(obs_data, model_data, reference_error, dim=None):
    """
    Compute the Absolute Skill Score (ASS).

    ASS = 1 - mean(|model - obs|) / mean(reference_error)

    `reference_error` may either be a scalar reference MAE or an array of
    per-element absolute errors from a reference forecast; in the latter case
    its mean over `dim` is used.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        reference_error (xarray.DataArray or float): The reference absolute
            error (per-element) or its aggregate.
        dim (str, list, or None): Dimension(s) to compute over.

    Returns:
        xarray.DataArray: The computed ASS values.
    """
    obs_data, model_data, reference_error = paired(obs_data, model_data, reference_error)
    abs_error = np.abs(model_data - obs_data).mean(dim=dim)

    if isinstance(reference_error, xr.DataArray) and reference_error.ndim > 0:
        dims = [dim] if isinstance(dim, str) else dim
        reduce = [d for d in (reference_error.dims if dims is None else dims)
                  if d in reference_error.dims]
        ref = reference_error.mean(dim=reduce) if reduce else reference_error
    else:
        ref = xr.DataArray(reference_error)

    return 1 - ratio(abs_error, ref)
