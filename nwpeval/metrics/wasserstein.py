"""Wasserstein Distance (W1)."""
import numpy as np
import xarray as xr
from ._base import paired


def wasserstein(obs_data, model_data, dim=None):
    """
    Compute the 1-Wasserstein distance between two empirical distributions.

    For samples of equal size, the 1-Wasserstein distance equals the
    mean absolute difference of the order statistics:
        W1 = mean(|sort(obs) - sort(model)|).

    It compares the spread of values only: where the values are does not
    matter. The other distributional metrics (e.g. :func:`hellinger`)
    compare where the mass is instead.

    Points missing (NaN) in either input are dropped from both samples.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) over which to sort and average.
            If None, the distance is computed over the entire flattened array.

    Returns:
        xarray.DataArray: The computed Wasserstein distance values.
    """
    obs_data, model_data = paired(obs_data, model_data)
    # Same layout for both, so the raw arrays below line up element by element.
    model_data = model_data.transpose(*obs_data.dims)

    if dim is None:
        # Missing pairs are NaN in both; NaN sorts last, so they pair up and drop out.
        diff = np.abs(np.sort(obs_data.values.ravel()) - np.sort(model_data.values.ravel()))
        valid = np.isfinite(diff)
        return xr.DataArray(diff[valid].mean() if valid.any() else np.nan)

    dims = [dim] if isinstance(dim, str) else list(dim)

    obs_axes = tuple(obs_data.get_axis_num(d) for d in dims)
    model_axes = tuple(model_data.get_axis_num(d) for d in dims)

    obs_sorted_vals = np.sort(obs_data.values, axis=obs_axes[0]) if len(obs_axes) == 1 else _sort_multi_axis(obs_data.values, obs_axes)
    model_sorted_vals = np.sort(model_data.values, axis=model_axes[0]) if len(model_axes) == 1 else _sort_multi_axis(model_data.values, model_axes)

    diff = np.abs(obs_sorted_vals - model_sorted_vals)
    sorted_da = xr.DataArray(
        diff,
        dims=obs_data.dims,
        coords={k: v for k, v in obs_data.coords.items() if not set(v.dims) & set(dims)},
    )
    return sorted_da.mean(dim=dim)


def _sort_multi_axis(arr, axes):
    """Sort an ndarray along multiple axes by flattening them, sorting, and reshaping."""
    other_axes = [a for a in range(arr.ndim) if a not in axes]
    perm = other_axes + list(axes)
    transposed = np.transpose(arr, perm)
    flat_shape = transposed.shape[: len(other_axes)] + (-1,)
    reshaped = transposed.reshape(flat_shape)
    sorted_reshaped = np.sort(reshaped, axis=-1)
    sorted_back = sorted_reshaped.reshape(transposed.shape)
    inverse_perm = np.argsort(perm)
    return np.transpose(sorted_back, inverse_perm)
