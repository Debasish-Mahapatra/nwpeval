"""Wasserstein Distance (W1)."""
import numpy as np
import xarray as xr


def wasserstein(obs_data, model_data, dim=None):
    """
    Compute the 1-Wasserstein distance between two empirical distributions.

    For samples of equal size, the 1-Wasserstein distance equals the
    mean absolute difference of the order statistics:
        W1 = mean(|sort(obs) - sort(model)|).

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        dim (str, list, or None): Dimension(s) over which to sort and average.
            If None, the distance is computed over the entire flattened array.

    Returns:
        xarray.DataArray: The computed Wasserstein distance values.
    """
    if dim is None:
        obs_flat = np.asarray(obs_data.values).ravel()
        model_flat = np.asarray(model_data.values).ravel()
        if obs_flat.size != model_flat.size:
            raise ValueError(
                "wasserstein requires obs and model to have the same number of samples."
            )
        return xr.DataArray(
            np.mean(np.abs(np.sort(obs_flat) - np.sort(model_flat)))
        )

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
