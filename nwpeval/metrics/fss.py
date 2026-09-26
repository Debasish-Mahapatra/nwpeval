"""Fractions Skill Score (FSS)."""
from ._base import binarize, paired, ratio

# Pairs of names taken as the horizontal dimensions when spatial_dims is None,
# in order of preference.
SPATIAL_NAMES = [
    ("lat", "lon"),
    ("latitude", "longitude"),
    ("y", "x"),
    ("rlat", "rlon"),
    ("south_north", "west_east"),
]


def _box_sum(data, dims, size):
    """Sum over a centred ``size``-wide box, truncated at the domain edge.

    A box sum is separable, so it is done one dimension at a time.
    """
    for d in dims:
        data = data.rolling({d: size}, center=True, min_periods=1).sum()
    return data


def fss(obs_data, model_data, threshold, neighborhood_size, spatial_dims=None, reduction_dim=None):
    """
    Compute the Fractions Skill Score (FSS) for a given threshold and neighborhood size.

    FSS = 1 - sum((O - M)^2) / sum(O^2 + M^2)   (Roberts and Lean, 2008)

    where O and M are the fractions of event points (``value >= threshold``)
    in the neighbourhood centred on each point.

    Missing data: a point that is NaN in either input is unknown, not a
    non-event. Fractions are taken over the valid neighbours only, and only
    valid points are scored. The same rule applies at the domain edge, so
    every valid point is scored at every neighbourhood size. To verify inside
    a footprint (e.g. radar coverage), set both inputs to NaN outside it.

    The sums run over every scored point and every dimension in
    ``reduction_dim`` (all dimensions if None), so the default is the
    aggregate FSS. Do not average FSS values computed per time step: that
    weights dry and wet times equally and is not the aggregate score.

    Args:
        obs_data (xarray.DataArray): The observed data.
        model_data (xarray.DataArray): The modeled data.
        threshold (float): The threshold value for binary classification.
        neighborhood_size (int): The width of the neighbourhood window, in
            grid points (odd values give a window centred on the point).
        spatial_dims (str, list, or None): The spatial dimension(s) for the
            neighbourhood window. If None, the first pair found among
            lat/lon, latitude/longitude, y/x, rlat/rlon and
            south_north/west_east is used, and data with at most two
            dimensions uses all of them. Otherwise a ValueError is raised:
            pass the names, e.g. ``spatial_dims=['lat', 'lon']``.
        reduction_dim (str, list, or None): The dimension(s) along which to reduce.

    Returns:
        xarray.DataArray: The computed FSS values. NaN where no event was
        observed or forecast (the reference MSE is zero).
    """
    obs_data, model_data = paired(obs_data, model_data)

    size = int(neighborhood_size)
    if size < 1:
        raise ValueError(f"neighborhood_size must be a positive integer, got {neighborhood_size}")

    # Determine spatial dimensions for rolling. Never guess from the order of
    # the dimensions: with (latitude, longitude, time) that would smooth in time.
    if spatial_dims is None:
        dims = list(obs_data.dims)
        spatial_dims = next((list(pair) for pair in SPATIAL_NAMES if set(pair) <= set(dims)), None)
        if spatial_dims is None:
            if len(dims) > 2:
                raise ValueError(
                    f"Cannot tell which of the dimensions {dims} are spatial. "
                    "Pass them with spatial_dims, e.g. spatial_dims=['lat', 'lon']."
                )
            spatial_dims = dims

    if isinstance(spatial_dims, str):
        spatial_dims = [spatial_dims]

    rolling_dims = [d for d in spatial_dims if d in obs_data.dims]
    if not rolling_dims:
        raise ValueError(f"None of the spatial dimensions {spatial_dims} found in data dimensions {list(obs_data.dims)}")

    valid = obs_data.notnull()
    count = _box_sum(valid.astype(float), rolling_dims, size)
    count = count.where(valid & (count > 0))

    # Event fractions among the valid neighbours of each valid point
    obs_fractions = _box_sum(binarize(obs_data, threshold).fillna(0.0), rolling_dims, size) / count
    model_fractions = _box_sum(binarize(model_data, threshold).fillna(0.0), rolling_dims, size) / count

    mse = ((obs_fractions - model_fractions) ** 2).mean(dim=reduction_dim)
    # Reference MSE of the no-overlap forecast: mean(O^2) + mean(M^2)
    mse_ref = (obs_fractions ** 2 + model_fractions ** 2).mean(dim=reduction_dim)

    return 1 - ratio(mse, mse_ref)
