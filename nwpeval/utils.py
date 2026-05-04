import xarray as xr
import pandas as pd


def load_data(data):
    """
    Load data from various formats.

    Supported inputs:
        - xarray.DataArray (returned as-is)
        - pandas.DataFrame (wrapped as DataArray)
        - file path (str) with one of these extensions:
            * .nc                  NetCDF
            * .grib, .grb          GRIB (requires `cfgrib`)
            * .h5, .hdf5, .hdf     HDF5 (requires `h5netcdf`)
            * .csv                 CSV
            * .xlsx, .xls          Excel

    Optional dependencies (cfgrib, h5netcdf) are imported lazily and only
    when the corresponding format is requested.

    Args:
        data (xarray.DataArray, pandas.DataFrame, or str): The input data
            or file path.

    Returns:
        xarray.DataArray: The loaded data as an xarray DataArray.
    """
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, pd.DataFrame):
        return xr.DataArray(data)
    if isinstance(data, str):
        lower = data.lower()
        if lower.endswith('.nc'):
            return xr.open_dataarray(data)
        if lower.endswith('.grib') or lower.endswith('.grb'):
            try:
                import cfgrib  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Reading GRIB files requires the 'cfgrib' package. "
                    "Install it with `pip install cfgrib`."
                ) from exc
            return xr.open_dataarray(data, engine='cfgrib')
        if lower.endswith('.h5') or lower.endswith('.hdf5') or lower.endswith('.hdf'):
            try:
                import h5netcdf  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Reading HDF5 files requires the 'h5netcdf' package. "
                    "Install it with `pip install h5netcdf`."
                ) from exc
            return xr.open_dataarray(data, engine='h5netcdf')
        if lower.endswith('.csv'):
            return xr.DataArray(pd.read_csv(data))
        if lower.endswith('.xlsx') or lower.endswith('.xls'):
            return xr.DataArray(pd.read_excel(data))
        raise ValueError(f"Unsupported file format: {data}")
    raise ValueError(f"Unsupported data type: {type(data)}")
