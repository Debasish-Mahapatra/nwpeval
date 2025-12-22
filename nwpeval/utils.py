import pygrib
import xarray as xr
import pandas as pd

def load_data(data):
    """
    Load data from various formats (xarray DataArray, pandas DataFrame, NetCDF, GRIB, CSV, or Excel).
    
    Args:
        data (xarray.DataArray, pandas.DataFrame, or str): The input data, either as an xarray DataArray, pandas DataFrame, or file path (NetCDF, GRIB, CSV, or Excel).
    
    Returns:
        xarray.DataArray: The loaded data as an xarray DataArray.
    """
    if isinstance(data, xr.DataArray):
        return data
    elif isinstance(data, pd.DataFrame):
        return xr.DataArray(data)
    elif isinstance(data, str):
        if data.endswith('.nc'):
            return xr.open_dataarray(data)
        elif data.endswith('.grib') or data.endswith('.grb'):
            return xr.open_dataarray(data, engine='cfgrib')
        elif data.endswith('.csv'):
            df = pd.read_csv(data)
            return xr.DataArray(df)
        elif data.endswith('.xlsx') or data.endswith('.xls'):
            df = pd.read_excel(data)
            return xr.DataArray(df)
        else:
            raise ValueError(f"Unsupported file format: {data}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")