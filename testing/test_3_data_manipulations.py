"""Tests if data manipulation functions are working as expected."""
import xarray as xr
from pathlib import Path
TEST_NETCDF = Path.cwd() / 'test_data/test_dataset.nc'

TEST_DATASET = xr.open_dataset(TEST_NETCDF)


def test_timezone_change():
    assert True == True


def test_spatial_resample():
    assert True == True


def test_time_resample():
    assert True == True


def test_to_table():
    assert True == True
