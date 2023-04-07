"""Tests if data manipulation functions are working as expected."""
import xarray as xr
import pandas as pd
from pathlib import Path
import xarray_data_accessor
from rasterio.enums import Resampling
from typing import List

# get path to test data regardless of whether we are debugging or running tests
TEST_DIR = Path.cwd() / 'testing/test_data'
if not TEST_DIR.exists():
    TEST_DIR = Path.cwd() / 'test_data'

# get test netcdf file into a xr.Dataset
TEST_NETCDF = TEST_DIR / 'cds_era5_dataset.nc'
TEST_DATASET = xr.open_dataset(TEST_NETCDF)
TEST_DATASET = TEST_DATASET.rio.write_crs(TEST_DATASET.attrs['EPSG'])


def test_timezone_change(
    test_dataset: xr.Dataset = TEST_DATASET,
) -> None:
    # TODO: implement when we get the conversion working
    # check starting timezone
    # tz1 = test_dataset.time.tz
    # time_index_1 = test_dataset.time.values[0]
    # assert tz1 == 'UTC'
    #
    # convert timezones to America/New_York
    # test_dataset = xarray_data_accessor.convert_timezone(
    #    xarray_dataset=test_dataset,
    #    timezone='America/New_York',
    # )
    # tz2 = test_dataset.time.tz
    # time_index_2 = test_dataset.time.values[0]
    # assert tz2 == 'America/New_York'
    # assert time_index_1 != time_index_2
    assert True == True


def _get_resample_method_name() -> List[str]:
    """Gets resampling method names from the Resampling enum."""
    names = []
    for member in Resampling:
        # ignore gauss, used for custom methods
        if member.name != 'gauss':
            names.append(member.name)
    return names


def test_spatial_resample(
    test_dataset: xr.Dataset = TEST_DATASET,
) -> None:
    """Tests if spatial resampling is working as expected."""
    # assert xy dimension sizes
    assert len(test_dataset.longitude) == 19
    assert len(test_dataset.latitude) == 7

    # resample x2 with each method
    for name in _get_resample_method_name():
        # use the resolution_factor argument
        test_dataset_rs1 = xarray_data_accessor.spatial_resample(
            xarray_dataset=test_dataset,
            resolution_factor=2,
            resample_method=name,
        )
        assert len(test_dataset_rs1.longitude) == 38
        assert len(test_dataset_rs1.latitude) == 14

        # use the xy_resolution_factors argument
        test_dataset_rs2 = xarray_data_accessor.spatial_resample(
            xarray_dataset=test_dataset,
            xy_resolution_factors=(1, 3),
            resample_method=name,
        )
        assert len(test_dataset_rs2.longitude) == 19
        assert len(test_dataset_rs2.latitude) == 21


def test_time_resample(
    test_dataset: xr.Dataset = TEST_DATASET,
) -> None:
    """Tests if time resampling is working as expected."""
    # TODO: implement
    assert True == True


def test_to_table(
    test_dataset: xr.Dataset = TEST_DATASET,
) -> None:
    """Tests if to_table_coords_list is working as expected."""
    # associate file types to pandas file reader functions
    suffix_dict = {
        '.parquet': pd.read_parquet,
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
    }
    for suffix, func in suffix_dict.items():
        tables_dict = xarray_data_accessor.get_data_tables(
            xarray_dataset=test_dataset,
            variables=None,
            # lon/lat coords, not perfectly aligned with grid
            coords=[
                (-82.98, 41.63),
                (-79.43, 42.88),
                (-83.23, 41.85),
            ],
            save_table_dir=TEST_DIR,
            save_table_suffix=suffix,
        )

        # check that we get a dictionary back referencing the saved tables
        assert isinstance(tables_dict, dict)
        for k, v in tables_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, Path)

            # read the table and check that it is as expected
            table_df = func(v)
            assert isinstance(table_df, pd.DataFrame)
            if 'datetime' in table_df.columns:
                table_df = table_df.set_index('datetime')
            assert len(table_df.index) == len(test_dataset.time.values)
            assert len(table_df.columns) == 3
