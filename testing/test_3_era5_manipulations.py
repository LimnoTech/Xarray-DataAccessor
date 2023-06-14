"""Tests if data manipulation functions are working as expected."""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import xarray_data_accessor
from xarray_data_accessor import (
    ConvertToTable,
)
import pytest
from rasterio.enums import Resampling
from typing import (
    List,
    get_args,
)

# define fixtures


@pytest.fixture
def test_dir() -> Path:
    """Gets the test data directory."""
    TEST_DIR = Path.cwd() / 'testing/test_data'
    if not TEST_DIR.exists():
        TEST_DIR = Path.cwd() / 'test_data'
    return TEST_DIR


@pytest.fixture
def test_dataset(test_dir) -> xr.Dataset:
    """Gets the test dataset."""
    # get test netcdf file into a xr.Dataset
    test_netcdf = test_dir / 'cds_era5_dataset.nc'
    ds = xr.open_dataset(test_netcdf)
    ds = ds.rio.write_crs(ds.attrs['EPSG'])
    return ds


@pytest.fixture
def spatial_resample_methods() -> List[str]:
    """Gets resampling method names from the Resampling enum."""
    names = []
    for member in Resampling:
        # ignore gauss, used for custom methods
        if member.name != 'gauss':
            names.append(member.name)
    return names


@pytest.fixture
def temporal_resample_methods() -> List[str]:
    """Gets resampling method names xarray_data_accessor.shared_types."""
    names = []
    for member in get_args(xr.core.types.Interp1dOptions):
        names.append(member)
    for member in get_args(xarray_data_accessor.shared_types.AggregationMethods):
        names.append(member)
    print(names)
    return names


def test_timezone_subset(test_dataset) -> None:
    """Tests if we can subset by a different timezone."""
    subset = xarray_data_accessor.subset_time_by_timezone(
        xarray_dataset=test_dataset,
        timezone='US/Eastern',
        end_time='2019-02-01T10:00:00.000000000',
    )
    assert len(subset.time) != len(test_dataset.time)
    assert len(subset.time) == 64
    assert subset.time[-1].item() == 1549033200000000000
    del subset


def test_spatial_resample(test_dataset, spatial_resample_methods) -> None:
    """Tests if spatial resampling is working as expected."""
    # assert xy dimension sizes
    assert len(test_dataset.longitude) == 19
    assert len(test_dataset.latitude) == 7

    # resample x2 with each method
    for name in spatial_resample_methods:
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


def test_temporal_resample(test_dataset, temporal_resample_methods) -> None:
    """Tests temporal resampling."""

    assert len(test_dataset.time) == 73

    # resample with each method
    for name in temporal_resample_methods:

        if name == 'polynomial':
            continue
        resampled_dataset = xarray_data_accessor.temporal_resample(
            xarray_dataset=test_dataset,
            resample_frequency='T',
            resample_method=name,
        )
        assert len(resampled_dataset.time) == 4321

    # test applying a custom function
    resampled_dataset = xarray_data_accessor.temporal_resample(
        xarray_dataset=test_dataset,
        resample_frequency='T',
        custom_resample_method=np.mean,
    )
    assert len(resampled_dataset.time) == 4321
    del resampled_dataset


def test_to_table(test_dataset, test_dir) -> None:
    """Tests if to_table_coords_list is working as expected."""
    # associate file types to pandas file reader functions
    suffix_dict = {
        '.parquet': pd.read_parquet,
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
    }
    for suffix, func in suffix_dict.items():
        tables_dict = ConvertToTable.points_to_tables(
            xarray_dataset=test_dataset,
            variables=None,
            # lon/lat coords, not perfectly aligned with grid
            coords=[
                (-82.98, 41.63),
                (-79.43, 42.88),
                (-83.23, 41.85),
            ],
            save_table_dir=test_dir,
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
