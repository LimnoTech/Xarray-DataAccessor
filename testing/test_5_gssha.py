"""Tests conversion to GSSHA format."""
import xarray_data_accessor as xda
import xarray as xr
import pytest
from pathlib import Path


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


def test_factory() -> None:
    """Make sure the function was correctly registered."""
    assert 'ConvertToGSSHA' in xda.DataConversionFactory.get_converter_classes().keys()
    gssha_obj = xda.DataConversionFactory.get_converter_classes()[
        'ConvertToGSSHA'
    ]
    assert isinstance(gssha_obj, object)
    assert issubclass(
        gssha_obj,
        xda.data_converters.base.DataConverterBase,
    )
    for func_name in gssha_obj.get_conversion_functions().keys():
        assert func_name in dir(xda.DataConversionFunctions)


def test_precipitation_input(test_dataset) -> None:
    out_path = xda.DataConversionFunctions.make_gssha_precipitation_input(
        test_dataset,
        precipitation_variable='2m_temperature',
        output_epsg=26915,
    )

    assert out_path.exists()
    assert out_path.suffix == '.gag'
    out_path.unlink()


def test_to_grass_ascii(test_dataset) -> None:

    # test with correct HMET variable
    out_list = xda.DataConversionFunctions.make_gssha_grass_ascii(
        test_dataset,
        variable='2m_temperature',
        hmet_variable='Dry Bulb Temperature',
        start_time=test_dataset.time.values[0],
        end_time=test_dataset.time.values[1],
    )
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    for file in out_list:
        assert file.exists()
        assert file.suffix == '.asc'
        file.unlink()

    # test with incorrect HMET variable


def test_hmet_wes_ascii(test_dataset) -> None:

    out_path = xda.DataConversionFunctions.make_gssha_hmet_wes(
        test_dataset,
        variable_to_hmet={'2m_temperature': 'Dry Bulb Temperature'},
        start_time=test_dataset.time.values[0],
        end_time=test_dataset.time.values[1],
        file_suffix='.test',
    )
    assert out_path.exists()
    assert out_path.suffix == '.test'
    assert out_path.name.replace('.test', '') == 'hmet_wes'
    out_path.unlink()
