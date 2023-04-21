"""Test the ERA5 data retrieval and processing using both CDS and AWS.

NOTE: to see print outputs one must use the $pytest -rx$ command.
"""
import xarray_data_accessor
import time
import xarray as xr
from pathlib import Path
from typing import List

TEST_DATA_DIR = Path.cwd() / 'test_data'
if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR = Path.cwd() / 'testing/test_data'
TEST_SHP = TEST_DATA_DIR / 'LEEM_boundary.shp'

CDS_VARIABLES = [
    '2m_temperature',
    '100m_u_component_of_wind',
]
AWS_VARIABLES = [
    'air_temperature_at_2_metres',
    'eastward_wind_at_100_metres',
]

# LATITUDE AND LONGITUDE TOLERANCE
# NOTE: the exact max/min lat/lon can change from CDS even using the same request
CDS_COORDS_TOLERANCE = 0.001

# DEFINE DATA RETRIEVAL FUNCTIONS ####################################


def cds_retrieval(
    bbox_shapefile: Path = TEST_SHP,
    cds_variables: List[str] = CDS_VARIABLES,
) -> xr.Dataset:
    return xarray_data_accessor.get_xarray_dataset(
        data_accessor_name='CDSDataAccessor',
        dataset_name='reanalysis-era5-single-levels',
        variables=cds_variables,
        start_time='2019-01-30',
        end_time='2019-02-02',
        shapefile=bbox_shapefile,
    )


def aws_retrieval(
    bbox_shapefile: Path = TEST_SHP,
    aws_variables: List[str] = AWS_VARIABLES,
) -> xr.Dataset:
    return xarray_data_accessor.get_xarray_dataset(
        data_accessor_name='AWSDataAccessor',
        dataset_name='reanalysis-era5-single-levels',
        variables=aws_variables,
        start_time='2019-01-30',
        end_time='2019-02-02',
        shapefile=bbox_shapefile,
    )

# RUN TESTS ########################################################


def test_bounding_box() -> None:
    """Test the bounding box."""
    # get the bounding box dictionary
    bbox = xarray_data_accessor.get_bounding_box(
        shapefile=TEST_SHP,
    )

    # assert it is as expected
    assert bbox == {
        'west': -83.47519999999993,
        'south': 41.382849899000156,
        'east': -78.85399999999997,
        'north': 42.90550549900012,
    }


def test_cds_dataset() -> None:
    """Test the CDS dataset."""

    # get the data
    p1 = time.perf_counter()
    cds_era5_dataset = cds_retrieval()
    p2 = time.perf_counter()
    print(f'cds_era5_dataset retrieved in {p2 - p1:0.4f} seconds')

    # make assertions about the dataset
    assert isinstance(cds_era5_dataset, xr.Dataset)
    assert cds_era5_dataset.attrs['dataset_name'] == 'reanalysis-era5-single-levels'
    assert cds_era5_dataset.attrs['institution'] == 'ECMWF'

    # check dimensions
    for dim in ['longitude', 'latitude', 'time']:
        assert dim in cds_era5_dataset.dims

    # check time dimensions
    assert cds_era5_dataset.attrs['time_step'] == 'hourly'
    assert cds_era5_dataset.attrs['time_zone'] == 'UTC'
    assert len(cds_era5_dataset.time) == 73
    assert cds_era5_dataset.time.dtype == 'datetime64[ns]'
    assert cds_era5_dataset.time[0].item() == 1548806400000000000
    assert cds_era5_dataset.time[-1].item() == 1549065600000000000

    # check longitude dimensions
    assert cds_era5_dataset.attrs['x_dim'] == 'longitude'
    assert len(cds_era5_dataset.longitude) == 19
    assert cds_era5_dataset.longitude.dtype == 'float32'
    assert abs(
        cds_era5_dataset.longitude[0].item() - float(-83.47599792480469)
    ) < CDS_COORDS_TOLERANCE
    assert abs(
        cds_era5_dataset.longitude[-1].item() - float(-78.9749984741211)
    ) < CDS_COORDS_TOLERANCE

    # check latitude dimension
    assert cds_era5_dataset.attrs['y_dim'] == 'latitude'
    assert len(cds_era5_dataset.latitude) == 7
    assert cds_era5_dataset.latitude.dtype == 'float32'
    assert abs(
        cds_era5_dataset.latitude[0].item() - float(42.882999420166016)
    ) < CDS_COORDS_TOLERANCE
    assert abs(
        cds_era5_dataset.latitude[-1].item() - float(41.382999420166016)
    ) < CDS_COORDS_TOLERANCE

    # check the variables
    for data_var in CDS_VARIABLES:
        assert data_var in cds_era5_dataset.data_vars
        assert cds_era5_dataset[data_var].dtype == 'float32'

    # check the spatial reference (note WGS 84 corresponds to EPSG:4326)
    assert cds_era5_dataset.attrs['EPSG'] == 4326
    assert cds_era5_dataset.spatial_ref.attrs['geographic_crs_name'] == 'WGS 84'

    # delete temp files
    xarray_data_accessor.delete_temp_files()


def test_aws_dataset() -> None:
    """Test the AWS dataset."""

    # get the data
    p1 = time.perf_counter()
    aws_era5_dataset = aws_retrieval()
    p2 = time.perf_counter()
    print(f'aws_era5_dataset retrieved in {p2 - p1:0.4f} seconds')

    # make assertions about the dataset
    assert isinstance(aws_era5_dataset, xr.Dataset)
    assert aws_era5_dataset.attrs['dataset_name'] == 'reanalysis-era5-single-levels'
    assert aws_era5_dataset.attrs['institution'] == 'ECMWF via Planet OS'

    # check dimensions
    for dim in ['longitude', 'latitude', 'time']:
        assert dim in aws_era5_dataset.dims

    # check time dimensions
    assert aws_era5_dataset.attrs['time_step'] == 'hourly'
    assert aws_era5_dataset.attrs['time_zone'] == 'UTC'
    assert len(aws_era5_dataset.time) == 73
    assert aws_era5_dataset.time.dtype == 'datetime64[ns]'
    assert aws_era5_dataset.time[0].item() == 1548806400000000000
    assert aws_era5_dataset.time[-1].item() == 1549065600000000000

    # check longitude dimensions
    assert aws_era5_dataset.attrs['x_dim'] == 'longitude'
    assert len(aws_era5_dataset.longitude) == 20
    assert aws_era5_dataset.longitude.dtype == 'float32'
    assert aws_era5_dataset.longitude[0].item() == -83.5
    assert aws_era5_dataset.longitude[-1].item() == -78.75

    # check latitude dimension
    assert aws_era5_dataset.attrs['y_dim'] == 'latitude'
    assert len(aws_era5_dataset.latitude) == 7
    assert aws_era5_dataset.latitude.dtype == 'float32'
    assert aws_era5_dataset.latitude[0].item() == 43.0
    assert aws_era5_dataset.latitude[-1].item() == 41.5

    # check the variables
    for data_var in AWS_VARIABLES:
        assert data_var in aws_era5_dataset.data_vars
        assert aws_era5_dataset[data_var].dtype == 'float32'

    # check the spatial reference (note WGS 84 corresponds to EPSG:4326)
    assert aws_era5_dataset.attrs['EPSG'] == 4326
    assert aws_era5_dataset.spatial_ref.attrs['geographic_crs_name'] == 'WGS 84'


if '__main__' == __name__:
    test_bounding_box()
    test_cds_dataset()
    test_aws_dataset()
