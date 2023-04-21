"""Here we test access to the NASA LPDAAC data server.

NOTE: to see print outputs one must use the $pytest -rx$ command.
"""
import xarray_data_accessor
from xarray_data_accessor.data_accessors.nasa_from_LPDAAC import AuthorizationDict
import os
import xarray as xr
from typing import List


BBOX_COORDS = [
    (14.96706, -90.01794),  # min lat, min lon
    (15.06368, -90.01794),  # max lat, min lon
    (14.96706, -89.93884),  # min lat, max lon
    (15.06368, -89.93884),  # max lat, max lon
]

DATASET_NAMES = [
    'NASADEM_NC',  # NetCDF format
    # 'NASADEM_SC',  # RAW format
    # 'GLanCE30',  # GeoTIFF format
]

# DEFINE FUNCTIONS TO ACCESS DIFFERENT DATA ####################################


def get_nasadem_nc(
    earthdata_auth_dict: AuthorizationDict,
    bbox_coords: List[float] = BBOX_COORDS,
) -> xr.Dataset:
    """
    Gets NASADEM_NC data (NetCDF format) from the LPDAAC DataPool.
    """
    return xarray_data_accessor.get_xarray_dataset(
        data_accessor_name='NASA_LPDAAC_Accessor',
        dataset_name='NASADEM_NC',
        variables='DEM',
        start_time='2019-01-30',
        end_time='2019-02-02',
        coordinates=bbox_coords,
        kwargs={'authorization': earthdata_auth_dict},
    )


# RUN TESTS ####################################################################

def test_bounding_box() -> None:
    """Test the bounding box from a list of coordinate tuples."""
    # get the bounding box dictionary
    bbox = xarray_data_accessor.get_bounding_box(
        coords=BBOX_COORDS,
    )

    # assert it is as expected
    assert bbox == {
        'west': -90.01794,
        'south': 14.96706,
        'east': -89.93884,
        'north': 15.06368,
    }


def test_dem() -> None:
    # check that the user has added their EarthData credentials to env variables
    try:
        EARTH_DATA_AUTH_DICT: AuthorizationDict = {
            'username': os.environ['EARTHDATA_USERNAME'],
            'password': os.environ['EARTHDATA_PASSWORD'],
        }
    except KeyError as e:
        raise KeyError(
            f'{e} ---- You must add your EarthData credentials to your environment '
            f'to run this test! ---- Ex: EARTHDATA_USERNAME="my_username", '
            f'EARTHDATA_PASSWORD="my_password".'
        ) from e

    dem = get_nasadem_nc(
        earthdata_auth_dict=EARTH_DATA_AUTH_DICT,
    )

    # make assertions about the dataset
    assert isinstance(dem, xr.Dataset)
    assert dem.attrs['dataset_name'] == 'NASADEM_NC'
    assert dem.attrs['institution'] == 'NASA/USGS LP DAAC'
    assert dem.attrs['time_step'] is None

    # check dimensions
    for dim in ['lon', 'lat']:
        assert dim in dem.dims
    assert 'time' not in dem.dims

    # check longitude dimensions
    assert dem.attrs['x_dim'] == 'lon'
    assert len(dem.lon) == 286
    assert dem.lon.dtype == 'float32'
    assert dem.lon[0].item() == -90.01805555555555
    assert dem.lon[-1].item() == -89.93888888888888

    # check latitude dimension
    assert dem.attrs['y_dim'] == 'lat'
    assert len(dem.lat) == 350
    assert dem.lat.dtype == 'float32'
    assert dem.lat[0].item() == 14.966944444444445
    assert dem.lat[-1].item() == 15.06361111111111

    # check the variable
    assert 'NASADEM_HGT' in dem.data_vars
    assert dem['NASADEM_HGT'].dtype == 'float32'

    # check the spatial reference (note WGS 84 corresponds to EPSG:4326)
    assert dem.attrs['EPSG'] == 4326
    assert dem.spatial_ref.attrs['geographic_crs_name'] == 'WGS 84'
