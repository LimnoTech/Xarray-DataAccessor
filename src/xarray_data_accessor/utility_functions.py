"""Stores useful utility functions

Example:
* datetime parsing
* timezone conversions
* AOI parsing

"""
import pytz
from datetime import datetime
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import (
    Union,
    List,
)
from xarray_data_accessor.shared_types import (
    TimeInput,
    CoordsTuple,
    TableInput,
    RasterInput,
    ShapefileInput,
    BoundingBoxDict,
)


def get_datetime(input_date: TimeInput) -> datetime:
    """Returns a datetime object from a variety of inputs."""
    if isinstance(input_date, datetime):
        return input_date

    # assume int is a year
    elif isinstance(input_date, int):
        if input_date not in list(range(1950, datetime.now().year + 1)):
            raise ValueError(
                f'integer start/end date input={input_date} is not a valid year.'
            )
        return pd.to_datetime(f'{input_date}-01-01')

    elif isinstance(input_date, str):
        return pd.to_datetime(input_date)
    else:
        raise ValueError(
            f'start/end date input={input_date} is invalid.'
        )


def convert_timezone(
    xarray_dataset: xr.Dataset,
    timezone: str,
) -> xr.Dataset:
    """
    Convert the datetime index of an xarray dataset to a specified timezone.

    :param data: xarray dataset with a time dimension
    :param timezone: string specifying the desired timezone
    :return: xarray dataset with the converted datetime index
    """
    # Get the desired timezone
    tz = pytz.timezone(timezone)

    # Convert the datetime index to the specified timezone
    xarray_dataset['time'] = (
        xarray_dataset['time']
        .to_pandas()
        .tz_localize('UTC')
        .tz_convert(tz)
    )

    return xarray_dataset


def bbox_from_coords(
    coords: Union[CoordsTuple, List[CoordsTuple]],
) -> BoundingBoxDict:
    """Returns a bounding box from a list of coordinates."""
    # TODO: add buffer so the edge isn't the exact coordinate?
    # TODO : make method
    if isinstance(coords, tuple):
        coords = [coords]
    north = coords[0][0]
    south = coords[0][0]
    east = coords[0][1]
    west = coords[0][1]
    for coord in coords:
        if coord[0] > north:
            north = coord[0]
        elif coord[0] < south:
            south = coord[0]
        if coord[1] > east:
            east = coord[1]
        elif coord[1] < west:
            west = coord[1]
    return {
        'west': west,
        'south': south,
        'east': east,
        'north': north,
    }


def bbox_from_coords_csv(
    csv: TableInput,
) -> BoundingBoxDict:
    """Gets the bounding box from a csv/dataframe of coordinates."""
    # TODO : make method
    raise NotImplementedError


def bbox_from_shp(
    shapefile: ShapefileInput,
) -> BoundingBoxDict:
    """Gets the bounding box from a shapefile."""
    # make sure we have geopandas
    if 'gpd' not in dir():
        import geopandas as gpd

    if isinstance(shapefile, gpd.GeoDataFrame):
        geo_df = shapefile
    else:
        if isinstance(shapefile, str):
            shapefile = Path(shapefile)
        if not shapefile.exists():
            raise FileNotFoundError(
                f'Input path {shapefile} is not found.')
        if not shapefile.suffix == '.shp':
            raise ValueError(
                f'Input path {shapefile} is not a .shp file!'
            )
        geo_df = gpd.read_file(shapefile)

    # read GeoDataFrame and reproject if necessary
    if geo_df.crs.to_epsg() != 4326:
        geo_df = geo_df.to_crs(4326)
    west, south, east, north = geo_df.geometry.total_bounds

    # return bounding box dictionary
    return {
        'west': west,
        'south': south,
        'east': east,
        'north': north,
    }


def bbox_from_raster(
    raster: RasterInput,
) -> BoundingBoxDict:
    """Gets the bounding box from a raster."""

    # TODO : make method
    raise NotImplementedError
