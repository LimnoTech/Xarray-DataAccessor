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
    Optional,
)
from xarray_data_accessor.shared_types import (
    TimeInput,
    CoordsTuple,
    TableInput,
    RasterInput,
    ShapefileInput,
    BoundingBoxDict,
)

# "Front-end" utility functions ###############################################


def get_bounding_box(
    coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
    csv: Optional[TableInput] = None,
    shapefile: Optional[ShapefileInput] = None,
    raster: Optional[RasterInput] = None,
    union_bbox: bool = False,
) -> BoundingBoxDict:
    """Gets the bounding box from a variety of inputs.

    NOTE: if multiple inputs are provided and param:union_bbox is False,
        an error will be raised.

    Arguments:
        coords: a tuple or list of coordinate tuples (lat, lon).
        csv: a csv file or dataframe of coordinates.
        shapefile: a shapefile path or GeoDataFrame.
        raster: a raster path or xarray dataset.
        union_bbox: if True, returns the union of all bounding boxes.

    Returns:
        A bounding box dictionary.
    """
    # get inputs in a dict
    inputs_dict = {
        'coords': coords,
        'csv': csv,
        'shapefile': shapefile,
        'raster': raster,
    }

    # make sure we are not using multiple inputs at once
    if sum(x is not None for x in inputs_dict.values()) > 1 and not union_bbox:
        raise ValueError(
            'Only one input can be used at a time unless param:union_bbox=True!'
        )

    # get bounding boxes
    outputs_dict = {}
    for key, value in inputs_dict.items():
        if value is not None:
            if key == 'coords':
                outputs_dict[key] = _bbox_from_coords(value)
            elif key == 'csv':
                outputs_dict[key] = _bbox_from_coords_csv(value)
            elif key == 'shapefile':
                outputs_dict[key] = _bbox_from_shp(value)
            elif key == 'raster':
                outputs_dict[key] = _bbox_from_raster(value)

    # if we are unionizing the bounding boxes, do that, otherwise return the bbox
    if union_bbox:
        return _unionize_bbox(list(outputs_dict.values()))
    else:
        return list(outputs_dict.values())[0]


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


# "Back-end" utility functions ###############################################


def _bbox_from_coords(
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


def _bbox_from_coords_csv(
    csv: TableInput,
) -> BoundingBoxDict:
    """Gets the bounding box from a csv/dataframe of coordinates."""
    # TODO : make method
    raise NotImplementedError


def _bbox_from_shp(
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


def _bbox_from_raster(
    raster: RasterInput,
) -> BoundingBoxDict:
    """Gets the bounding box from a raster."""

    # TODO : make method
    raise NotImplementedError


def _unionize_bbox(
    bbox_list: List[BoundingBoxDict],
) -> BoundingBoxDict:
    """Returns the union of multiple bounding boxes."""
    # iterate over the bounding boxes and get the union
    for i, bbox in enumerate(bbox_list):
        if i == 0:
            out_bbox = bbox.copy()
        else:
            if bbox['north'] > out_bbox['north']:
                out_bbox['north'] = bbox['north']
            if bbox['south'] < out_bbox['south']:
                out_bbox['south'] = bbox['south']
            if bbox['east'] > out_bbox['east']:
                out_bbox['east'] = bbox['east']
            if bbox['west'] < out_bbox['west']:
                out_bbox['west'] = bbox['west']
    return out_bbox
