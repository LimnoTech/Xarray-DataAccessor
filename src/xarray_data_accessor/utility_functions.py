"""Stores non-user facing utility functions"""
import logging
import warnings
from datetime import datetime
import pytz
import pyproj
from pytz.exceptions import UnknownTimeZoneError
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    Optional,
)
from xarray_data_accessor.shared_types import (
    TimeInput,
    CoordsTuple,
    TableInput,
    RasterInput,
    ShapefileInput,
    BoundingBoxDict,
    SpatialResampleDict,
)


def _get_datetime(input_date: TimeInput) -> datetime:
    """Returns a datetime object from a variety of inputs."""
    if isinstance(input_date, datetime):
        return input_date

    elif isinstance(input_date, np.datetime64) or isinstance(input_date, str):
        return pd.to_datetime(input_date)

    # assume int is a year
    elif isinstance(input_date, int):
        if input_date not in list(range(1950, datetime.now().year + 1)):
            raise ValueError(
                f'integer start/end date input={input_date} is not a valid year.',
            )
        return pd.to_datetime(f'{input_date}-01-01')

    else:
        raise ValueError(
            f'start/end date input={input_date} is invalid.',
        )


def _convert_timezone(
    datetime_obj: datetime,
    in_timezone: str,
    out_timezone: str,
) -> datetime:
    """Converts a datetime object to a different timezone."""
    # check if timezone is valid
    for tz in [in_timezone, out_timezone]:
        if tz not in pytz.all_timezones:
            raise UnknownTimeZoneError(
                f'Invalid timezone={tz}. '
                f'See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones '
                f'for a list of timezone identified strings (TZ identifier)',
            )

    # return converted datetime
    return (
        datetime_obj
        .tz_localize(pytz.timezone(in_timezone))
        .astimezone(pytz.timezone(out_timezone))
        .replace(tzinfo=None)
    )


def _prep_small_bbox(
    bbox: BoundingBoxDict,
) -> BoundingBoxDict:
    """Converts a single point bbox to a small bbox with 0.1 degree sides"""
    if bbox['north'] == bbox['south']:
        bbox['north'] += 0.05
        bbox['south'] -= 0.05
    if bbox['east'] == bbox['west']:
        bbox['east'] += 0.05
        bbox['west'] -= 0.05
    return bbox


def _bbox_from_coords(
    coords: List[CoordsTuple],
) -> BoundingBoxDict:
    """Returns a bounding box from a list of coordinates."""
    # TODO: add buffer so the edge isn't the exact coordinate?
    # TODO : make method
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
                f'Input path {shapefile} is not found.',
            )
        if not shapefile.suffix == '.shp':
            raise ValueError(
                f'Input path {shapefile} is not a .shp file!',
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


def _resample_slice(
    data: xr.Dataset,
    resample_dict: SpatialResampleDict,
) -> Tuple[int, xr.Dataset]:
    return (
        resample_dict['index'],
        data.rio.reproject(
            dst_crs=resample_dict['crs'],
            shape=(resample_dict['height'], resample_dict['width']),
            resampling=getattr(
                Resampling, resample_dict['resampling_method'],
            ),
            kwargs={'dst_nodata': np.nan},
        ),
    )


def _coords_in_bbox(
    bbox: BoundingBoxDict,
    coords: CoordsTuple,
) -> bool:
    lat, lon = coords
    conditionals = [
        (lat <= bbox['north']),
        (lat >= bbox['south']),
        (lon <= bbox['east']),
        (lon >= bbox['west']),
    ]
    if len(list(set(conditionals))) == 1 and conditionals[0] is True:
        return True
    return False


def _convert_xy_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    input_epsg: Optional[int] = None,
    output_epsg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms coordinates from one EPSG to another."""

    if not input_epsg and output_epsg:
        raise ValueError(
            'An input EPSG must be provided if an output EPSG is provided! '
            'This is pulled from xarray_dataset.attrs["EPSG"]. '
            'Please set this attribute to a valid integer EPSG code.',
        )
    if output_epsg and output_epsg != input_epsg:
        # Create a pyproj Transformer for the coordinate conversion
        transformer = pyproj.Transformer.from_crs(
            input_epsg,
            output_epsg,
            always_xy=True,
        )

        # Perform the coordinate conversion
        x, y = transformer.transform(x, y)
    return x, y


def _convert_bbox(
    bbox: BoundingBoxDict,
    known_epsg: int,
) -> BoundingBoxDict:
    """Converts bbox coordinates to a different EPSG."""

    # TODO: consider deprecating this method
    # EPSG:4326 bbox list
    bbox_list_in = [
        bbox['west'],
        bbox['south'],
        bbox['east'],
        bbox['north'],
    ]

    # create a PyProj transformer object
    transformer = pyproj.Transformer.from_crs(
        src_crs=f'EPSG:4326',
        target_crs=f'EPSG:{known_epsg}',
        always_xy=True,
    )

    bbox_list_out = transformer.transform(
        *bbox_list_in,
    )

    return BoundingBoxDict(
        west=bbox_list_out[0],
        east=bbox_list_out[2],
        south=bbox_list_out[1],
        north=bbox_list_out[3],
    )


def _verify_variables(
    xarray_dataset: xr.Dataset,
    variables: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    if variables is None:
        drops = ['spatial_ref']
        return [i for i in list(xarray_dataset.data_vars) if i not in drops]
    elif isinstance(variables, str):
        variables = [variables]

    # check which variables are available
    cant_add_variables = []
    data_variables = []
    for v in variables:
        if v in list(xarray_dataset.data_vars):
            data_variables.append(v)
        else:
            cant_add_variables.append(v)
    variables = list(set(data_variables))
    if len(cant_add_variables) > 0:
        warnings.warn(
            f'The following requested variables are not in the dataset:'
            f' {cant_add_variables}.',
        )
    return data_variables


def _get_coords_df(
    csv_of_coords: Optional[TableInput] = None,
    coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
    coords_id_column: Optional[str] = None,
) -> pd.DataFrame:
    if csv_of_coords is not None:
        if isinstance(csv_of_coords, str):
            csv_of_coords = Path(csv_of_coords)
        if isinstance(csv_of_coords, Path):
            if not csv_of_coords.exists() or not csv_of_coords.suffix == '.csv':
                raise ValueError(
                    f'param:csv_of_coords must be a valid .csv file.',
                )
        if isinstance(csv_of_coords, pd.DataFrame):
            coords_df = csv_of_coords
        else:
            coords_df = pd.read_csv(csv_of_coords)

        if coords_id_column is not None:
            coords_df.set_index(coords_id_column, inplace=True)

    elif coords is not None:
        # get lon/lat coordinates
        if isinstance(coords, tuple):
            coords = [coords]
        ids = pd.Index(
            data=[i for i in range(len(coords))],
            dtype='int32',
            name='point_id',
        )
        lons = pd.Series(
            data=[c[0] for c in coords],
            dtype='float32',
            name='lon',
            index=ids,
        )
        lats = pd.Series(
            data=[c[1] for c in coords],
            dtype='float32',
            name='lat',
            index=ids,
        )
        coords_df = pd.concat(
            [lons, lats],
            axis=1,
        )
    else:
        raise ValueError(
            'Must specify either param:coords or param:csv_of_coords',
        )
    return coords_df


def _get_data_table_vectorized(
    xarray_dataset: xr.Dataset,
    variable: str,
    point_ids: List[str],
    id_to_index: Dict[str, int],
    xy_dims: Tuple[str, str],
    save_table_dir: Optional[Union[str, Path]] = None,
    save_table_suffix: Optional[str] = None,
    save_table_prefix: Optional[str] = None,
) -> pd.DataFrame:

    # unpack dimension names
    x_dim, y_dim = xy_dims
    logging.info(
        f'Extracting {variable} data (vectorized method)',
    )

    # get batches of max 100 points to avoid memory overflow
    batch_size = 100
    start_stops_idxs = list(
        range(
            0,
            len(xarray_dataset.time) + 1,
            batch_size,
        ),
    )

    # init list to store dataframes
    out_dfs = []

    for i, num in enumerate(start_stops_idxs):
        start = num
        if num != start_stops_idxs[-1]:
            stop = start_stops_idxs[i + 1]
        else:
            stop = None
        logging.info(
            f'Processing time slice [{num}:{stop}]. datetime={datetime.now()}',
        )

        # make a copy of the data for our variable of interest
        ds = xarray_dataset[variable].isel(
            time=slice(start, stop),
        ).load()

        # convert x/y dimensions to integer indexes
        ds[x_dim] = list(range(len(ds[x_dim].values)))
        ds[y_dim] = list(range(len(ds[y_dim].values)))

        # "stack" the dataset and convert to a dataframe
        ds_df = ds.stack(
            xy_index=(x_dim, y_dim),
            create_index=False,
        ).to_dataframe().drop(columns=[x_dim, y_dim]).reset_index()
        del ds

        # pivot the dataframe to have all point combo ids as columns
        ds_df = ds_df.pivot(
            index='time',
            columns='xy_index',
            values=variable,
        )
        ds_df.index.name = 'datetime'

        # convert the dictionary to a dataframe
        index_map = pd.DataFrame(
            list(id_to_index.items()),
            columns=['key', 'index'],
        ).set_index('key')

        # get the point indexes to query data with
        point_indexes = index_map.loc[point_ids].values.flatten()
        data = ds_df.loc[:, point_indexes].values
        index = ds_df.index
        del ds_df

        # create your final dataframe
        out_dfs.append(
            pd.DataFrame(
                columns=point_ids,
                index=index,
                data=data,
            ).sort_index(axis=1).sort_index(axis=0),
        )
        del data
        del index
        del index_map

    out_df = pd.concat(
        out_dfs,
        axis=0,
    )
    del out_dfs

    # save to file
    if save_table_dir:
        logging.info(
            f'Saving df to {save_table_dir}, datetime={datetime.now()}',
        )
        table_path = _save_dataframe(
            out_df,
            variable=variable,
            save_table_dir=save_table_dir,
            save_table_suffix=save_table_suffix,
            save_table_prefix=save_table_prefix,
        )
        del out_df
        return table_path
    else:
        return out_df


def _save_dataframe(
    df: pd.DataFrame,
    variable: str,
    save_table_dir: Optional[Union[str, Path]] = None,
    save_table_suffix: Optional[str] = None,
    save_table_prefix: Optional[str] = None,
) -> Path:
    # save if necessary
    if not save_table_prefix:
        prefix = ''
    else:
        prefix = save_table_prefix

    no_success = False
    if isinstance(save_table_dir, str):
        save_table_dir = Path(save_table_dir)
    if not save_table_dir.exists():
        warnings.warn(
            f'Output directory {save_table_dir} does not exist!',
        )

    if save_table_suffix is None or save_table_suffix == '.parquet':
        out_path = Path(
            save_table_dir / f'{prefix}{variable}.parquet',
        )
        df.to_parquet(out_path)

    elif save_table_suffix == '.csv':
        out_path = Path(
            save_table_dir / f'{prefix}{variable}.csv',
        )
        df.to_csv(out_path)

    elif save_table_suffix == '.xlsx':
        out_path = Path(
            save_table_dir / f'{prefix}{variable}.xlsx',
        )
        df.to_excel(out_path)
    else:
        raise ValueError(
            f'{save_table_suffix} is not a valid table format!',
        )
    logging.info(
        f'Data for variable={variable} saved @ {save_table_dir}',
    )
    return out_path
