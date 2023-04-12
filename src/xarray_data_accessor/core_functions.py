import logging
import warnings
import itertools
import pytz
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from rasterio.enums import Resampling
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from xarray_data_accessor.shared_types import (
    CoordsTuple,
    TimeInput,
    TableInput,
    ShapefileInput,
    RasterInput,
    BoundingBoxDict,
    ResolutionTuple,
)
from xarray_data_accessor.data_accessors.factory import (
    DataAccessorFactory,
)
from xarray_data_accessor import utility_functions


def get_xarray_dataset(
    data_accessor_name: str,
    dataset_name: str,
    variables: Union[str, List[str]],
    start_time: TimeInput,
    end_time: TimeInput,
    start_end_timezone: Optional[str] = None,
    coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
    csv_of_coords: Optional[TableInput] = None,
    shapefile: Optional[ShapefileInput] = None,
    raster: Optional[RasterInput] = None,
    combine_aois: bool = False,
    resample_factor: Optional[int] = None,
    xy_resolution_factors: Optional[ResolutionTuple] = None,
    resample_method: Optional[str] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Arguments:
        :param data_accessor_name: A valid/supported data_accessor_name.
            NOTE: see DataAccessorFactory.data_accessor_names().
        :param dataset_name: A valid/supported dataset_name.
            NOTE: see DataAccessorFactory.supported_datasets() for a mapping
            of data accessor names to supported dataset names.
        :param variables: A list of variables from param:dataset_name.
            NOTE: use DataAccessorFactory.supported_variables() for a mapping
            of data accessor + dataset names to supported variables.
        :param start_time: Time/date to start at (inclusive).
        :param end_time: Time/date to stop at (exclusive).
        :param start_end_timezone: The timezone for start/end time (default is UTC).
            NOTE: See list of possible timezones at the link below
            https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
        :param aoi_coordinates: Coordinates to define the AOI.
        :param aoi_csv_of_coords: A csv of lat/longs to define the AOI.
        :param aoi_shapefile: A shapefile (.shp) to define the AOI.
        :param aoi_raster: A raster to define the AOI.
        :param combine_aois: If True, combines all AOIs into one.
        :param resample_factor: The factor to resample the data by.
        :param xy_resolution_factors: The X,Y dimension factors to resample the data by.
        :param kwargs: Additional keyword arguments to pass to 
            the underlying data accessor.get_data() function.

    Return:
        An xarray dataset.
    """
    # check that the data accessor exists and get its class
    if data_accessor_name not in DataAccessorFactory.data_accessor_names():
        raise ValueError(
            f"Data accessor '{data_accessor_name}' does not exist. "
            f"Please choose from {DataAccessorFactory.data_accessor_names()}."
        )
    else:
        data_accessor = DataAccessorFactory.get_data_accessor(
            data_accessor_name,
        )

    # clean up inputs
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(coordinates, tuple):
        coordinates = [coordinates]

    # define time AOI and convert timezone if necessary
    start_dt = utility_functions._get_datetime(start_time)
    end_dt = utility_functions._get_datetime(end_time)

    if start_end_timezone:
        start_dt = utility_functions._convert_timezone(
            start_dt,
            in_timezone=start_end_timezone,
            out_timezone='UTC',
        )
        end_dt = utility_functions._convert_timezone(
            end_dt,
            in_timezone=start_end_timezone,
            out_timezone='UTC',
        )

    # define spatial AOI
    bounding_box = get_bounding_box(
        coords=coordinates,
        csv=csv_of_coords,
        shapefile=shapefile,
        raster=raster,
        union_bbox=combine_aois,
    )

    # get data
    xarray_dataset = data_accessor.get_data(
        dataset_name=dataset_name,
        variables=variables,
        start_dt=start_dt,
        end_dt=end_dt,
        bbox=bounding_box,
        kwargs=kwargs,
    )

    # resample data is necessary
    if resample_factor or xy_resolution_factors:
        xarray_dataset = spatial_resample(
            xarray_dataset,
            resolution_factor=resample_factor,
            xy_resolution_factors=xy_resolution_factors,
            resample_method=resample_method,
        )

    # return the final dataset
    return xarray_dataset


def get_bounding_box(
    coords: Optional[List[CoordsTuple]] = None,
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
                outputs_dict[key] = utility_functions._bbox_from_coords(value)
            elif key == 'csv':
                outputs_dict[key] = utility_functions._bbox_from_coords_csv(
                    value)
            elif key == 'shapefile':
                outputs_dict[key] = utility_functions._bbox_from_shp(value)
            elif key == 'raster':
                outputs_dict[key] = utility_functions._bbox_from_raster(value)

    # if we are unionizing the bounding boxes, do that, otherwise return the bbox
    if union_bbox:
        return utility_functions._unionize_bbox(list(outputs_dict.values()))
    else:
        return list(outputs_dict.values())[0]


def subset_time_by_timezone(
    xarray_dataset: xr.Dataset,
    timezone: str,
    start_time: Optional[TimeInput] = None,
    end_time: Optional[TimeInput] = None,
) -> xr.Dataset:
    """Subsets the time dimension using a user supplied time zone

    Xarray does not allow for timezone based indexes. This function
    acts as a simple pytz wrapper to subselect data using a local timezone.

    Arguments:
        :param xarray_dataset: The xarray dataset to subset.
        :param timezone: A valid pytz timezone string. 
            Example: 'America/New_York'
        :start_time: The start time input to subset from.
        :end_time: The end time input to subset to.

    Return:
        The subset xarray dataset.
    """
    # check if necessary dimensions exist
    if 'time' not in xarray_dataset.dims:
        raise ValueError(
            'The dataset must have a time dimension to subset by timezone!'
        )
    if 'timezone' not in xarray_dataset.attrs.keys():
        warnings.warn(
            'The dataset is lacking a timezone attribute! Assuming UTC.'
        )
        out_timezone = 'UTC'
    out_timezone = xarray_dataset.attrs['timezone']

    # convert times to match the xarray dataset timezone
    if start_time:
        start_time = utility_functions._convert_timezone(
            start_time,
            in_timezone=timezone,
            out_timezone=out_timezone,
        )
    if end_time:
        end_time = utility_functions._convert_timezone(
            end_time,
            in_timezone=timezone,
            out_timezone=out_timezone,
        )

    # subset dataset
    return xarray_dataset.sel(
        time=slice(start_time, end_time),
    )


def spatial_resample(
    xarray_dataset: xr.Dataset,
    resolution_factor: Optional[Union[int, float]] = None,
    xy_resolution_factors: Optional[ResolutionTuple] = None,
    resample_method: Optional[str] = None,
) -> xr.Dataset:
    """Resamples self.xarray_dataset

    Arguments:
        :param resolution_factor: the number of times FINER to make the 
            dataset (applied to both dimensions).
            Example: resolution=10 on a 0.25 -> 0.025 resolution.
        :param xy_resolution_factors: Allows one to specify a resolution 
            factor for the X[0] and Y[1] dimensions.
        :param resample_method: A valid resampling method from rasterio.enums.Resample 
            NOTE: The default is 'nearest'. Do not use any averaging resample 
            methods when working with a categorical raster! 
            Bilinear resampling is the default.

    Returns:
        The resampled xarray dataset.
    """
    # TODO: switch to xarray.interp() This works but overflows memory for large datasets.
    # verify all required inputs are present
    if xarray_dataset is None:
        raise ValueError(
            'self.xarray_dataset is None! You must use get_data() first.'
        )
    if resolution_factor is None and xy_resolution_factors is None:
        raise ValueError(
            'Must provide an input for either param:resolution_factor or '
            'param:xy_resolution_factors'
        )

    # verify the resample methods
    real_methods = vars(Resampling)['_member_names_']
    if resample_method is None:
        resample_method = 'bilinear'
    elif resample_method not in real_methods:
        raise ValueError(
            f'Resampling method {resample_method} is invalid! Please select from {real_methods}'
        )

    # apply the resampling
    if xy_resolution_factors is None:
        xy_resolution_factors = (resolution_factor, resolution_factor)

    x_dim = xarray_dataset.attrs['x_dim']
    y_dim = xarray_dataset.attrs['y_dim']

    # rioxarray expects x/y dimensions
    renamed = False
    if x_dim != 'x' or y_dim != 'y':
        xarray_dataset = xarray_dataset.rename(
            {x_dim: 'x', y_dim: 'y'}
        )
        renamed = True

    # set our resampling arguments
    width = int(len(xarray_dataset.x) * xy_resolution_factors[0])
    height = int(len(xarray_dataset.y) * xy_resolution_factors[1])
    crs = xarray_dataset.rio.crs

    # chunk in a way that will speed up the resampling
    xarray_dataset = xarray_dataset.chunk(
        {'time': 1, 'x': 100, 'y': 100},
    )

    # resample and return the adjusted dataset
    logging.info(
        f'Resampling to height={height}, width={width}. datetime={datetime.now()}'
    )
    xarray_dataset = utility_functions._resample_slice(
        data=xarray_dataset,
        resample_dict={
            'height': height,
            'width': width,
            'resampling_method': resample_method,
            'crs': crs,
            'index': 1,
        })[-1]

    if renamed:
        xarray_dataset = xarray_dataset.rename(
            {'x': x_dim, 'y': y_dim}
        )
    logging.info(f'Resampling complete: datetime={datetime.now()}')
    logging.info(f'Resampled dataset info: {xarray_dataset.dims}')
    return xarray_dataset


def get_data_tables(
    xarray_dataset: xr.Dataset,
    variables: Optional[List[str]] = None,
    coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
    csv_of_coords: Optional[TableInput] = None,
    coords_id_column: Optional[str] = None,
    xy_columns: Optional[Tuple[str, str]] = None,
    save_table_dir: Optional[Union[str, Path]] = None,
    save_table_suffix: Optional[str] = None,
    save_table_prefix: Optional[str] = None,
) -> Dict[str, Union[pd.DataFrame, Path]]:
    """
    Returns:
        A dictionary with variable names as keys, and dataframes as values
            if save_table_dir==None, or the output table path as values if
            save_table_dir is not None.
    """
    # init output dictionary
    out_dict = {}

    # clean variables input
    variables = utility_functions._verify_variables(
        xarray_dataset,
        variables,
    )

    # get x/y columns
    if xy_columns is None:
        xy_columns = ('lon', 'lat')
    x_col, y_col = xy_columns

    # get coords input from csv
    coords_df = utility_functions._get_coords_df(
        coords=coords,
        csv_of_coords=csv_of_coords,
        coords_id_column=coords_id_column,
    )

    # get the point x/y values
    point_xs = coords_df[x_col].values
    point_ys = coords_df[y_col].values
    point_ids = [str(i) for i in coords_df.index.values]

    # get dimension names
    x_dim = xarray_dataset.attrs['x_dim']
    y_dim = xarray_dataset.attrs['y_dim']

    # get all coordinates from the dataset
    ds_xs = xarray_dataset[x_dim].values
    ds_ys = xarray_dataset[y_dim].values

    # get nearest lat/longs for each sample point
    nearest_x_idxs = np.abs(ds_xs - point_xs.reshape(-1, 1)).argmin(axis=1)
    nearest_y_idxs = np.abs(ds_ys - point_ys.reshape(-1, 1)).argmin(axis=1)

    # get a dict with point IDs as keys, and nearest x/y indices as values
    points_nearest_xy_idxs = dict(zip(
        point_ids,
        zip(nearest_x_idxs, nearest_y_idxs)
    ))

    # get all x/long to y/lat combos
    combos = list(itertools.product(
        range(len(xarray_dataset[x_dim].values)),
        range(len(xarray_dataset[y_dim].values)),
    ))

    # make sure they are in the right order to reshape!
    combo_dict = dict(zip(combos, range(len(combos))))

    # get point id to xy combo index
    id_to_index = {}
    for pid, coord in points_nearest_xy_idxs.items():
        id_to_index[pid] = combo_dict[coord]

    # clear some memory
    del (
        nearest_x_idxs,
        nearest_y_idxs,
        point_xs,
        point_ys,
        combos,
        combo_dict,
    )

    # prep chunks
    xarray_dataset = xarray_dataset.chunk(
        {'time': 10000, x_dim: 10, y_dim: 10}
    )

    # get data for each variable
    for variable in variables:
        out_dict[variable] = utility_functions._get_data_table_vectorized(
            xarray_dataset,
            variable,
            point_ids,
            id_to_index,
            xy_dims=(x_dim, y_dim),
            save_table_dir=save_table_dir,
            save_table_suffix=save_table_suffix,
            save_table_prefix=save_table_prefix,
        )
    return out_dict


def delete_temp_files(
    xarray_dataset: Optional[xr.Dataset] = None,
) -> None:
    """If temp files were created, this deletes them"""

    temp_files = []
    for file in Path.cwd().iterdir():
        if 'temp_data' in file.name:
            temp_files.append(file)

    if len(temp_files) > 1 and xarray_dataset is not None:
        # try to unlink data from file
        xarray_dataset.close()

    could_not_delete = []
    for t_file in temp_files:
        try:
            t_file.unlink()
        except PermissionError:
            could_not_delete.append(t_file)
    if len(could_not_delete) > 0:
        warnings.warn(
            message=(
                f'Could not delete {len(could_not_delete)} temp files '
                f'in directory {Path.cwd()}. You may want to clean them manually.'
            ),
        )
