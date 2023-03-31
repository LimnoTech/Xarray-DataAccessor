import warnings
import logging
import itertools
import multiprocessing
from xarray_data_accessor.multi_threading import get_multithread
from pathlib import Path
from datetime import datetime
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
    CoordsTuple,
    ResolutionTuple,
    Shapefile,
    ResampleDict,
    DataGrabberDict,
)
from typing import (
    Optional,
    Tuple,
    Union,
    List,
    Dict,
    TypedDict,
)
from types import ModuleType
import xarray as xr
import pandas as pd
import numpy as np
from rasterio.enums import Resampling

# control weather to use dask for xarray computation
try:
    import dask.distributed
    DASK_DISTRIBUTE = True
except ImportError:
    DASK_DISTRIBUTE = False
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True

except ImportError:
    HAS_GEOPANDAS = False
    Shapefile = Union[str, Path]


class DataAccessor:
    """Main class to get a data."""

    PossibleAOIInputs = Union[
        str,
        Path,
        CoordsTuple,
        List[CoordsTuple],
        xr.DataArray,
    ]

    def __init__(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        start_time: Union[datetime, str, int],
        end_time: Union[datetime, str, int],
        coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[Union[str, Path]] = None,
        shapefile: Optional[Union[str, Path]] = None,
        raster: Optional[Union[str, Path, xr.DataArray]] = None,
        multithread: bool = True,
        use_dask: bool = DASK_DISTRIBUTE,
    ) -> None:
        """Main data puller class.

        Acts as a portal to underlying data accessor classes defined for 
        specific datasets (i.e. ERA5, DAYMET, etc.). Responsible for cleaning
        non-dataset-specific inputs (i.e. bounding box, datetimes), and making
        sure the desired dataset exists.

        All datasets must have a {dataset}_data_accessor.py and 
        {dataset}_datasets_info.py file.
        NOTE: We should switch to a Factory/Plugin architecture for data accessors!
        NOTE: We should switch to "partial implementation" for BBOX arguments.

        Arguments:
            :param dataset_name: A valid/supported dataset_name.
            :param variables: A list of variables from param:dataset_name.
            :param start_time: Time/date to start at (inclusive).
            :param end_time: Time/date to stop at (exclusive).
            :param coordinates: Coordinates to define the AOI.
            :param csv_of_coords: A csv of lat/longs to define the AOI.
            :param shapefile: A shapefile (.shp) to define the AOI.
            :param raster: A raster to define the AOI.
            :param multithread: Whether to multi-thread of not.
                If dask is imported, multi-threading is handled by dask.
                Otherwise it is handled by base Python.
        """
        # see if the dataset requested is available
        self._supported_datasets_info = None
        self._supported_datasets = None
        self._supported_accessors = None

        self.dataset_key = None
        self.dataset_name = None
        for k, v in self.supported_datasets.items():
            if dataset_name in v:
                self.dataset_key = k
                self.dataset_name = dataset_name
        if self.dataset_name is None:
            return ValueError(
                f'Cant find support for param:dataset_name={dataset_name}'
            )

        # set variables
        if isinstance(variables, str):
            variables = [variables]
        self.variables = variables

        # control multithreading
        self.multithread = multithread
        self.use_dask = use_dask

        # init start/end time
        self.start_dt = self.get_datetime(start_time)
        self.end_dt = self.get_datetime(end_time)

        # get AOI inputs set up
        inputs = {
            'coordinates': coordinates,
            'csv_of_coords': csv_of_coords,
            'shapefile': shapefile,
            'raster': raster,
        }

        valid_inputs = [(k, v) for k, v in inputs.items() if v is not None]

        if len(valid_inputs) == 0:
            raise ValueError(
                f'Must use one of the following AOI selectors: {list(inputs.keys())}'
            )
        elif len(valid_inputs) > 1:
            raise ValueError(
                f'Can only use one AOI selector! Multiple applied {inputs.items()}'
            )
        else:
            valid_inputs = valid_inputs[0]

        self.aoi_input_type, self.aoi_input = valid_inputs
        print(f'Using {self.aoi_input_type} to select AOI')

        # get the bounding box coordinates
        self.bbox = self.get_bounding_box(
            aoi_input_type=self.aoi_input_type,
            aoi_input=self.aoi_input,
        )

        # set up empty attribute to store the dataset later
        self.xarray_dataset = None

        print(
            f'ERA5DataAccessor object successfully initialized! '
            f'Use ERA5DataAccessor.inputs_dict to verify your inputs.'
        )

    @property
    def supported_datasets_info(self) -> Dict[str, ModuleType]:
        """
        Finds all supported datasets by their info module.
        NOTE: This can be converted into a Factory implementation later.
        """
        if self._supported_datasets_info is None:
            self._supported_datasets_info = {}

            # go thru each dataset and try to add their info module
            # TODO: this could be a factory implementation!
            try:
                import xarray_data_accessor.era5_datasets_info as era5_datasets_info
                self._supported_datasets_info['ERA5'] = era5_datasets_info
            except ImportError:
                pass
        # check if things look correct
        if len(self._supported_datasets_info) == 0:
            raise ValueError(
                f'No datasets supported! Did you move/delete modules?'
            )
        return self._supported_datasets_info

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        """
        Lists names of sub-datasets associated with each dataset.
        """
        if self._supported_datasets is None:
            self._supported_datasets = {}

            for key, module in self.supported_datasets_info.items():
                self._supported_datasets[key] = module.DATASET_NAMES

        # check if things look correct
        if len(self._supported_datasets) != len(self.supported_datasets_info):
            warnings.warn(
                message=(
                    'Cant find a list of dataset names for all dataset info '
                    'files! This may cause problems.'
                ),
            )
        return self._supported_datasets

    @staticmethod
    def _get_era5_accessor() -> object:
        try:
            import xarray_data_accessor.era5_data_accessor as era5_data_accessor
            return era5_data_accessor.ERA5DataAccessor
        except ImportError:
            raise ImportError(
                'era5_datasets_info.py was found but not era5_data_accessor.py! '
                'Did you move or delete files?'
            )

    @property
    def _accessors(self) -> Dict[str, callable]:
        """Maps dataset names to an import of their accessors"""
        return {
            'ERA5': self._get_era5_accessor,
        }

    @property
    def supported_accessors(self) -> Dict[str, object]:
        """
        Returns the DataAccessor object for each dataset.
        """
        if self._supported_accessors is None:
            self._supported_accessors = {}

            for key in self.supported_datasets_info.keys():
                self._supported_accessors[key] = self._accessors[key]()

        # check if things look correct
        if len(self._supported_accessors) != len(self.supported_datasets_info):
            warnings.warn(
                message=(
                    'Cant find a DataAccessor for all dataset info files! '
                    'This may cause problems.'
                ),
            )
        return self._supported_accessors

    @staticmethod
    def get_datetime(input_date: Union[str, datetime, int]) -> datetime:
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

    @staticmethod
    def _bbox_from_coords(
        coords: Union[CoordsTuple, List[CoordsTuple]],
    ) -> BoundingBoxDict:
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

    @staticmethod
    def _bbox_from_coords_csv(
        csv: Union[str, Path, pd.DataFrame],
    ) -> BoundingBoxDict:
        # TODO : make method
        raise NotImplementedError

    @staticmethod
    def _bbox_from_shp(
        shapefile: Shapefile,
    ) -> BoundingBoxDict:
        if not HAS_GEOPANDAS:
            raise ImportError(
                f'To create a bounding box from shapefile you need geopandas installed!'
            )
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

    @staticmethod
    def _bbox_from_raster(
        raster: Union[str, Path, xr.DataArray],
    ) -> BoundingBoxDict:

        # TODO : make method
        raise NotImplementedError

    def get_bounding_box(
        self,
        aoi_input_type: str,
        aoi_input: PossibleAOIInputs,
    ) -> BoundingBoxDict:
        bbox_function_mapper = {
            'coordinates': self._bbox_from_coords,
            'csv_of_coords': self._bbox_from_coords_csv,
            'shapefile': self._bbox_from_shp,
            'raster': self._bbox_from_raster,
        }
        return bbox_function_mapper[aoi_input_type](aoi_input)

    @property
    def inputs_dict(
        self,
    ) -> Dict[str, Union[str, Dict[str, float], datetime, List[str]]]:
        return {
            'Dataset name': self.dataset_name,
            'Dataset source': self.dataset_source,
            'AOI type': self.aoi_input_type,
            'AOI bounding box': self.bbox,
            'Start datetime': self.start_dt,
            'End datetime': self.end_dt,
            'Variables': self.variables,
            'Multithreading': str(self.multithread),
        }

    def chunk_dataset(
        self,
        chunk_size: int,
    ) -> xr.Dataset:
        if self.xarray_dataset is None:
            raise ValueError(
                'self.xarray_dataset is None! You must use get_data() first.'
            )

    def _resample_slice(
        self,
        data: xr.Dataset,
        resample_dict: ResampleDict,
    ) -> Tuple[int, xr.Dataset]:
        return (
            resample_dict['index'],
            data.rio.reproject(
                dst_crs=resample_dict['crs'],
                shape=(resample_dict['height'], resample_dict['width']),
                resampling=getattr(
                    Resampling, resample_dict['resampling_method']),
                kwargs={'dst_nodata': np.nan},
            )
        )

    def resample_dataset(
        self,
        resolution_factor: Optional[Union[int, float]] = None,
        xy_resolution_factors: Optional[ResolutionTuple] = None,
        resample_method: Optional[str] = None,
    ) -> xr.Dataset:
        """Resamples self.xarray_dataset

        resolution_factor: the number of times FINER to make the dataset (applied to both dimensions).
            For example: resolution=10 on a 0.25 -> 0.025 resolution.
        xy_resolution_factors: Allows one to specify a resolution factor for the X[0] and Y[1] dimensions.
        resample_method: A valid resampling method from rasterio.enums.Resample (default='nearest').
            NOTE: Do not use any averaging resample methods when working with a categorical raster!
            Bilinear resampling is the default.
        """
        # TODO: switch to xarray.interp() This works but overflows memory for large datasets.
        # verify all required inputs are present
        if self.xarray_dataset is None:
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

        x_dim = self.xarray_dataset.attrs['x_dim']
        y_dim = self.xarray_dataset.attrs['y_dim']

        # rioxarray expects x/y dimensions
        renamed = False
        if x_dim != 'x' or y_dim != 'y':
            self.xarray_dataset = self.xarray_dataset.rename(
                {x_dim: 'x', y_dim: 'y'}
            )
            renamed = True

        width = int(len(self.xarray_dataset.x) * xy_resolution_factors[0])
        height = int(len(self.xarray_dataset.y) * xy_resolution_factors[1])
        crs = self.xarray_dataset.rio.crs
        self.xarray_dataset = self.xarray_dataset.chunk(
            {'time': 1, 'x': 100, 'y': 100},
        )

        logging.info(
            f'Resampling to height={height}, width={width}. dateime={datetime.now()}'
        )
        self.xarray_dataset = self._resample_slice(
            data=self.xarray_dataset,
            resample_dict={
                'height': height,
                'width': width,
                'resampling_method': resample_method,
                'crs': crs,
                'index': 1,
            })[-1]

        if renamed:
            self.xarray_dataset = self.xarray_dataset.rename(
                {'x': x_dim, 'y': y_dim}
            )
        logging.info(f'Resampling complete: datetime={datetime.now()}')
        logging.info(f'Resampled dataset info: {self.xarray_dataset.dims}')
        return self.xarray_dataset

    def get_data(
        self,
        overwrite: bool = False,
        resolution_factor: Optional[Union[int, float]] = None,
        xy_resolution_factors: Optional[ResolutionTuple] = None,
        chunk_size: Optional[int] = None,
        chunk_dict: Optional[Dict[str, int]] = None,
        dont_chunk: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Main function to get data. Updated self.xarray_dataset

        Arguments:
            :param overwrite:
            :param resolution_factor:
            :param chunk_size:
            :param chunk_dict:
            :param dont_chunk:

        Return:
            An xarray.Dataset of the desired specification.
        """
        # prevent accidental overwrite since the calls take a while
        if self.xarray_dataset is not None:
            if overwrite:
                warnings.warn(
                    'A xarray Dataset previously saved is being overwritten!'
                )
            else:
                raise ValueError(
                    'A xarray Dataset is already saved to this object. '
                    'To overwrite set .pull_data param:overwrite=True'
                )

        # get accessor and pull data
        data_accessor = self.supported_accessors[self.dataset_key](
            dataset_name=self.dataset_name,
            multithread=self.multithread,
            use_dask=self.use_dask,
            kwargs=kwargs,
        )

        dataset = data_accessor.get_data(
            self.variables,
            self.start_dt,
            self.end_dt,
            self.bbox,
        )

        # return the dictionary is something went wrong
        if isinstance(dataset, dict):
            warnings.warn(
                'Dictionary of datasets being returned since merged failed!'
            )
            return dataset

        # set object attribute to point to the dataset
        self.xarray_dataset = dataset

        # resample the dataset if desired
        if resolution_factor is not None:
            # quickly chunk to avoid worker memory overflow
            self.resample_dataset(
                resolution_factor=resolution_factor,
                xy_resolution_factors=xy_resolution_factors,
            )

        # chunk dataset if desired (default iss 500000 observations per chunk)
        if chunk_size is None:
            chunk_size = 500000
        if chunk_dict is not None:
            try:
                self.xarray_dataset = self.xarray_dataset.chunk(chunk_dict)
            except ValueError as e:
                warnings.warn(
                    f'Could not use param:chunk_dict due to the following exception: {e}'
                    f'Defaulting to chunk size = {chunk_size}.'
                )
                chunk_dict = None
        if not dont_chunk and chunk_dict is None:
            self.chunk_dataset(chunk_size=chunk_size)

        return self.xarray_dataset

    def unlock_and_clean(
        self,
    ) -> None:
        """If temp files were created, this deletes them"""

        temp_files = []
        for file in Path.cwd().iterdir():
            if 'temp_data' in file.name:
                temp_files.append(file)

        if len(temp_files) > 1:
            # try to unlink data from file
            self.xarray_dataset.close()

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

    def _coords_in_bbox(
        self,
        coords: CoordsTuple,
    ) -> bool:
        lat, lon = coords
        conditionals = [
            (lat <= self.bbox['north']),
            (lat >= self.bbox['south']),
            (lon <= self.bbox['east']),
            (lon >= self.bbox['west']),
        ]
        if len(list(set(conditionals))) == 1 and conditionals[0] is True:
            return True
        return False

    def _verify_variables(
        self,
        variables: Optional[Union[str, List[str]]] = None,
    ) -> List[str]:
        if variables is None:
            return list(self.xarray_dataset.data_vars)
        elif isinstance(variables, str):
            variables = [variables]

        # check which variables are available
        cant_add_variables = []
        data_variables = []
        for v in variables:
            if v in list(self.xarray_dataset.data_vars):
                data_variables.append(v)
            else:
                cant_add_variables.append(v)
        variables = list(set(data_variables))
        if len(cant_add_variables) > 0:
            warnings.warn(
                f'The following requested variables are not in the dataset:'
                f' {cant_add_variables}.'
            )
        return data_variables

    def _get_coords_df(
        self,
        csv_of_coords: Optional[Union[str, Path, pd.DataFrame]] = None,
        coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        coords_id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        if csv_of_coords is not None:
            if isinstance(csv_of_coords, str):
                csv_of_coords = Path(csv_of_coords)
            if isinstance(csv_of_coords, Path):
                if not csv_of_coords.exists() or not csv_of_coords.suffix == '.csv':
                    raise ValueError(
                        f'param:csv_of_coords must be a valid .csv file.'
                    )
            if isinstance(csv_of_coords, pd.DataFrame):
                coords_df = csv_of_coords
            else:
                coords_df = pd.read_csv(csv_of_coords)

            if coords_id_column is not None:
                coords_df.set_index(coords_id_column, inplace=True)

        elif coords is not None:
            # TODO: build a dataframe
            raise NotImplementedError
        else:
            raise ValueError(
                'Must specify either param:coords or param:csv_of_coords'
            )
        return coords_df

    @staticmethod
    def _grab_data_to_df(
        input: Tuple[str, xr.DataArray]
    ) -> pd.Series:
        """Testing performance of slicing first and extracting values in parallel"""
        if len(input[-1].values.shape) == 2:
            return pd.Series(
                data=input[-1].values.mean(axis=1),
                name=input[0],
            )
        return pd.Series(
            data=input[-1].values,
            name=input[0],
        )

    def _save_dataframe(
        self,
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
                f'Output directory {save_table_dir} does not exist!'
            )

        if save_table_suffix is None or save_table_suffix == '.parquet':
            out_path = Path(
                save_table_dir / f'{prefix}{variable}.parquet'
            )
            df.to_parquet(out_path)

        elif save_table_suffix == '.csv':
            out_path = Path(
                save_table_dir / f'{prefix}{variable}.csv'
            )
            df.to_csv(out_path)

        elif save_table_suffix == '.xlsx':
            out_path = Path(
                save_table_dir / f'{prefix}{variable}.xlsx'
            )
            df.to_excel(out_path)
        else:
            raise ValueError(
                f'{save_table_suffix} is not a valid table format!'
            )
        logging.info(
            f'Data for variable={variable} saved @ {save_table_dir}'
        )
        return out_path

    def _get_data_table_vectorized(
        self,
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
            f'Extracting {variable} data (vectorized method)'
        )

        # get batches of max 100 points to avoid memory overflow
        batch_size = 100
        start_stops_idxs = list(range(
            0,
            len(self.xarray_dataset.time) + 1,
            batch_size,
        ))

        # init list to store dataframes
        out_dfs = []

        for i, num in enumerate(start_stops_idxs):
            start = num
            if num != start_stops_idxs[-1]:
                stop = start_stops_idxs[i + 1]
            else:
                stop = None
            logging.info(
                f'Processing time slice [{num}:{stop}]. datetime={datetime.now()}'
            )

            # make a copy of the data for our variable of interest
            ds = self.xarray_dataset[variable].isel(
                time=slice(start, stop)
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
                ).sort_index(axis=1).sort_index(axis=0)
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
                f'Saving df to {save_table_dir}, datetime={datetime.now()}'
            )
            table_path = self._save_dataframe(
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

    def get_data_tables(
        self,
        variables: Optional[List[str]] = None,
        coords: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[Union[str, Path, pd.DataFrame]] = None,
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
        variables = self._verify_variables(variables)

        # get x/y columns
        if xy_columns is None:
            xy_columns = ('lon', 'lat')
        x_col, y_col = xy_columns

        # get coords input from csv
        coords_df = self._get_coords_df(
            coords=coords,
            csv_of_coords=csv_of_coords,
            coords_id_column=coords_id_column,
        )

        # get the point x/y values
        point_xs = coords_df[x_col].values
        point_ys = coords_df[y_col].values
        point_ids = [str(i) for i in coords_df.index.values]

        # get dimension names
        x_dim = self.xarray_dataset.attrs['x_dim']
        y_dim = self.xarray_dataset.attrs['y_dim']

        # get all coordinates from the dataset
        ds_xs = self.xarray_dataset[x_dim].values
        ds_ys = self.xarray_dataset[y_dim].values

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
            range(len(self.xarray_dataset[x_dim].values)),
            range(len(self.xarray_dataset[y_dim].values)),
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
        self.xarray_dataset = self.xarray_dataset.chunk(
            {'time': 10000, x_dim: 10, y_dim: 10}
        )

        # get data for each variable
        for variable in variables:
            # TODO: generalize if we bring on more datasets!
            encoding_dict = self.xarray_dataset[variable].encoding
            if 'source' in encoding_dict.keys():
                if 'S3FileSystem' in encoding_dict['source']:
                    warnings.warn(
                        'Extracting values from data being read from the cloud '
                        'is incredibly slow! Consider interrupting this function '
                        'and re-assigning DataAccessor.xarray_dataset to a locally '
                        'saved version of the data.'
                    )
            else:
                out_dict[variable] = self._get_data_table_vectorized(
                    variable,
                    point_ids,
                    id_to_index,
                    xy_dims=(x_dim, y_dim),
                    save_table_dir=save_table_dir,
                    save_table_suffix=save_table_suffix,
                    save_table_prefix=save_table_prefix,
                )
        return out_dict
