import warnings
import logging
import itertools
from pathlib import Path
from datetime import datetime
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
    CoordsTuple,
    ResolutionTuple,
    ShapefileInput,
    RasterInput,
    TimeInput,
    TableInput,
    PossibleAOIInputs,
    ResampleDict,
    DataAccessorBase,
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


class InputDict(TypedDict):
    """Stores all internal inputs to the DataAccessor."""
    dataset_name: str
    aoi_input_type: str
    bounding_box: BoundingBoxDict
    start_datetime: datetime
    end_datetime: datetime
    variables: List[str]
    multithreading: bool


class DataAccessor:
    """Main class to get a data."""

    def __init__(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        start_time: TimeInput,
        end_time: TimeInput,
        coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[TableInput] = None,
        shapefile: Optional[ShapefileInput] = None,
        raster: Optional[RasterInput] = None,
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
        self.start_dt = self._get_datetime(start_time)
        self.end_dt = self._get_datetime(end_time)

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

    # Static Helper Functions ####################################################
    @staticmethod
    def _get_era5_accessor() -> DataAccessorBase:
        # TODO: change to a factory implementation
        try:
            import xarray_data_accessor.era5_data_accessor as era5_data_accessor
            return era5_data_accessor.ERA5DataAccessor
        except ImportError:
            raise ImportError(
                'era5_datasets_info.py was found but not era5_data_accessor.py! '
                'Did you move or delete files?'
            )

    # TODO: replace with ref to utility functions
    @staticmethod
    def _get_datetime(input_date: TimeInput) -> datetime:
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

    @staticmethod
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

    @staticmethod
    def _bbox_from_coords_csv(
        csv: TableInput,
    ) -> BoundingBoxDict:
        """Gets the bounding box from a csv/dataframe of coordinates."""
        # TODO : make method
        raise NotImplementedError

    @staticmethod
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

    @staticmethod
    def _bbox_from_raster(
        raster: RasterInput,
    ) -> BoundingBoxDict:
        """Gets the bounding box from a raster."""

        # TODO : make method
        raise NotImplementedError

    # Properties ###############################################################
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

    @property
    def _accessors(self) -> Dict[str, callable]:
        """Maps dataset names to an import of their accessors"""
        return {
            'ERA5': self._get_era5_accessor,
        }

    @property
    def supported_accessors(self) -> Dict[str, DataAccessorBase]:
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

    @property
    def inputs_dict(
        self,
    ) -> InputDict:
        return {
            'dataset_name': self.dataset_name,
            'aoi_input_type': self.aoi_input_type,
            'bounding_box': self.bbox,
            'start_datetime': self.start_dt,
            'end_datetime': self.end_dt,
            'variables': self.variables,
            'multithreading': str(self.multithread),
        }

    # Non-Core Methods #########################################################
    def get_bounding_box(
        self,
        aoi_input_type: str,
        aoi_input: PossibleAOIInputs,
    ) -> BoundingBoxDict:
        """Returns a bounding box dictionary from the input AOI."""
        bbox_function_mapper = {
            'coordinates': self._bbox_from_coords,
            'csv_of_coords': self._bbox_from_coords_csv,
            'shapefile': self._bbox_from_shp,
            'raster': self._bbox_from_raster,
        }
        return bbox_function_mapper[aoi_input_type](aoi_input)

    def resample_dataset(
        self,
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

        # set our resampling arguments
        width = int(len(self.xarray_dataset.x) * xy_resolution_factors[0])
        height = int(len(self.xarray_dataset.y) * xy_resolution_factors[1])
        crs = self.xarray_dataset.rio.crs

        # chunk in a way that will speed up the resampling
        self.xarray_dataset = self.xarray_dataset.chunk(
            {'time': 1, 'x': 100, 'y': 100},
        )

        # resample and return the adjusted dataset
        logging.info(
            f'Resampling to height={height}, width={width}. datetime={datetime.now()}'
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

    # CORE DATA ACCESSOR FUNCTIONS ##############################################
    def get_data(
        self,
        overwrite: bool = False,
        resolution_factor: Optional[Union[int, float]] = None,
        xy_resolution_factors: Optional[ResolutionTuple] = None,
        chunk_dict: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Main function to get data. Updated self.xarray_dataset

        Arguments:
            :param overwrite: If True, will overwrite the xarray dataset if it already exists.
            :param resolution_factor: The factor to resample the dataset by.
            :param xy_resolution_factors: The factors to resample the dataset by for X/Y dimensions.
            :param chunk_dict: A dictionary of chunk sizes for each dimension.
            :param kwargs: Keyword arguments to pass to the data accessor.

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

        # chunk dataset if desired
        if chunk_dict is not None:
            try:
                self.xarray_dataset = self.xarray_dataset.chunk(chunk_dict)
            except ValueError as e:
                warnings.warn(
                    f'Could not use param:chunk_dict due to the following exception: {e}'
                )

        return self.xarray_dataset
