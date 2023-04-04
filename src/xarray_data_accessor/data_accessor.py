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
    InputDict,
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

# control weather to use dask for xarray computation
try:
    import dask.distributed
    DASK_DISTRIBUTE = True
except ImportError:
    DASK_DISTRIBUTE = False


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
        self.start_dt = self.__get_datetime(start_time)
        self.end_dt = self.__get_datetime(end_time)

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
