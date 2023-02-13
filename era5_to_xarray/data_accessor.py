import warnings
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import (
    Optional,
    Tuple,
    Union,
    List,
    Dict,
)
import xarray as xr
import pandas as pd
from era5_data_accessor import ERA5DataAccessor

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
try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False

CoordsTuple = Tuple[float, float]


class DataAccessor:
    """Main class to get a data."""

    PossibleAOIInputs = Union[
        str,
        Path,
        Tuple[float, float],
        List[Tuple[float, float]],
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
        no_aws: bool = False,
    ) -> None:

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

        self.aoi_input = valid_inputs[0][1]
        self.aoi_input_type, self.aoi_input = valid_inputs[0][0]
        print(f'Using {self.aoi_input_type} to select AOI')

        # get the bounding box coordinates
        self.bbox = self.get_bounding_box(
            aoi_input=self.aoi_input,
            aoi_input_type=self.aoi_input_type,
        )

        # set up empty attribute to store the dataset later
        self.xarray_dataset = None

        print(
            f'ERA5DataAccessor object successfully initialized! '
            f'Use ERA5DataAccessor.inputs_dict to verify your inputs.'
        )

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
    def _bbox_from_coords():
        # TODO: add buffer so the edge isn't the exact coordinate
        raise NotImplementedError

    @staticmethod
    def _bbox_from_shp():
        if not HAS_GEOPANDAS:
            raise ImportError(
                f'To create a bounding box from shapefile you need geopandas installed!'
            )
        raise NotImplementedError

    @staticmethod
    def _bbox_from_raster():
        if not HAS_RIOXARRAY:
            raise ImportError(
                f'To create a bounding box from raster you need rioxarray installed!'
            )
        raise NotImplementedError

    @staticmethod
    def get_bounding_box(
        aoi_input: PossibleAOIInputs,
        aoi_input_type: str,
    ):
        raise NotImplementedError

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

    def pull_data(
        self,
        overwrite: bool = False,
    ) -> xr.Dataset:
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
        data_accessor = self.dataset_accessors[self.dataset_source](
            multithread=self.multithread,
            use_dask=DASK_DISTRIBUTE,
        )

        dataset = data_accessor.get_data(
            self.dataset_name,
            self.variables,
            self.start_dt,
            self.end_dt,
            self.bbox,
        )
        # set object attribute to point to the dataset
        self.xarray_dataset = dataset
        return self.xarray_dataset

    # TODO: update this function

    def convert_output_to_table(
        self,
        variables_dict: Dict[str, str],
        coords_dict: Dict[str, Tuple[float, float]],
        output_dict: Dict[str, Dict[str, xr.Dataset]],
    ) -> pd.DataFrame:
        """Converts the output of a ERA5DataAccessor function to a pandas dataframe"""
        df_dicts = []

        for station_id, coords in coords_dict.items():
            df_dict = {
                'station_id': None,
                'datetime': None,
            }

            print(output_dict[station_id].keys())
            for variable, unit in variables_dict.items():
                print(f'Adding {variable}')
                data_array = output_dict[station_id][variable].to_array()
                data_array = data_array.sel(
                    {'longitude': coords[0], 'latitude': coords[1]},
                    method='nearest',
                )

                # init datetime and station id column if empty
                if df_dict['datetime'] is None:
                    df_dict['datetime'] = data_array.time.values
                if df_dict['station_id'] is None:
                    df_dict['station_id'] = [
                        station_id for i in range(len(data_array.time.values))]

                # add variable data
                df_dict[f'{variable}_{unit}'] = data_array.variable.values.squeeze()

            df_dicts.append(pd.DataFrame.from_dict(df_dict))

        out_df = pd.concat(df_dicts)

        # set the index
        if len(out_df.station_id.unique()) == 1:
            out_df.set_index('datetime', inplace=True)
        else:
            out_df.set_index(['station_id', 'datetime'], inplace=True)

        return out_df
