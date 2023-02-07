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
from cds_data_accessor import CDSDataAccessor
from aws_data_accessor import AWSDataAccessor
from prep_query import (
    get_datetime,
    get_bounding_box,
)
from datasets_info import (
    verify_dataset,
    list_variables,
    AWS_VARIABLES_DICT,
)

# control weather to use dask for xarray computation
try:
    import dask.distributed
    DASK_DISTRIBUTE = True
except ImportError:
    DASK_DISTRIBUTE = False

CoordsTuple = Tuple[float, float]


class GetERA5Data:
    """Main class to get a data."""

    def __init__(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        start_time: Union[datetime, str, int],
        end_time: Union[datetime, str, int],
        dataset_source: Optional[str] = 'CDS',
        coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[Union[str, Path]] = None,
        shapefile: Optional[Union[str, Path]] = None,
        raster: Optional[Union[str, Path, xr.DataArray]] = None,
        multithread: bool = True,
        no_aws: bool = False,
    ) -> None:

        # bring in dataset name and source
        verify_dataset(dataset_name, dataset_source)
        self.dataset_name = dataset_name
        self.dataset_source = dataset_source

        # control multithreading
        self.multithread = multithread
        self.cores = int(multiprocessing.cpu_count())

        # bring in variables
        self.possible_variables = list_variables(
            dataset_name,
            dataset_source,
        )
        self.variables = []
        cant_add_variables = []
        aws_compatible = []
        for var in variables:
            if var in self.possible_variables:
                self.variables.append(var)
                if var in AWS_VARIABLES_DICT:
                    aws_compatible.append(var)
            else:
                cant_add_variables.append(var)

        # warn users about non compatible variables
        if len(cant_add_variables) > 0:
            warnings.warn(
                f'variables {cant_add_variables} are not valid for param:'
                f'dataset_name={self.dataset_name}, param:dataset_source='
                f'{self.dataset_source}.\nPrint GetERA5Data.'
                f'possible_variables to see all valid variables for '
                f'the current dataset name/source combo!'
            )
            del cant_add_variables

        # switch to AWS if possible
        if (len(aws_compatible) == len(self.variables)) and not no_aws:
            warnings.warn(
                'All variables are compatible w/ AWS S3 data access! '
                'Switching self.dataset_source to = AWS. Set no_aws=True '
                'when initializing this object to override this behavior.'
            )
            self.dataset_source = 'AWS'

        # init start/end time
        self.start_dt = get_datetime(start_time)
        self.end_dt = get_datetime(end_time)

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
        self.bbox = get_bounding_box(
            aoi_input=self.aoi_input,
            aoi_input_type=self.aoi_input_type,
        )

        # set up empty attribute to store the dataset later
        self.xarray_dataset = None

        print(
            f'CDSDataAccessor object successfully initialized! '
            f'Use CDSDataAccessor.inputs_dict to verify your inputs.'
        )

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

    @property
    def dataset_accessors(self) -> Dict[str, object]:
        return {
            'AWS': AWSDataAccessor,
            'CDS': CDSDataAccessor,
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
        data_accessor = self.dataset_accessors[self.dataset_source]
        dataset = data_accessor.get_dataset(
            self.dataset_name,
            self.variables,
            self.start_dt,
            self.end_dt,
            self.bbox,
            multithread=self.multithread,
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
        """Converts the output of a CDSDataAccessor function to a pandas dataframe"""
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
