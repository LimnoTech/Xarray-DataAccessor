import xarray as xr
import cdsapi
import geopandas as gpd
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import urlopen
from read_into_xarray.era5_datasets_info import (
    verify_dataset,
    DATASET_NAMES,
    AWS_VARIABLES_DICT,
    SINGLE_LEVEL_VARIABLES,
    MISSING_MONTHLY_VARIABLES,
    MISSING_HOURLY_VARIABLES,
    PRESSURE_LEVEL_VARIABLES,
    ERA5_LAND_VARIABLES,
)
from typing import (
    Dict,
    Tuple,
    List,
    Union,
    Optional,
)
from read_into_xarray.data_accessor import BoundingBoxDict
"""
THIS IS THE MOVE: https://cds.climate.copernicus.eu/toolbox/doc/api.html
NOTE: Data is accessed for the whole globe, then cropped on their end.
    Therefore timesteps x variables are the main rate limiting step.
Live CDS API: https://cds.climate.copernicus.eu/live/limits
"""
# TODO: Use dask to parallelize API calls? use dask compute


class AWSDataAccessor:
    supported_datasets = [
        'reanalysis-era5-single-levels',
    ]

    # single level variables on AWS S3 bucket
    aws_variable_mapping = AWS_VARIABLES_DICT

    def __init__(
        self,
        dataset_name: str,
        thread_limit: Optional[int] = None,
        multithread: bool = True,
    ) -> None:
        # check dataset compatibility
        if dataset_name not in self.supported_datasets:
            raise ValueError(
                f'param:dataset_name must be one of the following: '
                f'{self.supported_datasets}'
            )
        self.dataset_name = dataset_name

        # get cores for multiprocessing
        if thread_limit is None:
            thread_limit = multiprocessing.cpu_count
        self.thread_limit = thread_limit

    def possible_variables(self) -> List:
        out_list = []
        for k, v in self.aws_variable_mapping.items():
            out_list.append(k)
            out_list.append(v)
        return out_list

    def get_data(
        self,
        variables: Union[str, List[str]],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        hours_step: Optional[int] = None,
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Main data getter function.

        NOTE: AWS multithreading is best handled across months.
        """
        pass
        #raise NotImplementedError


class CDSDataAccessor:
    InputDict = Dict[str, int]
    file_format_dict = {
        'netcdf': '.nc',
        'grib': '.grib',
    }
    valid_hour_steps = [1, 3, 6, 9, 12]

    supported_datasets = DATASET_NAMES

    def __init__(
        self,
        dataset_name: str,
        thread_limit: Optional[int] = None,
        multithread: bool = True,
        file_format: Optional[str] = None,
    ) -> None:

        # check dataset compatibility
        if dataset_name not in self.supported_datasets:
            raise ValueError(
                f'param:dataset_name must be one of the following: '
                f'{CDSDataAccessor.supported_datasets}'
            )
        self.dataset_name = dataset_name

        # set of multiprocessing threads, CDS enforces a concurrency limit
        cores = multiprocessing.cpu_count()
        if cores > 10:
            cores = 10
        if thread_limit > cores:
            thread_limit = cores
        self.thread_limit = thread_limit

        # init file format
        if file_format is None:
            file_format = 'netcdf'

        if not file_format in list(self.file_format_dict.keys()):
            warnings.warn(
                f'param:file_format={file_format} must be in '
                f'{self.file_format_dict.file_format_dict.keys()}. Defaulting to '
                f'file_format=netcdf'
            )
            file_format = 'netcdf'

        elif file_format == 'grib':
            try:
                import cfgrib
            except ImportError:
                warnings.warn(
                    'No GRIB support -> NetCDF only. Install cfgrib if needed. '
                    'Defaulting to file_format=netcdf'
                )
                file_format = 'netcdf'

        self.file_format = file_format

        # set up CDS client
        try:
            self.client = cdsapi.Client()
        except Exception as e:
            warnings.warn(
                message=(
                    'Follow the instructions on https://cds.climate.copernicus.eu/api-how-to'
                    ' to get set up! \nBasically manually make a .cdsapirc file '
                    '(no extension) where it is looking for it (see exception below).'
                ),
            )
            raise e

        if self.client is None:
            raise ValueError(
                'Must provide a cdsapi.Client() instance to init '
                'param:cdsapi_client'
            )

    def possible_variables(self) -> List[str]:
        if 'single-levels' in self.dataset_name:
            if 'monthly' in self.dataset_name:
                return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_MONTHLY_VARIABLES]
            else:
                return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_HOURLY_VARIABLES]
        elif 'pressure-levels' in self.dataset_name:
            return PRESSURE_LEVEL_VARIABLES
        elif 'land' in self.dataset_name:
            return ERA5_LAND_VARIABLES
        else:
            raise ValueError(f'Cannot return variables. Something went wrong.')

    @staticmethod
    def _get_years_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        return [str(i) for i in range(start_dt.year, stop_dt.year + 1)]

    @staticmethod
    def _get_months_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        if len(range(start_dt.year, stop_dt.year + 1)) > 1:
            return ['{0:0=2d}'.format(m) for m in range(1, 13)]
        else:
            months = [m for m in range(start_dt.month, stop_dt.month + 1)]
            return ['{0:0=2d}'.format(m) for m in months if m <= 12]

    @staticmethod
    def _get_days_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        if len(range(start_dt.month, stop_dt.month + 1)) > 1:
            return ['{0:0=2d}'.format(d) for d in range(1, 32)]
        else:
            days = [d for d in range(start_dt.day, stop_dt.day + 1)]
            return ['{0:0=2d}'.format(d) for d in days if d <= 31]

    def _get_hours_list(
        self,
        hours_step: int = 1,
        specific_hours: Optional[List[int]] = None,
    ) -> List[str]:
        if specific_hours is None:
            if hours_step not in self.valid_hour_steps:
                raise ValueError(
                    f'param:hours_time_step must be one of the following: '
                    f'{self.valid_hour_steps}'
                )
            specific_hours = list(range(0, 24, hours_step))

        else:
            i_len = len(specific_hours)
            specific_hours = [
                i for i in specific_hours if (i < 24) and (i >= 0)
            ]
            delta = i_len - len(specific_hours)
            if delta > 0:
                warnings.warn(
                    f'Not all param:specific hours were < 24, and >=0. '
                    f'{delta} have been ignored'
                )
            del i_len, delta

        return ['{0:0=2d}:00'.format(h) for h in specific_hours]

    def _get_time_dicts(
        self,
        start_dt: datetime,
        end_dt: datetime,
        hours_step: Optional[int] = None,
        specific_hours: Optional[List[int]] = None,
    ) -> List[Dict[str, List[str]]]:
        """Gets monthly time dictionaries for batch processing"""
        # store time_dicts for each API call
        time_dicts = []

        # get list of years
        years = self._get_years_list(start_dt, end_dt)

        for year in years:
            if int(year) == start_dt.year:
                s_dt = start_dt
            else:
                s_dt = pd.to_datetime(f'1/1/{year}')
            if int(year) == end_dt.year:
                e_dt = end_dt
            else:
                e_dt = pd.to_datetime(f'12/31/{year}')

            # get list of months
            months = self._get_months_list(s_dt, e_dt)

            for month in months:
                time_dict = {}
                time_dict['year'] = [year]
                time_dict['month'] = [month]

                if start_dt.year == int(year) and start_dt.month == int(month):
                    s_dt = start_dt
                else:
                    s_dt = pd.to_datetime(f'{month}/1/{year}')
                if end_dt.year == int(year) and end_dt.month == int(month):
                    e_dt = end_dt
                else:
                    e_dt = (
                        pd.to_datetime(
                            f'{int(month) + 1}/1/{year}') - timedelta(days=1)
                    )

                time_dict['day'] = self._get_days_list(s_dt, e_dt)
                time_dicts.append(time_dict)

        # add hours if necessary to each time dict
        if hours_step is not None or specific_hours is not None:
            hours = self._get_hours_list(hours_step, specific_hours)
            for i, time_dict in enumerate(time_dicts):
                time_dicts[i]['time'] = hours
        return time_dicts

    def _get_api_response(
        self,
        input_dict: InputDict,
    ) -> Tuple[int, xr.Dataset]:
        """Separated out as a function to support multithreading"""
        # set up temporary file output
        temp_file = Path(
            tempfile.TemporaryFile(
                dir=Path.cwd(),
                prefix='era5_hourly_data',
                suffix=self.file_format_dict[self.file_format],
            ).name
        ).name

        # remove index from input dict
        index = input_dict.pop('index')

        # get the data
        output = self.client.retrieve(
            self.dataset_name,
            input_dict,
            temp_file,
        )

        # open dataset in xarray
        with urlopen(output.location) as output:
            return (index, xr.open_dataset(output.read()))

    def get_data(
        self,
        variables: Union[str, List[str]],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        hours_step: int = 1,  # TODO: deal with optionality here
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Main data getter function.

        NOTE: CDS multithreading is best handled across time, but total
            observations limits must be considered.
        """
        # make time dict w/ CDS API formatting
        time_dicts = self._get_time_dicts(
            start_dt,
            end_dt,
            hours_step=hours_step,
            specific_hours=specific_hours,
        )

        # make a dictionary to store all data
        all_data_dict = {}

        for variable in variables:
            logging.info(f'Getting {variable} from CDS API')
            # store futures
            var_dict = {}
            input_dicts = []
            for i, time_dict in enumerate(time_dicts):
                input_dicts.append(dict(
                    time_dict,
                    **{
                        'product_type': 'reanalysis',
                        'variable': variable,
                        'format': self.file_format,
                        'grid': [0.25, 0.25],
                        'area': [
                            bbox['south'],
                            bbox['west'],
                            bbox['north'],
                            bbox['east'],
                        ],
                        'index': i,
                    }
                ))
            with ThreadPoolExecutor(max_workers=self.thread_limit) as executor:
                futures = {
                    executor.submit(self._get_api_response, arg): arg for arg in input_dicts
                }
                for future in as_completed(futures):
                    try:
                        index, ds = future.result()
                        var_dict[index] = ds
                    except Exception as e:
                        logging.warning(
                            f'Exception hit!: {e}'
                        )

            # reconstruct each variable into a DataArray
            keys = list(var_dict.keys())
            keys.sort()
            datasets = []
            for key in keys:
                datasets.append(var_dict[key])
            all_data_dict[variable] = xr.concat(datasets, dim='time')
            all_data_dict[variable]['name'] = variable

        return all_data_dict

    def unlock_and_clean(
        output_dict: Dict[str, Dict[str, xr.Dataset]],
    ) -> None:
        """Cleans out the temp files"""

        # unlock files
        for var_dict in output_dict.values():
            for ds in var_dict.values():
                ds.close()

        # delete temp files
        temp_files = []
        for path in Path.cwd().iterdir():
            if 'era5_hourly_data' in path.name:
                temp_files.append(path)

        for t_file in temp_files:
            try:
                t_file.unlink()
            except PermissionError:
                warnings.warn(
                    message=f'Could not delete temp file {t_file}',
                )

    def get_era5_hourly_point_data(
        self,
        variables_dict: Dict[str, str],
        coords_dict: Optional[Dict[str, Tuple[float, float]]] = None,
        file_format: str = 'netcdf',
    ) -> Dict[str, Dict[str, xr.Dataset]]:
        """DEPRECATED JUST FOR REFERENCE"""

        # make a list to store the output datasets
        out_datasets = {}

        # prep request dictionary
        time_dict = self.make_hourly_time_dict()

        # verify file_format
        if file_format not in list(self.file_format_dict.keys()):
            raise ValueError(
                f'param:file_format must be in {self.file_format_dict.keys()}!'
            )

        for station_id, coords in coords_dict.items():
            out_datasets[station_id] = {}

            for variable in list(variables_dict.keys()):
                long, lat = coords
                area = [lat-0.5, long-0.5, lat+0.5, long+0.5]

                input_dict = dict(
                    time_dict,
                    **{
                        'product_type': 'reanalysis',
                        'variable': variable,
                        'format': file_format,
                        'grid': [1.0, 1.0],
                        'area': area,
                    }
                )

                # set up temporary file output
                temp_file = Path(
                    tempfile.TemporaryFile(
                        dir=Path.cwd(),
                        prefix='era5_hourly_data',
                        suffix=self.file_format_dict[file_format],
                    ).name
                ).name

                # get the data
                output = self.client.retrieve(
                    'reanalysis-era5-single-levels',
                    input_dict,
                    temp_file,
                )

                # open dataset in xarray
                with urlopen(output.location) as output:
                    out_datasets[station_id][variable] = xr.open_dataset(
                        output.read(),
                    )

        return out_datasets


class ERA5DataAccessor:

    def __init__(
        self,
        dataset_name: str,
        multithread: bool = True,
        use_dask: bool = False,
        dask_client_kwargs: Optional[dict] = None,
        use_cds_only: bool = False,
        file_format: str = 'netcdf',
        **kwargs,
    ) -> None:
        """Accessor class for ERA5 data. Uses both AWS and CDS endpoints.

        Note that for most datasets only a single class is necessary,
            however due to reading from the ERA5 AWS bucket being vastly faster
            than CDS API calls, we mixed the two methods and use this class as a
            wrapper of both.

        Arguments:
            :param dataset_name: Name of an ERA5 dataset.
            :param multithread: Whether to multithread data fetching calls.
            :param use_dask: Whether to use dask for multithreading.
            :param dask_client_kwargs: Dask client kwargs for init.
            :param use_cds_only: Controls whether to only use CDSDataAccessor.
                NOTE: This is a kwarg not in DataAccessor.pull_data() arguments.
            :param file_format: Controls temp files format saved by CDSDataAccessor.
                NOTE: This is a kwarg not in DataAccessor.pull_data() arguments.
        """
        # init multithreading
        self.multithread = multithread
        self.cores = int(multiprocessing.cpu_count())
        self.use_dask = use_dask

        if self.use_dask:
            from dask.distributed import Client
            dask_client = Client(kwargs=dask_client_kwargs)

        # set file format (checking it is handled in CDSDataAccessor)
        self.file_format = file_format

        # bring in dataset name
        verify_dataset(dataset_name)
        self.dataset_name = dataset_name

        # control multithreading
        self.multithread = multithread
        self.cores = int(multiprocessing.cpu_count())

        # see if we can attempt to use aws
        # TODO: pull file_format from the kwargs instead of just the default
        self.cds_data_accessor = CDSDataAccessor(
            dataset_name=self.dataset_name,
            thread_limit=self.cores,
            multithread=self.multithread,
            file_format=file_format,
        )

        if self.dataset_name in AWSDataAccessor.supported_datasets and not use_cds_only:
            self.use_aws = True
            self.aws_data_accessor = AWSDataAccessor(
                dataset_name=self.dataset_name,
                thread_limit=self.cores,
                multithread=self.multithread,
            )
        else:
            self.use_aws = False

        # warn users about non compatible variables
        self.possible_variables = self.cds_data_accessor.possible_variables()
        # if len(cant_add_variables) > 0:
        #    warnings.warn(
        #        f'variables {cant_add_variables} are not valid for param:'
        #        f'dataset_name={self.dataset_name}, param:dataset_source='
        #        f'{self.dataset_source}.\nPrint DataAccessor.'
        #        f'possible_variables to see all valid variables for '
        #        f'the current dataset name/source combo!'
        #    )
        #    del cant_add_variables

    @property
    def dataset_accessors(self) -> Dict[str, object]:
        return {
            'AWS': AWSDataAccessor,
            'CDS': ERA5DataAccessor,
        }

    def get_data(
        self,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        hours_step: int = 1,  # TODO: deal with optionality here
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """Gathers the desired variables for ones time/space AOI.

        NOTE: Data is gathered from ERA5DataAccessor.dataset_name.

        Arguments:
            :param variables: List of variables to access.
            :param start_dt: Datetime to start at (inclusive),
            :param end_dt: Datetime to stop at (inclusive).
            :param bbox: Dictionary with bounding box EPSG 4326 lat/longs.
            :param hours_step: Changes the default hours time step from 1.
                NOTE: This argument is not accessible from DataAccessor!
            :param specific_hours: Only pull data from a specific hour(s).
                NOTE: This argument is not accessible from DataAccessor!

        Returns:
            A xarray Dataset with all desired data.
        """
        # TODO: make compatible with monthly requests
        # TODO: think about how we want to allow access to hours_step and specific_hours

        # map variables to underlying data accessor
        accessor_variables_mapper = {}
        cant_add_variables = []

        # see which variables can be fetched from AWS
        aws_variables = []
        cds_variables = []
        if self.use_aws:
            aws_variables = [
                i for i in variables if i in self.aws_data_accessor.possible_variables()
            ]
            accessor_variables_mapper[self.aws_data_accessor] = aws_variables

        # map remaining variables to CDS
        if not len(aws_variables) == len(variables):
            for var in [i for i in variables if i not in aws_variables]:
                if var in self.cds_data_accessor.possible_variables():
                    cds_variables.append(var)
                else:
                    cant_add_variables.append(var)
            accessor_variables_mapper[self.cds_data_accessor] = cds_variables

        # init a dictionary to store outputs
        datasets_dict = {}
        for var in [v for v in variables if v not in cant_add_variables]:
            datasets_dict[var] = None

        # get the data from both sources
        for accessor, vars in accessor_variables_mapper.items():
            try:
                datasets_dict.update(
                    accessor.get_data(
                        vars,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bbox=bbox,
                        hours_step=hours_step,
                        specific_hours=specific_hours,
                    ),
                )
            except TypeError:
                warnings.warn(
                    f'{accessor.__str__()} returned None'
                )

        # remove and warn about NoneType responses
        del_keys = []
        for k, v in datasets_dict.items():
            if v is None:
                del_keys.append(k)
        if len(del_keys) > 0:
            warnings.warn(
                f'Could not get data for the following variables: {del_keys}'
            )
            for k in del_keys:
                datasets_dict.pop(k)

        # if just one variable, return the dataset
        if len(datasets_dict) == 1:
            return datasets_dict[list(datasets_dict.keys())[0]]
        elif len(datasets_dict) == 0:
            raise ValueError(
                f'A problem occurred! No data was returned.'
            )

        # combine the data from multiple sources
        try:
            return xr.merge(list(datasets_dict.values()))
        # allow data to be salvaged if merging fails
        except Exception as e:
            warnings.warn(
                f'There was an issue combing ERA5 data from multiple sources. '
                f'All requested data is within the returned list of xr.Datasets. '
                f'Exception: {e}'
            )
            return datasets_dict
