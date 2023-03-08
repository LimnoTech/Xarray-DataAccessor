from read_into_xarray.data_accessor import BoundingBoxDict
from read_into_xarray.multi_threading import DaskClass, get_multithread
from typing import (
    Dict,
    Tuple,
    List,
    Union,
    Optional,
    TypedDict,
)
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
import fsspec
from urllib.request import urlopen
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pandas as pd
import multiprocessing
import logging
import warnings
import rioxarray
import numpy as np

# stop xarray from importing cfgrib by default since it only works well with linux
import xarray as xr
import os
os.environ["XARRAY_GH_CONFIGURE_WITH_CFGRIB"] = "0"


class AWSRequestDict(TypedDict):
    variable: str
    aws_endpoint: str
    index: int
    bbox: BoundingBoxDict


class AWSResponseDict(AWSRequestDict):
    dataset: xr.Dataset


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

    @staticmethod
    def _rename_dimensions(dataset: xr.Dataset) -> xr.Dataset:
        time_dim = [d for d in list(dataset.coords) if 'time' in d]
        if len(time_dim) > 1:
            warnings.warn(
                f'Multiple time dimensions found! {time_dim}. '
                'Changing the first to time. This may cascade errors.'
            )
        rename_dict = {
            'lon': 'longitude',
            'lat': 'latitude',
        }
        time_dim = time_dim[0]
        if time_dim != 'time':
            rename_dict[time_dim] = 'time'
        return dataset.rename(rename_dict)

    @staticmethod
    def _crop_aws_data(
        ds: xr.Dataset,
        bbox: BoundingBoxDict,
    ) -> xr.Dataset:
        """Crops AWS ERA5 to the nearest 0.25 resolution to align with CDS output"""
        # make sure we have inclusive bounds at 0.25
        std_bbox = ERA5DataAccessor._standardize_bbox(bbox)
        x_bounds = np.array([std_bbox['west'], std_bbox['east']])
        y_bounds = np.array([std_bbox['south'], std_bbox['north']])

        # find closest x, y values in the data
        nearest_x_idxs = np.abs(
            ds.lon.values - x_bounds.reshape(-1, 1)
        ).argmin(axis=1)
        nearest_y_idxs = np.abs(
            ds.lat.values - y_bounds.reshape(-1, 1)
        ).argmin(axis=1)

        # return the sliced dataset
        return ds.isel(
            {
                'lon': slice(nearest_x_idxs.min(), nearest_x_idxs.max() + 1),
                'lat': slice(nearest_y_idxs.min(), nearest_y_idxs.max() + 1),
            }
        ).copy()

    def _get_requests_dicts(
        self,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
    ) -> List[AWSRequestDict]:

        endpoint_prefix = r's3://era5-pds'

        # init list to store request tuples
        aws_request_dicts = []

        # iterate over variables and create requests
        for variable in variables:
            count = 0
            if variable in self.aws_variable_mapping.keys():
                endpoint_suffix = f'{self.aws_variable_mapping[variable]}.nc'
            elif variable in self.aws_variable_mapping.values():
                endpoint_suffix = f'{variable}.nc'
            else:
                warnings.warn(
                    message=(
                        f'Variable={variable} cannot be found for AWS'
                    ),
                )
            for year in range(start_dt.year, end_dt.year + 1):
                m_i, m_f = 1, 13
                if year == start_dt.year:
                    m_i = start_dt.month
                if year == end_dt.year:
                    m_f = end_dt.month + 1
                for m in range(m_i, m_f):
                    m = str(m).zfill(2)
                    aws_request_dicts.append(
                        {
                            'variable': variable,
                            'aws_endpoint': f'{endpoint_prefix}/{year}/{m}/data/{endpoint_suffix}',
                            'index': count,
                            'bbox': bbox,
                        }
                    )
                    count += 1
        return aws_request_dicts

    def _get_aws_data(
        self,
        aws_request_dict: AWSRequestDict,
    ) -> AWSResponseDict:
        # read data from the s3 bucket
        endpoint = aws_request_dict['aws_endpoint']
        logging.info(f'Accessing endpoint: {endpoint}')
        aws_request_dict['dataset'] = xr.open_dataset(
            fsspec.open(endpoint).open(),
            engine='h5netcdf',
        )

        # adjust to switch to standard lat/lon
        aws_request_dict['dataset']['lon'] = aws_request_dict['dataset']['lon'] - 180

        aws_request_dict['dataset'] = self._crop_aws_data(
            aws_request_dict['dataset'],
            aws_request_dict['bbox'],
        )

        # rename time dimension if necessary
        aws_request_dict['dataset'] = self._rename_dimensions(
            aws_request_dict['dataset'],
        )

        return aws_request_dict

    def get_data(
        self,
        variables: Union[str, List[str]],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        use_dask: bool = False,
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """
        Main data getter function.

        NOTE: AWS multithreading is best handled across months.
        """

        # make a dictionary to store all data
        all_data_dict = {}

        # get a dictionary to store the AWS requests
        if isinstance(variables, str):
            variables = [variables]

        aws_request_dicts = self._get_requests_dicts(
            variables,
            start_dt,
            end_dt,
            bbox,
        )

        # set up multithreading client
        client, as_completed_func = get_multithread(
            use_dask=False,
            n_workers=self.thread_limit,
            threads_per_worker=1,
            processes=True,
            close_existing_client=False,
        )

        # init dictionary to store data sorted by variable
        data_dicts = {}
        for variable in variables:
            data_dicts[variable] = {}

        # init a dictionary to store outputs
        all_data_dict = {}

        with client as executor:
            logging.info(
                f'Reading {len(aws_request_dicts)} data months from S3 bucket.')
            # map all our input dicts to our data getter function
            futures = {
                executor.submit(self._get_aws_data, arg): arg for arg in aws_request_dicts
            }
            # add outputs to data_dicts
            for future in as_completed_func(futures):
                try:
                    aws_response_dict = future.result()
                    var = aws_response_dict['variable']
                    index = aws_response_dict['index']
                    ds = aws_response_dict['dataset']
                    data_dicts[var][index] = ds
                except Exception as e:
                    logging.warning(
                        f'Exception hit!: {e}'
                    )

        for variable in variables:
            var_dict = data_dicts[variable]

            # reconstruct each variable into a DataArray
            keys = list(var_dict.keys())
            keys.sort()
            datasets = []
            for key in keys:
                datasets.append(var_dict[key])

            # only concat if necessary
            if len(datasets) > 1:
                ds = xr.concat(
                    datasets,
                    dim='time',
                )
            else:
                ds = datasets[0]

            # crop by time
            ds = ds.sel(
                {
                    'time': slice(start_dt, end_dt),
                },
            )

            all_data_dict[variable] = ds.rename(
                {list(ds.data_vars)[0]: variable},
            )

        return all_data_dict


class CDSDataAccessor:
    InputDict = Dict[str, int]
    file_format_dict = {
        'netcdf': '.nc',
        'grib': '.grib',
    }

    supported_datasets = DATASET_NAMES
    institution = 'ECMWF'

    def __init__(
        self,
        dataset_name: str,
        thread_limit: Optional[int] = None,
        file_format: Optional[str] = None,
    ) -> None:

        # import the API (done here to avoid multiprocessing imports)
        import cdsapi

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
        specific_hours: Optional[List[int]] = None,
    ) -> List[str]:
        if specific_hours is None:
            specific_hours = list(range(0, 24, 1))

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

                days = self._get_days_list(s_dt, e_dt)

                # get weekly chunks of days for API calls
                days_lists = []
                sub_days = []
                for day in days:
                    if len(sub_days) < 7:
                        sub_days.append(day)
                    else:
                        days_lists.append(sub_days)
                        sub_days = [day]
                if len(sub_days) > 0:
                    days_lists[-1].extend(sub_days)

                # add to list of dictionaries
                for day_list in days_lists:
                    time_dict = {}
                    time_dict['year'] = [year]
                    time_dict['month'] = [month]
                    time_dict['day'] = day_list
                    time_dicts.append(time_dict)

        # add hours if necessary to each time dict
        if specific_hours is not None:
            hours = self._get_hours_list(specific_hours)
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
                prefix='temp_data',
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
        use_dask: bool = False,
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
            specific_hours=specific_hours,
        )

        client, as_completed_func = get_multithread(
            use_dask=use_dask,
            n_workers=self.thread_limit,
            threads_per_worker=1,
            processes=True,
            close_existing_client=False,
        )

        # make a dictionary to store all data
        all_data_dict = {}

        with client as executor:
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

                # only send 10 at once to prevent being throttled
                batches = list(range((len(input_dicts) // 10) + 1))
                batches = [b + 1 for b in batches]
                for j, batch in enumerate(batches):
                    if j == 0:
                        start_i = 0
                    else:
                        start_i = int(batches[j - 1] * 10)
                    if batch == batches[-1]:
                        end_i = None
                    else:
                        end_i = int(batch * 10)

                    futures = {
                        executor.submit(self._get_api_response, arg): arg for arg in input_dicts[start_i:end_i]
                    }
                    for future in as_completed_func(futures):
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
                ds = xr.concat(
                    datasets,
                    dim='time',
                )

                # crop by time
                ds = ds.sel(
                    {
                        'time': slice(start_dt, end_dt),
                    },
                )

                all_data_dict[variable] = ds.rename(
                    {list(ds.data_vars)[0]: variable},
                )

        return all_data_dict


class ERA5DataAccessor:

    def __init__(
        self,
        dataset_name: str,
        use_dask: bool = False,
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
            :param use_dask: Whether to use dask for multithreading.
            :param dask_client_kwargs: Dask client kwargs for init.
            :param use_cds_only: Controls whether to only use CDSDataAccessor.
                NOTE: This is a kwarg not in DataAccessor.pull_data() arguments.
            :param file_format: Controls temp files format saved by CDSDataAccessor.
                NOTE: This is a kwarg not in DataAccessor.pull_data() arguments.
        """
        # init multithreading
        self.cores = int(multiprocessing.cpu_count())
        self.use_dask = use_dask

        # set file format (checking it is handled in CDSDataAccessor)
        self.file_format = file_format

        # bring in dataset name
        verify_dataset(dataset_name)
        self.dataset_name = dataset_name

        # control multithreading
        self.cores = int(multiprocessing.cpu_count())

        # see if we can attempt to use aws
        # TODO: pull file_format from the kwargs instead of just the default
        self.cds_data_accessor = CDSDataAccessor(
            dataset_name=self.dataset_name,
            thread_limit=self.cores,
            file_format=file_format,
        )
        self.all_possible_variables = self.cds_data_accessor.possible_variables()

        if self.dataset_name in AWSDataAccessor.supported_datasets and not use_cds_only:
            self.use_aws = True
            self.aws_data_accessor = AWSDataAccessor(
                dataset_name=self.dataset_name,
                thread_limit=self.cores,
            )
            self.all_possible_variables = list(
                set(self.all_possible_variables +
                    self.aws_data_accessor.possible_variables())
            )
        else:
            self.use_aws = False

    @property
    def dataset_accessors(self) -> Dict[str, object]:
        return {
            'AWS': AWSDataAccessor,
            'CDS': ERA5DataAccessor,
        }

    @staticmethod
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

    @staticmethod
    def _round_to_nearest(
        number: float,
        shift_up: bool,
    ) -> float:
        """Rounds number to nearest 0.25. Either shifts up or down."""
        num = round((number * 4)) / 4
        if shift_up:
            if num < number:
                num += 0.25
        else:
            if num > number:
                num -= 0.25
        return num

    @staticmethod
    def _standardize_bbox(
        bbox: BoundingBoxDict,
    ) -> BoundingBoxDict:
        """
        Converts a bbox to the nearest 0.25 increments.

        NOTE: This is used when combining CDS and AWS data since CDS API returns
            data at 0.25 increments starting/stopping at the exact bbox coords.
            In contrast, AWS returns all data in 0.25 increments for the whole 
            globe and is converted via AWSDataAccessor._crop_aws_data().
        """
        out_bbox = {}

        def _round_to_nearest(
            number: float,
            shift_up: bool,
        ) -> float:
            """Rounds number to nearest 0.25. Either shifts up or down."""
            num = round((number * 4)) / 4
            if shift_up:
                if num < number:
                    num += 0.25
            else:
                if num > number:
                    num -= 0.25
            return num

        out_bbox['west'] = _round_to_nearest(bbox['west'], shift_up=False)
        out_bbox['south'] = _round_to_nearest(bbox['south'], shift_up=False)
        out_bbox['east'] = _round_to_nearest(bbox['east'], shift_up=True)
        out_bbox['north'] = _round_to_nearest(bbox['north'], shift_up=True)

        return out_bbox

    def _write_attrs(
        self,
        cds_variables: List[str],
        aws_variables: List[str],
    ) -> dict:
        """Used to write aligned attributes to all datasets before merging"""
        attrs = {}

        # write attrs storing top level data source info
        attrs['dataset_name'] = self.dataset_name
        attrs['institution'] = CDSDataAccessor.institution

        # write attrs storing projection info
        attrs['x_dim'] = 'longitude'
        attrs['y_dim'] = 'latitude'
        attrs['EPSG'] = 4326

        # write attrs storing time dimension info
        attrs['time_step'] = 'hourly'

        # write attrs storing variable source info
        attrs['Data from CDS API'] = cds_variables
        attrs['Data from Planet OS AWS S3 bucket'] = aws_variables
        return attrs

    def get_data(
        self,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """Gathers the desired variables for ones time/space AOI.

        NOTE: Data is gathered from ERA5DataAccessor.dataset_name.

        Arguments:
            :param variables: List of variables to access.
            :param start_dt: Datetime to start at (inclusive),
            :param end_dt: Datetime to stop at (exclusive).
            :param bbox: Dictionary with bounding box EPSG 4326 lat/longs.
            :param specific_hours: Only pull data from a specific hour(s).
                NOTE: This argument is not accessible from DataAccessor!

        Returns:
            A xarray Dataset with all desired data.
        """
        # TODO: make compatible with monthly requests
        # TODO: think about how we want to allow access specific_hours

        # adjust end_dt to make the range exclusive
        end_dt = end_dt - timedelta(hours=1)

        # map variables to underlying data accessor
        accessor_variables_mapper = {}
        cant_add_variables = []

        # prep bbox
        bbox = self._prep_small_bbox(bbox)

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

        # if using both CDS and AWS, convert bbox to 0.25 increments
        if len(cds_variables) > 0 and len(aws_variables) > 0:
            bbox = self._standardize_bbox(bbox)

        # get the data from both sources
        for accessor, vars in accessor_variables_mapper.items():
            try:
                datasets_dict.update(
                    accessor.get_data(
                        vars,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        bbox=bbox,
                        use_dask=self.use_dask,
                        specific_hours=specific_hours,
                    ),
                )
            except TypeError as e:
                warnings.warn(
                    f'{accessor.__str__()} returned None. Exception: {e}'
                )

        # make an updates attributes dictionary
        attrs_dict = self._write_attrs(
            cds_variables,
            aws_variables,
        )

        # remove and warn about NoneType responses
        del_keys = []
        for k, v in datasets_dict.items():
            if v is None:
                del_keys.append(k)
            else:
                # write new metadata
                datasets_dict[k].attrs = attrs_dict

        if len(del_keys) > 0:
            warnings.warn(
                f'Could not get data for the following variables: {del_keys}'
            )
            for k in del_keys:
                datasets_dict.pop(k)

        # if just one variable, return the dataset
        if len(datasets_dict) == 0:
            raise ValueError(
                f'A problem occurred! No data was returned.'
            )

        # combine the data from multiple sources
        try:
            logging.info('Combining all variable Datasets...')
            out_ds = xr.merge(list(datasets_dict.values())).rio.write_crs(4326)
            logging.info('Done! Returning combined dataset.')
            return out_ds

        # allow data to be salvaged if merging fails
        except Exception as e:
            warnings.warn(
                f'There was an issue combing ERA5 data from multiple sources. '
                f'All requested data is within the returned list of xr.Datasets. '
                f'Exception: {e}'
            )
            return datasets_dict
