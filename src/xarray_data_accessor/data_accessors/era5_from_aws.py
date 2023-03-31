"""Data accessor for ERA5 data from AWS Open Data Registry.

Info: https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
"""
import logging
import warnings
import multiprocessing
import fsspec
import xarray as xr
import numpy as np
from datetime import datetime
from typing import (
    Union,
    List,
    Dict,
    Optional,
    TypedDict,
)
from xarray_data_accessor.multi_threading import (
    get_multithread,
)
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
)


class AWSRequestDict(TypedDict):
    variable: str
    aws_endpoint: str
    index: int
    bbox: BoundingBoxDict


class AWSResponseDict(AWSRequestDict):
    dataset: xr.Dataset


class AWSDataAccessor:
    """Data accessor for ERA5 data from AWS Open Data Registry."""

    def __init__(
        self,
        thread_limit: Optional[int] = None,
    ) -> None:

        # get cores for multiprocessing
        if thread_limit is None:
            thread_limit = multiprocessing.cpu_count
        self.thread_limit = thread_limit

    @property
    def supported_datasets(self) -> List[str]:
        """Returns all datasets that can be accessed."""""
        return [
            'reanalysis-era5-single-levels',
            'reanalysis-era5-single-levels-preliminary-back-extension',
            'reanalysis-era5-single-levels-monthly-means',
            'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
            'reanalysis-era5-pressure-levels',
            'reanalysis-era5-pressure-levels-monthly-means',
            'reanalysis-era5-pressure-levels-preliminary-back-extension',
            'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
            'reanalysis-era5-land',
            'reanalysis-era5-land-monthly-means',
        ]

    # TODO: fill this out!
    @property
    def dataset_variables(self) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        raise NotImplementedError

    def _write_attrs(self) -> None:
        """Used to write aligned attributes to all sub datasets before merging"""
        raise NotImplementedError

    def get_data(
        self,
        dataset_name: str,
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
        # check dataset compatibility
        if dataset_name not in self.supported_datasets:
            raise ValueError(
                f'param:dataset_name must be one of the following: '
                f'{self.supported_datasets}'
            )

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
            ).copy(deep=True)

            all_data_dict[variable] = ds.rename(
                {list(ds.data_vars)[0]: variable},
            )

        return all_data_dict

    def possible_variables(self) -> List:
        # replace with dataset_variables
        raise NotImplementedError
        out_list = []
        for k, v in self.aws_variable_mapping.items():
            out_list.append(k)
            out_list.append(v)
        return out_list

    # AWS specific methods #####################################################
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
        x_bounds = np.array([bbox['west'], bbox['east']])
        y_bounds = np.array([bbox['south'], bbox['north']])

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

        # set filesystem and endpooint prefix
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
