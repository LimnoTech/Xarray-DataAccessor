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
from numbers import Number
from xarray_data_accessor.multi_threading import (
    get_multithread,
)
from xarray_data_accessor.data_accessors.shared_functions import (
    combine_variables,
    apply_kwargs,
    write_crs,
    crop_data,
    crop_time_dimension,
)
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
)
from xarray_data_accessor.data_accessors.base import (
    DataAccessorBase,
    AttrsDict,
)
from xarray_data_accessor.data_accessors.factory import (
    DataAccessorProduct,
)

# TODO: remove probably but keep for now
CDS_TO_AWS_NAMES_CROSSWALK = {
    '10m_u_component_of_wind': 'eastward_wind_at_10_metres',
    '10m_v_component_of_wind': 'northward_wind_at_10_metres',
    '100m_u_component_of_wind': 'eastward_wind_at_100_metres',
    '100m_v_component_of_wind': 'northward_wind_at_100_metres',
    '2m_dewpoint_temperature': 'dew_point_temperature_at_2_metres',
    '2m_temperature': 'air_temperature_at_2_metres',
    'maximum_2m_temperature_since_previous_post_processing': 'air_temperature_at_2_metres_1hour_Maximum',
    'minimum_2m_temperature_since_previous_post_processing': 'air_temperature_at_2_metres_1hour_Minimum',
    'mean_sea_level_pressure': 'air_pressure_at_mean_sea_level',
    'mean_wave_period': 'sea_surface_wave_mean_period',
    'mean_wave_direction': 'sea_surface_wave_from_direction',
    'significant_height_of_total_swell': 'significant_height_of_wind_and_swell_waves',
    'snow_density': 'snow_density',
    'snow_depth': 'lwe_thickness_of_surface_snow_amount',
    'surface_pressure': 'surface_air_pressure',
    'surface_solar_radiation_downwards': 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
    'total_precipitation': 'precipitation_amount_1hour_Accumulation',
}


class AWSKwargsDict(TypedDict):
    """kwargs for AWSDataAccessor get_data() method."""
    use_dask: bool
    thread_limit: int


class AWSRequestDict(TypedDict):
    """Request dictionary for accessing the S3 bucket."""
    variable: str
    aws_endpoint: str
    index: int
    bbox: BoundingBoxDict


class AWSResponseDict(AWSRequestDict):
    """Dictionary of xarray datasets read from the S3 bucket."""
    dataset: xr.Dataset


@DataAccessorProduct
class AWSDataAccessor(DataAccessorBase):
    """Data accessor for ERA5 data from AWS Open Data Registry."""

    institution = 'ECMWF via Planet OS'

    def __init__(self) -> None:

        # store last accessed dataset name
        self.dataset_name: str = None

        # set default kwargs
        self.thread_limit: int = multiprocessing.cpu_count() - 1
        self.use_dask: bool = True

    @classmethod
    def supported_datasets(cls) -> List[str]:
        """Returns all datasets that can be accessed."""""
        return [
            'reanalysis-era5-single-levels',
        ]

    @classmethod
    def dataset_variables(cls) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        return {
            cls.supported_datasets()[0]: [
                'eastward_wind_at_10_metres',
                'northward_wind_at_10_metres',
                'eastward_wind_at_100_metres',
                'northward_wind_at_100_metres',
                'dew_point_temperature_at_2_metres',
                'air_temperature_at_2_metres',
                'air_temperature_at_2_metres_1hour_Maximum',
                'air_temperature_at_2_metres_1hour_Minimum',
                'air_pressure_at_mean_sea_level',
                'sea_surface_wave_mean_period',
                'sea_surface_wave_from_direction',
                'significant_height_of_wind_and_swell_waves',
                'snow_density',
                'lwe_thickness_of_surface_snow_amount',
                'surface_air_pressure',
                'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
                'precipitation_amount_1hour_Accumulation',
            ],
        }

    @property
    def attrs_dict(self) -> AttrsDict:
        """Used to write aligned attributes to all datasets before merging"""
        attrs = {}

        # write attrs storing top level data source info
        attrs['dataset_name'] = self.dataset_name
        attrs['institution'] = self.institution

        # write attrs storing projection info
        attrs['x_dim'] = 'longitude'
        attrs['y_dim'] = 'latitude'
        attrs['EPSG'] = 4326

        # write attrs storing time dimension info
        attrs['time_zone'] = 'UTC'
        attrs['time_step'] = 'hourly'
        return attrs

    def _parse_kwargs(
        self,
        kwargs_dict: AWSKwargsDict,
    ) -> None:
        """Parses kwargs and sets class attributes"""

        # apply the kwargs
        apply_kwargs(
            accessor_object=self,
            accessor_kwargs_dict=AWSKwargsDict,
            kwargs_dict=kwargs_dict,
        )

    def get_data(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        bbox: BoundingBoxDict,
        start_dt: datetime,
        end_dt: datetime,
        **kwargs,
    ) -> xr.Dataset:
        """
        Main data getter function.

        NOTE: AWS multithreading is best handled across months.
        """
        # check dataset compatibility
        if dataset_name not in self.supported_datasets():
            raise ValueError(
                f'param:dataset_name must be one of the following: '
                f'{self.supported_datasets()}',
            )
        else:
            self.dataset_name = dataset_name

        # parse kwargs
        self._parse_kwargs(kwargs)

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
            use_dask=self.use_dask,
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
                f'Reading {len(aws_request_dicts)} data months from S3 bucket.',
            )
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
                    ds = crop_time_dimension(
                        ds,
                        start_dt,
                        end_dt,
                    )
                    data_dicts[var][index] = ds
                except Exception as e:
                    logging.warning(
                        f'Exception hit!: {e}',
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

            # drop every variable except the one we want (works for aws)
            for var in ds.data_vars:
                if var != variable:
                    ds = ds.drop_vars(var)

            all_data_dict[variable] = ds.rename(
                {list(ds.data_vars)[0]: variable},
            )

            all_data_dict[variable] = write_crs(
                all_data_dict[variable],
                known_epsg=4326,
            )

        # return the combined data
        return combine_variables(
            all_data_dict,
            self.attrs_dict,
        )

    # AWS specific methods #####################################################

    @staticmethod
    def _rename_dimensions(dataset: xr.Dataset) -> xr.Dataset:
        time_dim = [d for d in list(dataset.coords) if 'time' in d]
        if len(time_dim) > 1:
            warnings.warn(
                f'Multiple time dimensions found! {time_dim}. '
                'Changing the first to time. This may cascade errors.',
            )
        rename_dict = {
            'lon': 'longitude',
            'lat': 'latitude',
        }
        time_dim = time_dim[0]
        if time_dim != 'time':
            rename_dict[time_dim] = 'time'
        return dataset.rename(rename_dict)

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
            if variable in self.dataset_variables()[self.dataset_name]:
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

                    # create the request/access dictionary
                    endpoint = f'{endpoint_prefix}/{year}/{m}/data/{endpoint_suffix}'
                    aws_request_dicts.append(
                        {
                            'variable': variable,
                            'aws_endpoint': endpoint,
                            'index': count,
                            'bbox': bbox,
                        },
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

        aws_request_dict['dataset'].attrs['EPSG'] = 4326
        aws_request_dict['dataset'] = crop_data(
            aws_request_dict['dataset'],
            aws_request_dict['bbox'],
            xy_dim_names=('lon', 'lat'),
        )

        # rename time dimension if necessary
        aws_request_dict['dataset'] = self._rename_dimensions(
            aws_request_dict['dataset'],
        )

        return aws_request_dict
