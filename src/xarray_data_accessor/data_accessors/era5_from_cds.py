import logging
import warnings
import multiprocessing
import tempfile
import cdsapi
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from urllib.request import urlopen
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    Union,
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
from xarray_data_accessor.data_accessors.era5_from_cds_info import (
    SINGLE_LEVEL_VARIABLES,
    MISSING_MONTHLY_VARIABLES,
    MISSING_HOURLY_VARIABLES,
    PRESSURE_LEVEL_VARIABLES,
    ERA5_LAND_VARIABLES,
)


class CDSKwargsDict(TypedDict):
    """kwargs for CDSDataAccessor get_data() method."""
    use_dask: bool
    thread_limit: int
    file_format: str
    specific_hours: List[int]


class CDSInputDict(TypedDict):
    """Input dictionary for CDS API request"""
    product_type: str
    format: str
    variable: str
    year: Union[str, List[str]]
    month: Union[str, List[str]]
    day: Optional[Union[str, List[str]]]
    time: Optional[Union[str, List[str]]]
    area: List[Number]


@DataAccessorProduct
class CDSDataAccessor(DataAccessorBase):
    file_format_dict = {
        'netcdf': '.nc',
        'grib': '.grib',
    }

    institution = 'ECMWF'

    def __init__(self) -> None:

        # set up CDS client
        self._client: cdsapi.Client = None

        # store the last dataset name grabbed for caching
        self.dataset_name: str = None

        # set default kwargs
        cores = multiprocessing.cpu_count() - 1
        if cores > 10:
            cores = 10
        self.thread_limit: int = cores
        self.use_dask: bool = True
        self.file_format: str = 'netcdf'
        self.specific_hours: List[int] = None

    @classmethod
    def supported_datasets(cls) -> List[str]:
        """Returns all datasets that can be accessed."""""
        return [
            'reanalysis-era5-single-levels',
            'reanalysis-era5-single-levels-preliminary-back-extension',
            # 'reanalysis-era5-single-levels-monthly-means',
            # 'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
            # 'reanalysis-era5-pressure-levels',
            # 'reanalysis-era5-pressure-levels-monthly-means',
            # 'reanalysis-era5-pressure-levels-preliminary-back-extension',
            # 'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
            'reanalysis-era5-land',
            # 'reanalysis-era5-land-monthly-means',
        ]

    @classmethod
    def dataset_variables(cls) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        out_dict = {}
        for dataset in cls.supported_datasets():
            out_dict[dataset] = cls._possible_variables(dataset)

        return out_dict

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
        if 'monthly' in self.dataset_name:
            attrs['time_step'] = 'monthly'
        else:
            attrs['time_step'] = 'hourly'
        attrs['time_zone'] = 'UTC'
        return attrs

    def _parse_kwargs(
        self,
        kwargs_dict: CDSKwargsDict,
    ) -> None:
        """Parses kwargs for CDS data accessors"""

        # apply the kwargs
        apply_kwargs(
            accessor_object=self,
            accessor_kwargs_dict=CDSKwargsDict,
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

        NOTE: CDS multithreading is best handled across time, but total
            observations limits must be considered.
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

        # make time dict w/ CDS API formatting
        time_dicts = self._get_time_dicts(
            start_dt,
            end_dt,
            specific_hours=self.specific_hours,
        )

        client, as_completed_func = get_multithread(
            use_dask=self.use_dask,
            n_workers=self.thread_limit,
            threads_per_worker=1,
            processes=True,
            close_existing_client=False,
        )

        # make a dictionary to store all data
        all_data_dict = {}

        with client as executor:
            for variable in variables:
                # check if variable is supported
                if not variable in self.dataset_variables()[self.dataset_name]:
                    warnings.warn(
                        message=(
                            f'Variable={variable} cannot be found for AWS'
                        ),
                    )
                    continue
                else:
                    logging.info(f'Getting {variable} from CDS API')

                # store futures
                var_dict = {}
                input_dicts = []
                for i, time_dict in enumerate(time_dicts):
                    input_dicts.append(
                        dict(
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
                        ),
                    )

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
                        executor.submit(
                            self._get_api_response,
                            arg,
                        ): arg for arg in input_dicts[start_i:end_i]
                    }
                    for future in as_completed_func(futures):
                        try:
                            index, ds = future.result()
                            var_dict[index] = ds
                        except Exception as e:
                            logging.warning(
                                f'Exception hit!: {e}',
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
                ds = crop_time_dimension(
                    ds,
                    start_dt,
                    end_dt,
                )

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

    # CDS API specific methods #################################################
    @property
    def client(self) -> cdsapi.Client:
        """Returns a CDS API client."""
        if self._client is None:
            try:
                self._client = cdsapi.Client()
            except Exception as e:
                warnings.warn(
                    message=(
                        'Follow the instructions on https://cds.climate.copernicus.eu/api-how-to'
                        ' to get set up! \nBasically manually make a .cdsapirc file '
                        '(no extension) where it is looking for it (see exception below).'
                    ),
                )
                raise e
        return self._client

    @staticmethod
    def _possible_variables(
        dataset_name: str,
    ) -> List[str]:
        """Returns all possible variables for a given dataset."""
        if 'single-levels' in dataset_name:
            if 'monthly' in dataset_name:
                return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_MONTHLY_VARIABLES]
            else:
                return [i for i in SINGLE_LEVEL_VARIABLES if i not in MISSING_HOURLY_VARIABLES]
        elif 'pressure-levels' in dataset_name:
            return PRESSURE_LEVEL_VARIABLES
        elif 'land' in dataset_name:
            return ERA5_LAND_VARIABLES
        else:
            raise ValueError(f'Cannot return variables. Something went wrong.')

    @classmethod
    def _verify_file_format(
        cls,
        file_format: str,
    ) -> str:
        if not file_format in list(cls.file_format_dict.keys()):
            warnings.warn(
                f'param:file_format={file_format} must be in '
                f'{cls.file_format_dict.keys()}. Defaulting to '
                f'file_format=netcdf',
            )
            return 'netcdf'

        elif file_format == 'grib':
            try:
                import cfgrib
                return file_format
            except ImportError:
                warnings.warn(
                    'No GRIB support -> NetCDF only. Install cfgrib if needed. '
                    'Defaulting to file_format=netcdf',
                )
                return 'netcdf'

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
                    f'{delta} have been ignored',
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
                            f'{int(month) + 1}/1/{year}',
                        ) - timedelta(days=1)
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
                    if len(days_lists) == 0:
                        days_lists.append(sub_days)
                    days_lists[-1].extend(sub_days)

                # add to list of dictionaries
                for day_list in days_lists:
                    time_dict = {}
                    time_dict['year'] = [year]
                    time_dict['month'] = [month]
                    time_dict['day'] = day_list
                    time_dicts.append(time_dict)

        # add hours if necessary to each time dict
        hours = self._get_hours_list(specific_hours)
        for i, time_dict in enumerate(time_dicts):
            time_dicts[i]['time'] = hours
        return time_dicts

    def _get_api_response(
        self,
        input_dict: CDSInputDict,
    ) -> Tuple[int, xr.Dataset]:
        """Separated out as a function to support multithreading"""
        # set up temporary file output
        temp_file = Path(
            tempfile.TemporaryFile(
                dir=Path.cwd(),
                prefix='temp_data',
                suffix=self.file_format_dict[self.file_format],
            ).name,
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
