import logging
import warnings
import requests
import io
import multiprocessing
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
    Any,
    Callable,
    TypedDict,
)
from numbers import Number
from xarray_data_accessor.multi_threading import (
    get_multithread,
)
from xarray_data_accessor.data_accessors.shared_functions import (
    combine_variables,
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


class NASAKwargsDict(TypedDict):
    """TypedDict for NASA data accessor kwargs"""
    pass


class GranuleDict(TypedDict):
    """TypedDict for NASA granule metadata"""
    granule_id: str
    granule_url: str
    dataset_id: str
    data_center: str
    bbox: BoundingBoxDict
    start_date: datetime
    end_date: datetime


@DataAccessorProduct
class NASA_LPDAAC_Accessor(DataAccessorBase):
    """Retrieves data from NASA/USGS's LP DAAC Data Pool."""

    institution = 'NASA/USGS LP DAAC'

    def __init__(self) -> None:

        # set up authentication credentials
        self._username: str = None
        self._password: str = None
        self._session: requests.Session = None
        self._auth_tuple: Tuple[str, str] = None

        # store the last dataset name grabbed for caching
        self.dataset_name: str = None

    @classmethod
    def supported_datasets(cls) -> List[str]:
        """Returns all datasets that can be accessed."""""
        # TODO: start with elevation products
        # datsets to check out below:
        # https://lpdaac.usgs.gov/products/eco3etptjplv001/
        # https://lpdaac.usgs.gov/products/glchmtv001/
        return [
            'NASADEM_NC',  # netcdf w/ single product
            'NASADEM_SC',  # RAW w/ multiple sub products
            'GLanCE30',  # geotiff with multiple sub products
        ]

    @classmethod
    def dataset_variables(cls) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        out_dict = {}
        # TODO: get this working for NASA
        return out_dict

    @property
    def attrs_dict(self) -> AttrsDict:
        """Used to write aligned attributes to all datasets before merging"""
        attrs = {}
        # TODO: update for NASA
        raise NotImplementedError
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
        kwargs_dict: NASAKwargsDict,
    ) -> None:
        """Parses kwargs for NASA data accessor"""
        # TODO: start by making sure username and password are set
        if 'use_dask' in kwargs_dict.keys():
            use_dask = kwargs_dict['use_dask']
            if isinstance(use_dask, bool):
                self.use_dask = use_dask
            else:
                warnings.warn(
                    'kwarg:use_dask must be a boolean. '
                    'Defaulting to True.'
                )
        else:
            self.use_dask = True

        if 'thread_limit' in kwargs_dict.keys():
            thread_limit = kwargs_dict['thread_limit']
            if isinstance(thread_limit, int):
                self.thread_limit = thread_limit
            else:
                warnings.warn(
                    'kwarg:thread_limit must be an integer. '
                    'Defaulting to number of cores.'
                )
        else:
            self.thread_limit = multiprocessing.cpu_count()

    def get_data(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        start_dt: datetime,
        end_dt: datetime,
        bbox: BoundingBoxDict,
        **kwargs,
    ) -> xr.Dataset:

        # remember last dataset name for caching
        self.dataset_name = dataset_name

        # parse kwargs (check for EarthData login credentials)
        self._parse_kwargs(kwargs)

        # search for granules
        granules = self._search_granules(
            dataset_name,
            start_dt,
            end_dt,
            bbox,
        )

        # TODO: make this stuff work!
        # download granules in parallel
        if len(granules) == 0:
            raise ValueError('No granules found for given search parameters.')

        # if there is only one granule, just download it
        elif len(granules) == 1:
            xarray_dataset = self.get_granule_functions[self.dataset_name](
                granules[0],
            )

        # if there are multiple granules, download them in parallel
        else:
            client, as_completed_func = get_multithread(
                use_dask=self.use_dask,
                n_workers=self.thread_limit,
                threads_per_worker=1,
                processes=True,
                close_existing_client=False,
            )
            futures = client.map(
                self.get_granule_functions[self.dataset_name],
                granules,
            )

            # wait for all futures to complete
            data = []
            for future in as_completed_func(futures):
                data.append(future.result())

            # close client
            client.close()

            # merge datasets
            xarray_dataset = xr.merge(data)

        # add attributes
        xarray_dataset.attrs = self.attrs_dict

        raise NotImplementedError

    # CDS API specific methods #################################################
    @property
    def request_session(self) -> requests.Session:
        """Returns a requests session and authentication tuple"""
        if self._session is None:
            self._session = requests.Session()
            if self._auth_tuple is None:
                self._auth_tuple = (self._username, self._password)
            self._session.auth = self.auth_tuple
        return self._session

    @property
    def link_identifier(self) -> Dict[str, str]:
        """Returns a dictionary of link identifiers for each dataset

        This is used to parse CRM Search JSON responses.
        """
        return {
            'NASADEM_NC': '.nc',
            'NASADEM_SC': '.zip',
            'GLanCE30': 'LC.tif',
        }

    @property
    def get_granule_functions(self) -> Dict[str, Callable[[GranuleDict], xr.Dataset]]:
        """Returns a dictionary of functions to open data for each dataset"""
        return {
            'NASADEM_NC': self._get_netcdf_granule,
            'NASADEM_SC': self._get_raw_granule,
            'GLanCE30': self._get_tiff_granule,
        }

    @staticmethod
    def _format_datetime_string(
        dt: datetime,
    ) -> str:
        """Formats datetime string for CRM Search API

        See: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html#temporal-range-searches
        """
        if dt is None:
            return ''
        return dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

    def _get_granule_dict(
        self,
        entry_dict: Dict[str, Any],
    ) -> GranuleDict:
        """Parses granule metadata from CRM Search API response"""
        granule_dict = {}
        granule_dict['granule_id'] = entry_dict['producer_granule_id']

        # find the correct granule link using the link identifier
        for link in entry_dict['links']:
            if self.link_identifier[self.dataset_name] in link['title']:
                granule_dict['granule_url'] = link['href']
                break

        granule_dict['dataset_id'] = entry_dict['dataset_id']
        granule_dict['data_center'] = entry_dict['data_center']
        granule_dict['bbox'] = {
            'west': entry_dict['bounding_box'][0],
            'south': entry_dict['bounding_box'][1],
            'east': entry_dict['bounding_box'][2],
            'north': entry_dict['bounding_box'][3],
        }
        granule_dict['start_date'] = datetime.strptime(
            entry_dict['time_start'],
            '%Y-%m-%dT%H:%M:%SZ',
        )
        granule_dict['end_date'] = datetime.strptime(
            entry_dict['time_end'],
            '%Y-%m-%dT%H:%M:%SZ',
        )
        return granule_dict

    @staticmethod
    def _find_matching_granules(
        self,
        dataset_name: str,
        bbox: BoundingBoxDict,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> List[GranuleDict]:
        """Uses CRM Search API to find all granules matching the given parameters.

        Returns:
            A list of granule endpoints.
        """
        # get base url for CRM search API (json responses)
        crm_base_url = 'https://cmr.earthdata.nasa.gov/search/granules.json?'

        # get dataset short name for CRM search API
        dataset_str = f'short_name={dataset_name}'

        # get bbox query url string
        bbox_str = f'&bounding_box[]={bbox["west"]},{bbox["south"]},{bbox["east"]},{bbox["north"]}'

        # get temporal query url string
        start_dt_str = self._format_datetime_string(start_dt)
        end_dt_str = self._format_datetime_string(end_dt)
        if start_dt_str != '' or end_dt_str != '':
            temporal_str = f'&temporal\[\]={start_dt_str},{end_dt_str}&[temporal][exclude_boundary]=true'
        else:
            temporal_str = ''

        # get granule urls
        search_url = crm_base_url + dataset_str + bbox_str + temporal_str
        response = requests.get(search_url)

        # return list of granule endpoint urls
        if response.ok:
            granule_links = dict(response.json())['feed']['entry']
            return [self._get_granule_dict(granule) for granule in granule_links]
        else:
            raise ValueError(
                f'Error retrieving searching granules! See response text: {response.text}'
            )

    def _request_granule(
        self,
        granule_dict: GranuleDict,
    ) -> requests.Response:
        """Gets a response for a single granule from the NASA Data Pool."""
        # send auth then data request
        response1 = self.request_session.get(granule_dict['granule_url'])
        response2 = self.request_session.get(
            response1.url,
            auth=self._auth_tuple,
        )

        if response2.ok:
            # return as dataset
            del response1
            return response2
        else:
            raise ValueError(
                f'Error retrieving granule! See response text: {response2.text}'
            )

    def _get_netcdf_granule(
        self,
        granule_dict: GranuleDict,
    ) -> xr.Dataset:
        """Retrieves a single NetCDF granule from the NASA Data Pool."""
        response = self._request_granule(granule_dict)
        return xr.open_dataset(
            io.BytesIO(response.content),
            engine='h5netcdf',
        )

    def _get_tiff_granule(
        self,
        granule_dict: GranuleDict,
    ) -> xr.Dataset:
        """Retrieves a single GeoTIFF granule from the NASA Data Pool."""
        raise NotImplementedError

    def _get_raw_granule(
        self,
        granule_dict: GranuleDict,
    ) -> xr.Dataset:
        """Retrieves a single RAW granule from the NASA Data Pool."""
        raise NotImplementedError

    def _parse_zip_contents(
        self,
        response: requests.Response,
    ) -> Any:
        """Parses the contents of a zip file keeping only variables of interest."""
        # TODO: come back to this later.
        raise NotImplementedError

    def _concat_granules() -> xr.Dataset:
        """Concatenates all granules into a single dataset."""
        raise NotImplementedError