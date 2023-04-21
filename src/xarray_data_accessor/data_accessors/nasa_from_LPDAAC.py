import logging
import warnings
import requests
import io
import multiprocessing
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
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
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
)
from xarray_data_accessor.data_accessors.base import (
    DataAccessorBase,
    AttrsDict,
)
from xarray_data_accessor.data_accessors.shared_functions import (
    apply_kwargs,
    write_crs,
    crop_data,
)
from xarray_data_accessor.data_accessors.factory import (
    DataAccessorProduct,
)
from xarray_data_accessor.data_accessors.nasa_info import (
    LPDAAC_VARIABLES,
    LPDAAC_TIME_DIMS,
)


class AuthorizationDict(TypedDict):
    """TypedDict for NASA authorization credentials"""
    username: str
    password: str


class NASAKwargsDict(TypedDict):
    """TypedDict for NASA data accessor kwargs"""
    authorization: AuthorizationDict
    use_dask: bool
    thread_limit: int


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

        # set kwarg defaults
        self.use_dask = True
        self.thread_limit = multiprocessing.cpu_count() - 1

    @classmethod
    def supported_datasets(cls) -> List[str]:
        """Returns all datasets that can be accessed."""""
        return list(LPDAAC_VARIABLES.keys())

    @classmethod
    def dataset_variables(cls) -> Dict[str, List[str]]:
        """Returns all variables for each dataset that can be accessed."""
        return LPDAAC_VARIABLES

    @property
    def attrs_dict(self) -> AttrsDict:
        """Used to write aligned attributes to all datasets before merging"""
        attrs = {}

        # write attrs storing top level data source info
        attrs['dataset_name'] = self.dataset_name
        attrs['institution'] = self.institution

        # write attrs storing projection info
        attrs['x_dim'] = 'lon'
        attrs['y_dim'] = 'lat'
        attrs['EPSG'] = 4326

        # write attrs storing time dimension info
        attrs['time_step'] = LPDAAC_TIME_DIMS[self.dataset_name]
        return attrs

    def _parse_kwargs(
        self,
        kwargs_dict: NASAKwargsDict,
    ) -> None:
        """Parses kwargs for NASA data accessor"""
        # if kwargs are buried, dig them out
        while 'kwargs' in kwargs_dict.keys():
            kwargs_dict = kwargs_dict['kwargs']

        # make sure authentication credentials are set!
        credential_error = ValueError(
            'NASA data accessors require EarthData login credentials. '
            'Please provide them as the argument "authorization" using '
            'a dictionary with keys: "username" and "password".'
        )
        if 'authorization' not in kwargs_dict.keys():
            raise credential_error
        elif not isinstance(kwargs_dict['authorization'], dict):
            raise credential_error
        elif 'username' not in kwargs_dict['authorization'].keys():
            raise credential_error
        elif 'password' not in kwargs_dict['authorization'].keys():
            raise credential_error
        else:
            # pop out authorization credentials and apply them
            authorization = kwargs_dict.pop('authorization')
            self._username = authorization['username']
            self._password = authorization['password']

        # apply the kwargs
        apply_kwargs(
            accessor_object=self,
            accessor_kwargs_dict=NASAKwargsDict,
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

        # remember last dataset name for caching
        self.dataset_name = dataset_name

        # parse kwargs (check for EarthData login credentials)
        self._parse_kwargs(kwargs)

        # search for granules
        granules = self._find_matching_granules(
            dataset_name=dataset_name,
            bbox=bbox,
            start_dt=start_dt,
            end_dt=end_dt,
            variables=variables,
        )

        # TODO: make this stuff work!
        # download granules in parallel
        if len(granules) == 0:
            raise ValueError('No granules found for given search parameters.')

        # if there is only one granule, just download it
        elif len(granules) == 1:
            xarray_dataset = self._get_granule_functions[self.dataset_name](
                granules[0],
            )

        # if there are multiple granules, download them in parallel
        else:
            client, as_completed_func = get_multithread(
                use_dask=self.use_dask,
                n_workers=self.thread_limit,
                threads_per_worker=1,
                processes=False,
                close_existing_client=False,
            )

            with client as executor:
                futures = {
                    executor.submit(
                        self._get_granule_functions[self.dataset_name],
                        granule,
                    ): granule for granule in granules
                }
                data = []
                for future in as_completed_func(futures):
                    try:
                        data.append(future.result())
                    except Exception as e:
                        logging.warning(
                            f'Exception hit!: {e}'
                        )

            # close client
            client.close()

            # merge datasets
            xarray_dataset = xr.merge(data)

        # add attributes
        xarray_dataset.attrs = self.attrs_dict

        # write CRS (different datasets may come in different projections)
        xarray_dataset = write_crs(xarray_dataset)

        # crop data to bbox
        xarray_dataset = crop_data(
            ds=xarray_dataset,
            bbox=bbox,
        )

        return xarray_dataset

    # CDS API specific methods #################################################
    @property
    def _request_session(self) -> requests.Session:
        """Returns a requests session and authentication tuple"""
        if self._session is None:
            self._session = requests.Session()
            if self._auth_tuple is None:
                self._auth_tuple = (self._username, self._password)
            self._session.auth = self._auth_tuple
        return self._session

    @property
    def _get_link_identifier(self) -> Dict[str, str]:
        """Returns a dictionary of link identifiers for each dataset

        This is used to parse CRM Search JSON responses.
        """
        return {
            'NASADEM_NC': '.nc',
            'NASADEM_SC': '.zip',
            'GLanCE30': '.tif',
        }

    @property
    def _get_granule_functions(self) -> Dict[str, Callable[[GranuleDict], xr.Dataset]]:
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
            if self._get_link_identifier[self.dataset_name] in link['title']:
                granule_dict['granule_url'] = link['href']
                break

        granule_dict['dataset_id'] = entry_dict['dataset_id']
        granule_dict['data_center'] = entry_dict['data_center']

        # TODO: make sure this works for all datasets
        bbox = [float(i) for i in entry_dict['boxes'][0].split(' ')]
        granule_dict['bbox'] = {
            'west': bbox[1],
            'south': bbox[0],
            'east': bbox[3],
            'north': bbox[2],
        }
        granule_dict['start_date'] = datetime.strptime(
            entry_dict['time_start'],
            '%Y-%m-%dT%H:%M:%S.%fZ',
        )
        granule_dict['end_date'] = datetime.strptime(
            entry_dict['time_end'],
            '%Y-%m-%dT%H:%M:%S.%fZ',
        )
        return granule_dict

    def _find_matching_granules(
        self,
        dataset_name: str,
        bbox: BoundingBoxDict,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        variables: Optional[List[str]] = None,
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

        # get temporal query url string (for datasets that need it)
        if LPDAAC_TIME_DIMS[dataset_name]:
            start_dt_str = self._format_datetime_string(start_dt)
            end_dt_str = self._format_datetime_string(end_dt)
        else:
            start_dt_str = ''
            end_dt_str = ''

        if start_dt_str != '' or end_dt_str != '':
            # &options[temporal][exclude_boundary]=true' -> this causes issues with non-time dependent datasets
            temporal_str = f'&temporal={start_dt_str},{end_dt_str}'
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
        response1 = self._request_session.get(granule_dict['granule_url'])
        response2 = self._request_session.get(
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
