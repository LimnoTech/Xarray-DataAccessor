import xarray as xr
import cdsapi
import geopandas as gpd
import warnings
import multiprocessing
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen
from prep_query import CDSQueryFormatter
from typing import (
    Dict,
    Tuple,
    List,
    Union,
    Optional,
)
"""
THIS IS THE MOVE: https://cds.climate.copernicus.eu/toolbox/doc/api.html
NOTE: Data is accessed for the whole globe, then cropped on their end.
    Therefore timesteps x variables are the main rate limiting step.
Live CDS API: https://cds.climate.copernicus.eu/live/limits
"""
# TODO: Handle query size
# TODO: Use dask to parallelize API calls? use dask compute


class CDSDataAccessor(CDSQueryFormatter):

    # CDS enforces a concurrency limit
    thread_limit = 10
    file_format_dict = {
        'netcdf': '.nc',
        'grib': '.grib',
    }

    # TODO: make a typed dict for this in CDSQueryFormatter
    InputDict = Dict[str, int]

    def __init__(
        self,
        file_format: str = 'netcdf',
        multithread: bool = True,
        use_dask: bool = True,
        dask_client_kwargs: Optional[dict] = None,
    ) -> None:

        # init multithreading
        self.multithread = multithread
        self.cores = int(multiprocessing.cpu_count())
        self.use_dask = use_dask

        if self.use_dask:
            from dask.distributed import Client
            dask_client = Client(kwargs=dask_client_kwargs)

        # init file format
        if not file_format in list(CDSDataAccessor.file_format_dict.keys()):
            warnings.warn(
                f'param:file_format={file_format} must be in '
                f'{CDSDataAccessor.file_format_dict.keys()}. Defaulting to '
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

    def make_time_dict(
        self,
        start_dt: datetime,
        end_dt: datetime,
        hours_step: Optional[int] = None,
        specific_hours: Optional[List[int]] = None,
    ) -> Dict[str, Union[str, List[str], List[float]]]:
        time_dict = {
            'day': self.get_days_list(start_dt, end_dt),
            'month': self.get_months_list(start_dt, end_dt),
            'year': self.get_years_list(start_dt, end_dt),
        }
        if hours_step is not None or specific_hours is not None:
            time_dict['time'] = self.get_hours_list(hours_step, specific_hours)
        return time_dict

    def _get_api_response(
        self,
        input_dict: InputDict,
    ) -> xr.Dataset:
        """Separated out as a function to support multithreading"""
        raise NotImplementedError

    def get_data(
        self,
        dataset_name: str,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: Dict[str, float],
        hours_step: Optional[int] = None,
        specific_hours: Optional[List[int]] = None,
    ) -> xr.Dataset:
        # TODO: make compatible with monthly requests
        # TODO:
        time_dict = self.make_time_dict(
            start_dt,
            end_dt,
            hours_step=hours_step,
            specific_hours=specific_hours,
        )
        # TODO: set up multithreading using futures
        # TODO: iterate by time step since it pulls the globe every time
        raise NotImplementedError

    def get_era5_hourly_point_data(
        self,
        variables_dict: Dict[str, str],
        coords_dict: Optional[Dict[str, Tuple[float, float]]] = None,
        file_format: str = 'netcdf',
    ) -> Dict[str, Dict[str, xr.Dataset]]:

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


def unlock_and_clean(output_dict: Dict[str, Dict[str, xr.Dataset]]) -> None:
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
