import warnings
import logging
import pandas as pd
import numpy as np
import xarray as xr
from xarray_data_accessor import utility_functions
from xarray_data_accessor.shared_types import (
    BoundingBoxDict,
    TimeInput,
)
from xarray_data_accessor.data_converters.base import (
    DataConverterBase,
)
from xarray_data_accessor.data_converters.factory import (
    DataConversionFactory,
)
from xarray_data_accessor.info.gssha import (
    PrecipitationType,
    HMETVariables,
)
from datetime import datetime
from pathlib import Path
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    Optional,
    TypedDict,
)


class EventIntervals(TypedDict):
    name: str
    start: datetime
    end: datetime


@ DataConversionFactory.register
class ConvertToGSSHA(DataConverterBase):
    """Converts xarray datasets to GSSHA input files."""

    @ staticmethod
    def _get_file_path(
        file_dir: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
        file_suffix: Optional[str] = None,
    ) -> Path:
        """Generate a valid output file path."""

        # make sure the file directory exists
        if not file_dir:
            file_dir = Path.cwd()
        else:
            file_dir = Path(file_dir)
        if not file_dir.exists():
            raise FileNotFoundError(
                f'File directory {file_dir} does not exist!',
            )

        # make sure the file name is valid
        if not file_name:
            file_name = 'gssha_input'
            warnings.warn(
                f'No file name was provided! Using default file name {file_name}.',
            )
        if not isinstance(file_name, str):
            raise TypeError(
                f'param:file_name must be a string! Not {type(file_name)}.',
            )

        # get the file suffix
        if '.asc' in file_name:
            file_name = file_name.replace('.asc', '')
        if not file_suffix:
            file_suffix = '.asc'
        if not isinstance(file_suffix, str):
            raise TypeError(
                f'param:file_suffix must be a string! Not {type(file_suffix)}.',
            )
        if not file_suffix.startswith('.'):
            file_suffix = f'.{file_suffix}'

        # return the file path
        return Path(file_dir / f'{file_name}{file_suffix}')

    @ staticmethod
    def _write_ascii_file(
        text_content: str,
        file_path: Path,
    ) -> None:
        """Writes the text content to the file path."""
        # write the text content to the file path
        with open(
            file_path,
            'w',
            encoding='ascii',
        ) as file:
            file.write(text_content)

        # validate the ASCII file
        if not file_path.exists():
            raise FileNotFoundError(f'File {file_path} was not created!')

        with open(
            file_path,
            'r',
            encoding='ascii',
            errors='ignore',
        ) as file:
            try:
                file.read().encode('ascii')
            except UnicodeDecodeError:
                raise UnicodeDecodeError(
                    f'Something went wrong - File {file_path} is not a valid ASCII file.'
                )

    @ staticmethod
    def _write_precip_coords(
        easting: pd.Series,
        northing: pd.Series,
    ) -> str:
        """Writes the coordinate lines of a precipitation ASCII file.

        NOTE: This function assumes that the easting and northing coordinates
        are in the same projection and correspond to the same time step!

        Arguments:
            easting: A pandas series of easting coordinates.
            northing: A pandas series of northing coordinates.

        Returns: A string of the precipitation coordinates in ASCII format.
            Output format:
                NRGAG 2
                COORD 204555.0  4751268.0 "Center of precipitation pixel #1"
                COORD 205642.0  4750491.0 "Center of precipitation pixel #2"
        """
        # zip the coordinates
        coordinates = zip(easting.to_list(), northing.to_list())

        # get the number of "gages"
        num_gages = len(easting)

        output = f'NRGAG {num_gages}\n'
        for i, (easting, northing) in enumerate(coordinates):
            output += f'COORD {easting} {northing} "Center of precipitation pixel #{i+1}"\n'
        return output

    @classmethod
    def make_gssha_precipitation_input(
        cls,
        xarray_dataset: xr.Dataset,
        precipitation_variable: str,
        precipitation_type: Optional[PrecipitationType] = None,
        event_intervals: Optional[List[EventIntervals]] = None,
        file_dir: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
        file_suffix: Optional[str] = None,
    ) -> Path:
        """Creates a GSSHA precipitation input file from an xarray dataset.

        For more information on the GSSHA precipitation input file format, see:
            https://www.gsshawiki.com/Precipitation_Input

        Arguments:
            xarray_dataset: The xarray dataset to convert.
            precipitation_variable: The name of the precipitation variable.
            precipitation_type: The type of precipitation (i.e., GAGE, RADAR,...).
            event_intervals: A list of event intervals. Each interval must be
                a dict of form {event_name: str, start: datetime, end: datetime}.
            file_dir: The directory to save the file to.
            file_name: The name of the file to save.
            file_suffix: The file suffix to use.

        Returns:
            The path of the output precipitation ASCII input file.
        """
        # get a file path
        if not file_suffix:
            file_suffix = '.gag'
        file_path: Path = cls._get_file_path(
            file_dir=file_dir,
            file_name=file_name,
            file_suffix=file_suffix,
        )

        # check precipitation type
        if not precipitation_type:
            warnings.warn(
                'No precipitation type provided. '
                'Defaulting to GAGE precipitation.'
            )
            precipitation_type: PrecipitationType = 'GAGE'

        # get the x and y coordinate names
        x_dim: str = xarray_dataset.attrs['x_dim']
        y_dim: str = xarray_dataset.attrs['y_dim']

        # get coordinates
        # TODO: figure out projection and units
        data_df = (
            xarray_dataset[precipitation_variable]
            .to_dataframe()
            .reset_index()
            .sort_values(
                by=[
                    x_dim,
                    'time',
                ],
            )
        )
        ts1: datetime = data_df['time'].unique()[0]
        coordinates_header: str = cls._write_precip_coords(
            easting=data_df.loc[data_df.time == ts1, x_dim],
            northing=data_df.loc[data_df.time == ts1, y_dim],
        )

        # get events
        if not event_intervals:
            event_intervals: List[EventIntervals] = [
                EventIntervals(
                    name='precipitation_event_1',
                    start=xarray_dataset.time.values[0],
                    end=xarray_dataset.time.values[-1],
                ),
            ]

        # write events data to file
        event_strings: List[str] = []
        for event in event_intervals:
            event_string: str = f'EVENT {event["name"]}\n'
            sub_df = data_df.loc[data_df.time.between(
                event['start'],
                event['end'],
            )]
            event_string += f'NRPDS {len(sub_df)}\n'
            event_string += coordinates_header
            for t in sub_df.time.unique():
                time_str: str = t.strftime('%Y %m %d %H %M')
                data_string = (
                    sub_df.loc[sub_df.time == t][precipitation_variable]
                    .T
                    .to_string(index=False, header=False)
                )
                data_string = ' '.join(data_string.split())
                event_string += f'{precipitation_type} {time_str} {data_string}\n'
            event_strings.append(event_string)

        # join the events
        ascii_text: str = '\n'.join(event_strings)
        del event_strings

        # write the ASCII file
        cls._write_ascii_file(
            text_content=ascii_text,
            file_path=file_path,
        )
        logging.info(f'Precipitation ASCII file saved @ {file_path}.')
        return file_path

    @ classmethod
    def make_grass_ascii_input(
        cls,
        xarray_dataset: xr.Dataset,
        variable: str,
        hmet_variable: Optional[str] = None,
        start_time: Optional[TimeInput] = None,
        end_time: Optional[TimeInput] = None,
        file_dir: Optional[Union[str, Path]] = None,
        file_suffix: Optional[str] = None,
    ) -> Path:
        """Creates a GRASS ASCII input file from an xarray dataset.

        For more information on the GRASS ASCII input file format, see:
            https://grasswiki.osgeo.org/wiki/GRASS_ASCII_raster_format

        Arguments:
            xarray_dataset: The xarray dataset to convert.
            variable: The name of the variable to convert.
            hmet_variable: The name of the GSSHA HMET variable.
                See: https://www.gsshawiki.com/Continuous:Hydrometeorological_Data
            start_time: The start time to trim the data to.
            end_time: The end time to trim the data to.
            file_dir: The directory to save the file to.
                NOTE: The file name is automatically generated.
            file_suffix: The file suffix to use.

        Returns:
            The path of the directory containing all GRASS ASCII input files.
        """

        # make sure we are using an appropriate variable name
        if variable not in xarray_dataset:
            raise KeyError(
                f'Variable {variable} not found in xarray dataset!',
            )
        if hmet_variable not in HMETVariables.keys():
            raise KeyError(
                f'Variable {hmet_variable} not found in HMETVariables! '
                f'Please specify a valid HMET variable from list: {HMETVariables.keys()}',
            )

        file_name: str = HMETVariables[hmet_variable].ascii_file_name

        # trim to time range if necessary
        if start_time or end_time:
            start_dt = utility_functions._get_datetime(start_time)
            end_dt = utility_functions._get_datetime(end_time)
            xarray_dataset = xarray_dataset.time.where(
                (xarray_dataset.time.dt >=
                 start_dt and xarray_dataset.time.dt <= end_dt),
                drop=True,
            ).copy()

        # make GRASS ASCII header with bounds
        grass_header: str = ''

        x_dim = xarray_dataset.attrs['x_dim']
        y_dim = xarray_dataset.attrs['y_dim']

        bbox = BoundingBoxDict(
            north=np.max(xarray_dataset[y_dim].values),
            south=np.min(xarray_dataset[y_dim].values),
            east=np.max(xarray_dataset[x_dim].values),
            west=np.min(xarray_dataset[x_dim].values),
        )

        for direction in ['north', 'south', 'east', 'west']:
            grass_header += f'{direction}: {bbox[direction]}\n'

        grass_header += f'rows: {len(xarray_dataset[y_dim].values)}\n'
        grass_header += f'cols: {len(xarray_dataset[x_dim].values)}\n'

        # iterate over time steps
        for time in xarray_dataset.time.values:
            data_str = (
                np.array2string(
                    xarray_dataset[variable].sel(time=time).values,
                    max_line_width=100000000,
                    formatter={'float': lambda x: str(x)},
                    separator=' ',
                )
                .replace('  ', '')
                .replace('[', '')
                .replace(']', '')
                .replace('\n ', '\n')
            )

            ascii_text = grass_header + data_str

            # get a file path (YYYYMMDDHH_TYPE.asc format)
            timestamp: str = pd.to_datetime(time).strftime('%Y%m%d%H')
            file_path: Path = cls._get_file_path(
                file_dir=file_dir,
                file_name=f'{timestamp}_{file_name}',
                file_suffix=file_suffix,
            )

            # write the ASCII file
            cls._write_ascii_file(
                text_content=ascii_text,
                file_path=file_path,
            )
        logging.info(f'GRASS ASCII files saved to {file_dir}.')
        return file_path.parent

    @ classmethod
    def get_conversion_functions(
        cls,
    ) -> Dict[str, DataConverterBase.ConversionFunctionType]:
        """Returns a dictionary of conversion functions."""
        return {
            cls.make_gssha_precipitation_input.__name__: cls.make_gssha_precipitation_input,
            cls.make_grass_ascii_input.__name__: cls.make_grass_ascii_input,
        }
