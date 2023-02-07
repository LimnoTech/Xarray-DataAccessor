import warnings
import pandas as pd
import xarray as xr
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime

# control what is possible based on dependencies
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    import rioxarray
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False

PossibleAOIInputs = Union[
    str,
    Path,
    Tuple[float, float],
    List[Tuple[float, float]],
    xr.DataArray,
]


def get_datetime(input_date: Union[str, datetime, int]) -> datetime:
    if isinstance(input_date, datetime):
        return input_date

    # assume int is a year
    elif isinstance(input_date, int):
        if input_date not in list(range(1950, datetime.now().year + 1)):
            raise ValueError(
                f'integer start/end date input={input_date} is not a valid year.'
            )
        return pd.to_datetime(f'{input_date}-01-01')

    elif isinstance(input_date, str):
        return pd.to_datetime(input_date)
    else:
        raise ValueError(
            f'start/end date input={input_date} is invalid.'
        )


class CDSQueryFormatter:
    valid_hour_steps = [1, 3, 6, 9, 12]

    @staticmethod
    def get_years_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        return [str(i) for i in range(start_dt.year, stop_dt.year + 1)]

    @staticmethod
    def get_months_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        if len(range(start_dt.year, stop_dt.year + 1)) > 1:
            return ['{0:0=2d}'.format(m) for m in range(1, 13)]
        else:
            months = [m for m in range(start_dt.month, stop_dt.month + 1)]
            return ['{0:0=2d}'.format(m) for m in months if m <= 12]

    @staticmethod
    def get_days_list(
        start_dt: datetime,
        stop_dt: datetime,
    ) -> List[str]:
        if len(range(start_dt.month, stop_dt.month + 1)) > 1:
            return ['{0:0=2d}'.format(d) for d in range(1, 32)]
        else:
            days = [d for d in range(start_dt.day, stop_dt.day + 1)]
            return ['{0:0=2d}'.format(d) for d in days if d <= 31]

    @classmethod
    def get_hours_list(
        cls,
        hours_step: int = 1,
        specific_hours: Optional[List[int]] = None,
    ) -> List[str]:
        if hours_step != 1:
            if hours_step not in CDSQueryFormatter.valid_hour_steps:
                raise ValueError(
                    f'param:hours_time_step must be one of the following: '
                    f'{CDSQueryFormatter.valid_hour_steps}'
                )
            specific_hours = list(range(0, 24, hours_step))

        elif specific_hours is not None:
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


def _get_coords_dict() -> Dict[Union[str, int], Tuple[float, float]]:
    if coordinates is not None:
        if isinstance(coordinates, tuple):
            coordinates = [coordinates]

        coords_dict = {}
        for i, coords in enumerate(coordinates):
            coords_dict[f'station{i}'] = coords

    elif csv_of_coords is not None:
        if isinstance(csv_of_coords, str):
            csv_of_coords = Path(csv_of_coords)
        if csv_of_coords.exists():
            coords_df = pd.read_csv(
                csv_of_coords,
                index_col='station_id',
            )
            coords_dict = {}
            try:
                for i, row in coords_df.iterrows():
                    coords_dict[row['station_id']] = (
                        row['lon'],
                        row['lat']
                    )
            except KeyError:
                raise KeyError(
                    'Make sure your csv has columns station_id, lon, and lat!'
                )
        else:
            raise FileNotFoundError(
                f'{csv_of_coords} is not a valid .csv path!'
            )
    return coords_dict


def _bbox_from_coords():
    # TODO: add buffer so the edge isn't the exact coordinate
    raise NotImplementedError


def _bbox_from_shp():
    if not HAS_GEOPANDAS:
        raise ImportError(
            f'To create a bounding box from shapefile you need geopandas installed!'
        )
    raise NotImplementedError


def _bbox_from_raster():
    if not HAS_RIOXARRAY:
        raise ImportError(
            f'To create a bounding box from raster you need rioxarray installed!'
        )
    raise NotImplementedError


def get_bounding_box(
    aoi_input: PossibleAOIInputs,
    aoi_input_type: str,
):
    raise NotImplementedError
