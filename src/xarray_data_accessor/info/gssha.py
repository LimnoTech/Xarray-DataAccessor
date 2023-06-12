import dataclasses
import numpy as np
from typing import (
    Literal,
    Dict,
)

PrecipitationType = Literal[
    'GAGES',
    'RADAR',
    'RATES',
    'ACCUM',
]


@dataclasses.dataclass
class HMETVariableInfo:
    units: str
    ascii_file_name: str
    wes_index: int
    dtype: np.dtype
    nodata_value: float | int
    alias: str | None = None


HMETVariables: Dict[str, HMETVariableInfo] = {}

HMETVariables['Barometric Pressure'] = HMETVariableInfo(
    units='in Hg',
    ascii_file_name='Pres',
    wes_index=0,
    dtype=np.dtype('float32'),
    nodata_value=99.999,
    alias='Atmospheric Pressure',
)
HMETVariables['Relative Humidity'] = HMETVariableInfo(
    units='%',
    ascii_file_name='RIHm',
    wes_index=1,
    dtype=np.dtype('int'),
    nodata_value=999,
)
HMETVariables['Total Sky Cover'] = HMETVariableInfo(
    units='%',
    ascii_file_name='Clod',
    wes_index=2,
    dtype=np.dtype('int'),
    nodata_value=999,
    alias='Cloud Cover',
)
HMETVariables['Wind Speed'] = HMETVariableInfo(
    units='kts',
    ascii_file_name='WndS',
    wes_index=3,
    dtype=np.dtype('int'),
    nodata_value=999,
)
HMETVariables['Dry Bulb Temperature'] = HMETVariableInfo(
    units='F',
    ascii_file_name='Temp',
    wes_index=4,
    dtype=np.dtype('int'),
    nodata_value=999,
    alias='Temperature',
)
HMETVariables['Direct Radiation'] = HMETVariableInfo(
    units='W*h/m^2',
    ascii_file_name='Drad',
    wes_index=5,
    dtype=np.dtype('float32'),
    nodata_value=999.99,
)
HMETVariables['Global Radiation'] = HMETVariableInfo(
    units='W*h/m^2',
    ascii_file_name='Grad',
    wes_index=6,
    dtype=np.dtype('float32'),
    nodata_value=999.99,
)
