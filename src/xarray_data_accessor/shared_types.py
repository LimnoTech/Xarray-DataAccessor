import abc
from typing import (
    List,
    Tuple,
    Dict,
    Union,
    TypedDict,
)
from datetime import datetime
from xarray import DataArray, Dataset
from rasterio.crs import CRS
from geopandas import GeoDataFrame
from pandas import DataFrame
from pathlib import Path

TimeInput = Union[datetime, str, int]
TableInput = Union[str, Path, DataFrame]
ShapefileInput = Union[str, Path, GeoDataFrame]
RasterInput = Union[str, Path, DataArray, Dataset]
CoordsTuple = Tuple[float, float]  # lat/long
ResolutionTuple = Tuple[Union[int, float], Union[int, float]]

PossibleAOIInputs = Union[
    CoordsTuple,
    List[CoordsTuple],
    TableInput,
    ShapefileInput,
    RasterInput,
]


class BoundingBoxDict(TypedDict):
    west: float
    south: float
    east: float
    north: float


class InputDict(TypedDict):
    """Stores all internal inputs to the DataAccessor."""
    dataset_name: str
    aoi_input_type: str
    bounding_box: BoundingBoxDict
    start_datetime: datetime
    end_datetime: datetime
    variables: List[str]
    multithreading: bool


class ResampleDict(TypedDict):
    width: int
    height: int
    resampling_method: str
    crs: CRS
    index: int
